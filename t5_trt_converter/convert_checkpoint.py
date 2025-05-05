import argparse
import configparser
import copy
import json
import logging
import os
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Union, cast

import safetensors
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Blip2ForConditionalGeneration,
    MBartForConditionalGeneration,
    Pix2StructForConditionalGeneration,
    T5ForConditionalGeneration,
    VisionEncoderDecoderModel,
)

from .helper import convert_weight_to_dtype, fuse_qkv_one_layer, reshape, split

try:
    from fairseq.models.transformer import TransformerModel
except ImportError:
    TransformerModel = None
    print("fairseq not installed, NMT support disabled.")

# Removed: from tensorrt_llm.functional import LayerNormPositionType, LayerNormType, MLPType
# Removed: from tensorrt_llm.models import PretrainedConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
LOGGER = logging.getLogger(__name__)

HFModelType = Union[
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel,
    Pix2StructForConditionalGeneration,
    Blip2ForConditionalGeneration,
]
FairseqModelType = Any
ModelTypeHint = Union[HFModelType, FairseqModelType]

ArgsNamespace = argparse.Namespace
ComponentConfigType = types.SimpleNamespace
WeightDict = dict[str, torch.Tensor]
ConfigParserType = configparser.ConfigParser


# --- Minimal Placeholders for TensorRT-LLM Config Objects ---
class MinimalMapping:
    """Minimal replacement for tensorrt_llm.mapping needed by config."""

    def __init__(self, world_size=1, tp_size=1, pp_size=1):
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.tp_rank = 0  # Default or set via set_rank
        self.pp_rank = 0  # Default or set via set_rank

    def pp_layers(self, num_layers):
        """Mimic pp_layers calculation (basic version)."""
        layers_per_pipeline_stage = num_layers // self.pp_size
        start_layer = self.pp_rank * layers_per_pipeline_stage
        end_layer = (self.pp_rank + 1) * layers_per_pipeline_stage
        return range(start_layer, end_layer)


class MinimalConfig:
    """Minimal replacement for tensorrt_llm.models.PretrainedConfig."""

    def __init__(self, config_dict: dict[str, Any]):
        self._config = config_dict
        # Initialize mapping from the config dict
        mapping_dict = config_dict.get("mapping", {})
        self.mapping = MinimalMapping(
            world_size=mapping_dict.get("world_size", 1),
            tp_size=mapping_dict.get("tp_size", 1),
            pp_size=mapping_dict.get("pp_size", 1),
        )
        # Store dtype separately for convenience, default if not present
        self.dtype = config_dict.get("dtype", "float16")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        """Class method to create an instance from a dictionary."""
        return cls(config_dict)

    def set_rank(self, rank: int):
        """Set the tp_rank and pp_rank based on global rank."""
        self.mapping.tp_rank = rank % self.mapping.tp_size
        self.mapping.pp_rank = rank // self.mapping.tp_size
        # Add rank to the main config dict if needed elsewhere, though unlikely
        self._config["rank"] = rank

    def __getattr__(self, name: str) -> Any:
        """Allow direct attribute access to the underlying config dict."""
        if name in self._config:
            return self._config[name]
        # Add a check for 'mapping' and 'dtype' before raising error
        if name == "mapping":
            return self.mapping
        if name == "dtype":
            return self.dtype
        # Raise AttributeError for attributes not in the config, mapping, or dtype
        # This helps catch issues if the conversion functions expect attributes
        # that aren't present in the original JSON config.
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
            + " (also checked underlying config dict)"
        )


# --- End Placeholders ---


def copy_args_to_component_config(
    component_config: ComponentConfigType, args: ArgsNamespace
) -> ComponentConfigType:
    for arg in vars(args):
        setattr(component_config, arg, getattr(args, arg))
    return component_config


def parse_t5_config(
    args: ArgsNamespace, hf_model: T5ForConditionalGeneration
) -> tuple[ComponentConfigType | None, ComponentConfigType]:
    config = configparser.ConfigParser()

    config["encoder"] = {}
    if hf_model.encoder.config:
        for key, val in hf_model.encoder.config.to_dict().items():
            config["encoder"][key] = f"{val}"

    def get_offset_q_scaling(component_cfg: ComponentConfigType) -> float:
        head_size = getattr(component_cfg, "head_size", 0)
        if not isinstance(head_size, int) or head_size <= 0:
            LOGGER.warning("Invalid head_size in component config, returning default scaling.")
            return 1.0
        return 1.0 / (head_size**0.5)

    config["decoder"] = {}
    if hf_model.decoder.config:
        for key, val in hf_model.decoder.config.to_dict().items():
            config["decoder"][key] = f"{val}"

    config["structure"] = {}
    config["structure"]["t5_with_bias"] = "false"
    config["structure"]["use_gated_activation"] = str(hf_model.config.is_gated_act)
    config["structure"]["position_embedding_type"] = "relative"
    config["structure"]["model_type"] = args.model_type

    def parse_t5_config_by_component(
        cfg_parser: ConfigParserType, component: str, args_ns: ArgsNamespace
    ) -> ComponentConfigType:
        component_config = types.SimpleNamespace()
        component_config = copy_args_to_component_config(component_config, args_ns)

        component_config.n_head = cfg_parser.getint(component, "num_heads", fallback=0)
        component_config.head_size = cfg_parser.getint(component, "d_kv", fallback=0)
        component_config.hidden_size = cfg_parser.getint(component, "d_model", fallback=0)
        component_config.ffn_hidden_size = cfg_parser.getint(component, "d_ff", fallback=0)
        component_config.vocab_size = cfg_parser.getint(component, "vocab_size", fallback=0)
        component_config.n_positions = cfg_parser.getint(component, "n_positions", fallback=512)
        component_config.has_position_embedding = cfg_parser.getboolean(
            component, "has_position_embedding", fallback=False
        )
        component_config.has_token_type_embedding = cfg_parser.getboolean(
            component, "has_token_type_embedding", fallback=False
        )
        component_config.has_embedding_layernorm = cfg_parser.getboolean(
            component, "has_embedding_layernorm", fallback=False
        )
        component_config.has_embedding_scale = cfg_parser.getboolean(
            component, "has_embedding_scale", fallback=False
        )
        component_config.q_scaling = get_offset_q_scaling(component_config)
        component_config.has_attention_qkvo_bias = cfg_parser.getboolean(
            component, "has_attention_qkvo_bias", fallback=False
        )
        component_config.has_mlp_bias = cfg_parser.getboolean(
            component, "has_mlp_bias", fallback=False
        )
        component_config.has_model_final_layernorm = cfg_parser.getboolean(
            component, "has_model_final_layernorm", fallback=True
        )
        component_config.layernorm_eps = cfg_parser.getfloat(
            component, "layer_norm_epsilon", fallback=1e-6
        )
        component_config.layernorm_position = cfg_parser.get(
            component, "layernorm_position", fallback="pre_layernorm"
        )
        component_config.layernorm_type = cfg_parser.get(
            component, "layernorm_type", fallback="RmsNorm"
        )
        component_config.hidden_act = cfg_parser.get(component, "dense_act_fn", fallback="relu")
        component_config.gated_act = cfg_parser.getboolean(
            component, "is_gated_act", fallback=False
        )
        component_config.mlp_type = "GatedMLP" if component_config.gated_act else "MLP"
        component_config.num_buckets = cfg_parser.getint(
            component, "relative_attention_num_buckets", fallback=32
        )
        component_config.max_distance = cfg_parser.getint(
            component, "relative_attention_max_distance", fallback=128
        )
        component_config.position_embedding_type = cfg_parser.get(
            "structure", "position_embedding_type", fallback="relative"
        )
        component_config.logits_dtype = cfg_parser.get(
            component, "logits_dtype", fallback="float32"
        )

        if component == "encoder":
            component_config.n_layer = cfg_parser.getint(component, "num_layers", fallback=0)
            component_config.relative_attention = (
                component_config.position_embedding_type == "relative"
            )
        elif component == "decoder":
            component_config.n_layer = cfg_parser.getint(
                component, "num_decoder_layers", fallback=0
            )
            component_config.has_lm_head_bias = cfg_parser.getboolean(
                component,
                "has_lm_head_bias",
                fallback=False,
            )
            component_config.relative_attention = cfg_parser.getboolean(
                component, "relative_attention", fallback=True
            )
            component_config.rescale_before_lm_head = cfg_parser.getboolean(
                component, "tie_word_embeddings", fallback=True
            )
            component_config.encoder_hidden_size = cfg_parser.getint(
                "encoder", "d_model", fallback=0
            )
            component_config.encoder_num_heads = cfg_parser.getint(
                "encoder", "num_heads", fallback=0
            )
            component_config.encoder_head_size = cfg_parser.getint("encoder", "d_kv", fallback=0)
            component_config.decoder_start_token_id = cfg_parser.getint(
                "decoder", "decoder_start_token_id", fallback=None
            )
            component_config.eos_token_id = cfg_parser.getint(
                "decoder", "eos_token_id", fallback=None
            )
            bos_token_id_str = cfg_parser.get("decoder", "bos_token_id", fallback="None")
            component_config.bos_token_id = (
                int(bos_token_id_str) if bos_token_id_str != "None" else None
            )
            component_config.pad_token_id = cfg_parser.getint(
                "decoder", "pad_token_id", fallback=None
            )
        else:
            raise AssertionError(f"Unsupported component: {component}")

        return component_config

    encoder_config = parse_t5_config_by_component(config, "encoder", args)
    decoder_config = parse_t5_config_by_component(config, "decoder", args)

    return encoder_config, decoder_config


def convert_t5_weights_to_tllm_safetensors(
    config: MinimalConfig, component: str, params: WeightDict
) -> WeightDict:
    weights: WeightDict = {}

    mapping = config.mapping
    if mapping is None:
        LOGGER.error("Mapping config is required for weight conversion but is missing.")
        return weights

    typed_params = cast("dict[str, torch.Tensor]", params)
    convert_weight_to_dtype(typed_params, config.dtype)

    hidden_size = getattr(config, "hidden_size", 0)
    ffn_hidden_size = getattr(config, "intermediate_size", getattr(config, "ffn_hidden_size", 0))
    num_layers = getattr(config, "num_hidden_layers", 0)
    n_head = getattr(config, "num_attention_heads", 0)
    head_size = getattr(config, "head_size", 0)
    attention_hidden_size = n_head * head_size
    num_buckets = getattr(config, "num_buckets", 32)
    vocab_size = getattr(config, "vocab_size", 0)
    model_type = getattr(config, "model_type", "t5")

    if any(
        v == 0 for v in [hidden_size, ffn_hidden_size, num_layers, n_head, head_size, vocab_size]
    ):
        LOGGER.error("Core configuration properties are missing or zero, cannot convert weights.")
        return weights

    hf_param_prefix = f"{component}"
    trtllm_layer_name = f"{component}_layers"
    trtllm_attn_layer_name = "attention" if component == "encoder" else "self_attention"
    trtllm_attn_layernorm_name = (
        "self_attention_layernorm" if component == "decoder" else "attention_layernorm"
    )
    hf_component_idx = 1 if component == "encoder" else 2

    def get_attn_module_name(comp: str, block_idx: int, layer_idx: int, attn_type: str) -> str:
        return f"{comp}.block.{block_idx}.layer.{layer_idx}.{attn_type}"

    if "shared.weight" in typed_params:
        weights["embedding.vocab_embedding.weight"] = reshape(
            typed_params["shared.weight"].clone(), None
        )
    else:
        LOGGER.warning("shared.weight not found in parameters, skipping vocab embedding.")

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        trtllm_layer_name_prefix = f"{trtllm_layer_name}.{local_layer_idx}"
        hf_layer_name_prefix = f"{hf_param_prefix}.block.{layer_idx}"

        HiddenLayerWeightInfo = dict[str, Any]
        hidden_layer_name_split: dict[str, HiddenLayerWeightInfo] = {
            f"{hf_layer_name_prefix}.layer.0.SelfAttention.o.weight": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.dense.weight",
                "shape": (hidden_size, attention_hidden_size // mapping.tp_size),
                "split_dim": -1,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.proj.weight",
                "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
                "split_dim": -1,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.weight",
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_0.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.weight",
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0,
            },
        }

        hidden_layer_name_no_split: dict[str, HiddenLayerWeightInfo] = {
            f"{hf_layer_name_prefix}.layer.0.layer_norm.weight": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layernorm_name}.weight",
                "shape": None,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.layer_norm.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp_layernorm.weight",
                "shape": None,
            },
        }

        gated_act = getattr(config, "gated_act", False)
        if gated_act:
            hidden_layer_name_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi2.weight": {
                        "name": f"{trtllm_layer_name_prefix}.mlp.gate.weight",
                        "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                        "split_dim": 0,
                    },
                    f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_1.weight": {
                        "name": f"{trtllm_layer_name_prefix}.mlp.gate.weight",
                        "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                        "split_dim": 0,
                    },
                }
            )

        if component == "decoder":
            hidden_layer_name_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.1.EncDecAttention.o.weight": {
                        "name": f"{trtllm_layer_name_prefix}.cross_attention.dense.weight",
                        "shape": (hidden_size, attention_hidden_size // mapping.tp_size),
                        "split_dim": -1,
                    },
                }
            )
            hidden_layer_name_no_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.1.layer_norm.weight": {
                        "name": f"{trtllm_layer_name_prefix}.cross_attention_layernorm.weight",
                        "shape": None,
                    },
                }
            )
            cross_attn_module_name = get_attn_module_name(
                component, layer_idx, 1, "EncDecAttention"
            )
            weights.update(
                fuse_qkv_one_layer(
                    typed_params,
                    cross_attn_module_name,
                    f"{trtllm_layer_name_prefix}.cross_attention",
                    mapping.tp_size,
                    mapping.tp_rank,
                    model_type,
                    (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                    None,
                )
            )

        self_attn_module_name = get_attn_module_name(component, layer_idx, 0, "SelfAttention")
        weights.update(
            fuse_qkv_one_layer(
                typed_params,
                self_attn_module_name,
                f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}",
                mapping.tp_size,
                mapping.tp_rank,
                model_type,
                (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                None,
            )
        )

        rel_attn_bias_name = (
            f"{hf_param_prefix}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        )
        if rel_attn_bias_name in typed_params:
            weights[f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.rel_attn_table"] = (
                reshape(
                    split(
                        typed_params[rel_attn_bias_name].T,
                        mapping.tp_size,
                        mapping.tp_rank,
                        0,
                    ),
                    (n_head // mapping.tp_size, num_buckets),
                )
            )
        elif f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.rel_attn_table" not in weights:
            LOGGER.warning(f"Relative attention bias weight '{rel_attn_bias_name}' not found.")

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            if hf_weight_name in typed_params:
                target_name = cast("str", weight_info.get("name"))
                split_dim = cast("int", weight_info.get("split_dim"))
                shape = weight_info.get("shape")
                if target_name and split_dim is not None:
                    weights[target_name] = reshape(
                        split(
                            typed_params[hf_weight_name],
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=split_dim,
                        ),
                        shape,
                    )
                else:
                    LOGGER.warning(
                        f"Skipping weight {hf_weight_name} due to missing info in definition."
                    )

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            if hf_weight_name in typed_params:
                target_name = cast("str", weight_info.get("name"))
                shape = weight_info.get("shape")
                if target_name:
                    weights[target_name] = reshape(
                        typed_params[hf_weight_name].clone(), shape=shape
                    )
                else:
                    LOGGER.warning(
                        f"Skipping weight {hf_weight_name} due to missing name in definition."
                    )

    final_ln_weight_name = f"{hf_param_prefix}.final_layer_norm.weight"
    if final_ln_weight_name in typed_params:
        weights["final_layernorm.weight"] = reshape(
            typed_params[final_ln_weight_name].clone(), None
        )
    else:
        LOGGER.warning(f"Final LayerNorm weight '{final_ln_weight_name}' not found.")

    if component == "decoder":
        lm_head_weight_name = "lm_head.weight"
        if lm_head_weight_name in typed_params:
            weights["lm_head.weight"] = reshape(
                split(typed_params[lm_head_weight_name], mapping.tp_size, mapping.tp_rank, dim=0),
                (vocab_size // mapping.tp_size, hidden_size),
            )
        else:
            LOGGER.warning(f"LM head weight '{lm_head_weight_name}' not found.")

    return weights


convert_blip2_weights_to_tllm_safetensors = convert_t5_weights_to_tllm_safetensors


def parse_nmt_config(args, model):
    config = configparser.ConfigParser()
    fairseq_config = vars(model.cfg.model)

    config["encoder"] = {}
    for key, val in fairseq_config.items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["q_scaling"] = "1"
    config["encoder"]["has_model_final_layernorm"] = config["encoder"]["encoder_normalize_before"]
    config["encoder"]["vocab_size"] = str(len(model.src_dict))

    config["decoder"] = {}
    for key, val in fairseq_config.items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["q_scaling"] = "1"
    config["decoder"]["rescale_before_lm_head"] = "false"
    config["decoder"]["has_model_final_layernorm"] = str(
        config["decoder"].getboolean("decoder_normalize_before", False)
        and not config["decoder"].getboolean("no_decoder_final_norm", False)
    )
    config["decoder"]["vocab_size"] = str(len(model.tgt_dict))

    config["structure"] = {}
    config["structure"]["t5_with_bias"] = "true"
    config["structure"]["use_gated_activation"] = "false"
    config["structure"]["position_embedding_type"] = "learned_absolute"
    config["structure"]["model_type"] = args.model_type

    def parse_nmt_config_by_component(config, component, args):
        assert component in ("encoder", "decoder"), "Unsupported component!"
        component_config = types.SimpleNamespace()
        component_config = copy_args_to_component_config(component_config, args)
        component_config.n_layer = config.getint(component, f"{component}_layers")
        component_config.n_head = config.getint(component, f"{component}_attention_heads")
        component_config.hidden_size = config.getint(component, f"{component}_embed_dim")
        component_config.head_size = config.getint(
            component, "d_kv", fallback=component_config.hidden_size // component_config.n_head
        )
        component_config.ffn_hidden_size = config.getint(component, f"{component}_ffn_embed_dim")
        component_config.vocab_size = config.getint(component, "vocab_size")
        component_config.n_positions = config.getint(component, "max_source_positions")
        component_config.has_position_embedding = not config.getboolean(
            component, "no_token_positional_embeddings", fallback=False
        )
        component_config.has_token_type_embedding = config.getboolean(
            component, "has_token_type_embedding", fallback=False
        )
        component_config.has_embedding_layernorm = config.getboolean(
            component, "layernorm_embedding", fallback=True
        )
        component_config.has_embedding_scale = not config.getboolean(
            component, "no_scale_embedding"
        )
        component_config.q_scaling = config.getfloat(component, "q_scaling", fallback=1.0)
        component_config.has_attention_qkvo_bias = config.getboolean(
            "structure", "t5_with_bias", fallback=True
        )
        component_config.has_mlp_bias = config.getboolean(
            "structure", "t5_with_bias", fallback=True
        )
        component_config.has_model_final_layernorm = config.getboolean(
            component, "has_model_final_layernorm"
        )
        component_config.layernorm_eps = config.getfloat(
            component, "layer_norm_epsilon", fallback=1e-5
        )

        normalize_before = config.getboolean(component, f"{component}_normalize_before")
        component_config.layernorm_position = (
            "pre_layernorm" if normalize_before else "post_layernorm"
        )

        component_config.layernorm_type = config.get(
            component, "layernorm_type", fallback="LayerNorm"
        )
        component_config.hidden_act = config.get(component, "activation_fn")
        component_config.gated_act = config.getboolean(component, "is_gated_act")
        component_config.mlp_type = "GatedMLP" if component_config.gated_act else "MLP"
        component_config.num_buckets = config.getint(component, "relative_attention_num_buckets")
        component_config.max_distance = config.getint(component, "relative_attention_max_distance")
        component_config.position_embedding_type = config.get(
            "structure", "position_embedding_type"
        )
        component_config.logits_dtype = config.get(component, "logits_dtype", fallback="float32")
        component_config.rescale_before_lm_head = config.getboolean(
            component, "tie_word_embeddings"
        )
        component_config.encoder_hidden_size = config.getint("encoder", "encoder_embed_dim")
        component_config.encoder_num_heads = config.getint("encoder", "encoder_attention_heads")
        component_config.encoder_head_size = config.getint(
            "encoder",
            "d_kv",
            fallback=component_config.encoder_hidden_size // component_config.encoder_num_heads,
        )
        component_config.decoder_start_token_id = None
        component_config.eos_token_id = None
        component_config.bos_token_id = None
        component_config.pad_token_id = None

        return component_config

    encoder_config = parse_nmt_config_by_component(config, "encoder", args)
    decoder_config = parse_nmt_config_by_component(config, "decoder", args)

    return encoder_config, decoder_config


def convert_nmt_weights_to_tllm_safetensors(config, component, params, sin_pos_embedding):
    weights = {}

    mapping = config.mapping

    hidden_size = config.hidden_size

    convert_weight_to_dtype(params, config.dtype)
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    hf_param_prefix = f"models.0.{component}"
    trtllm_layer_name = f"{component}_layers"
    trtllm_attn_layer_name = "attention" if component == "encoder" else "self_attention"
    trtllm_attn_layernorm_name = (
        "self_attention_layernorm" if component == "decoder" else "attention_layernorm"
    )

    hidden_layer_name_split = {
        "self_attn.out_proj.weight": {
            "name": f"{trtllm_attn_layer_name}.dense.weight",
            "shape": (hidden_size, hidden_size // mapping.tp_size),
            "split_dim": -1,
        },
        "fc1.weight": {
            "name": "mlp.fc.weight",
            "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
            "split_dim": 0,
        },
        "fc1.bias": {
            "name": "mlp.fc.bias",
            "shape": (ffn_hidden_size // mapping.tp_size),
            "split_dim": 0,
        },
        "fc2.weight": {
            "name": "mlp.proj.weight",
            "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
            "split_dim": -1,
        },
    }

    hidden_layer_name_no_split = {
        "self_attn.out_proj.bias": {
            "name": f"{trtllm_attn_layer_name}.dense.bias",
            "shape": (hidden_size),
        },
        "self_attn_layer_norm.weight": {
            "name": f"{trtllm_attn_layernorm_name}.weight",
            "shape": None,
        },
        "self_attn_layer_norm.bias": {"name": f"{trtllm_attn_layernorm_name}.bias", "shape": None},
        "fc2.bias": {"name": "mlp.proj.bias", "shape": (hidden_size)},
        "final_layer_norm.weight": {"name": "mlp_layernorm.weight", "shape": None},
        "final_layer_norm.bias": {"name": "mlp_layernorm.bias", "shape": None},
    }

    if component == "decoder":
        hidden_layer_name_split.update(
            {
                "encoder_attn.out_proj.weight": {
                    "name": "cross_attention.dense.weight",
                    "shape": (hidden_size, hidden_size // mapping.tp_size),
                    "split_dim": -1,
                },
            }
        )
        hidden_layer_name_no_split.update(
            {
                "encoder_attn.out_proj.bias": {
                    "name": "cross_attention.dense.bias",
                    "shape": (hidden_size),
                },
                "encoder_attn_layer_norm.weight": {
                    "name": "cross_attention_layernorm.weight",
                    "shape": None,
                },
                "encoder_attn_layer_norm.bias": {
                    "name": "cross_attention_layernorm.bias",
                    "shape": None,
                },
            }
        )

    def get_attn_module_name(component, layer, attn_type) -> str:
        return f"models.0.{component}.layers.{int(layer)}.{attn_type}"

    weights["embedding.vocab_embedding.weight"] = reshape(
        params[f"{hf_param_prefix}.embed_tokens.weight"].clone(), (vocab_size, -1)
    )
    weights["embedding.position_embedding.weight"] = reshape(
        sin_pos_embedding, (config.max_position_embeddings, hidden_size)
    )

    num_layers = config.num_hidden_layers

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        hf_layer_name_prefix = f"{hf_param_prefix}.layers.{layer_idx}"
        trtllm_layer_name_prefix = f"{trtllm_layer_name}.{local_layer_idx}"

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            weights[f"{trtllm_layer_name_prefix}.{weight_info['name']}"] = reshape(
                split(
                    params[hf_weight_name],
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=weight_info["split_dim"],
                ),
                weight_info["shape"],
            )

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            trtllm_layer_fullname = f"{trtllm_layer_name_prefix}.{weight_info['name']}"
            hf_layer_fullname = f"{hf_layer_name_prefix}.{hf_weight_name}"
            weights[trtllm_layer_fullname] = reshape(
                params[hf_layer_fullname].clone(), shape=weight_info["shape"]
            )

        self_attn_module_name = get_attn_module_name(component, layer_idx, "self_attn")
        weights.update(
            fuse_qkv_one_layer(
                params,
                self_attn_module_name,
                f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}",
                mapping.tp_size,
                mapping.tp_rank,
                config.model_type,
                (hidden_size * 3 // mapping.tp_size, hidden_size),
                None,
            )
        )
        if component == "decoder":
            cross_attn_module_name = get_attn_module_name(component, layer_idx, "encoder_attn")
            weights.update(
                fuse_qkv_one_layer(
                    params,
                    cross_attn_module_name,
                    f"{trtllm_layer_name_prefix}.cross_attention",
                    mapping.tp_size,
                    mapping.tp_rank,
                    config.model_type,
                    (hidden_size * 3 // mapping.tp_size, hidden_size),
                    None,
                )
            )

    if component == "decoder":
        weights["lm_head.weight"] = reshape(
            split(
                params[f"{hf_param_prefix}.output_projection.weight"],
                mapping.tp_size,
                mapping.tp_rank,
                dim=0,
            ),
            (config.vocab_size // mapping.tp_size, hidden_size),
        )

    if config.has_model_final_layernorm:
        weights["final_layernorm.weight"] = params[f"{hf_param_prefix}.layer_norm.weight"].clone()
        weights["final_layernorm.bias"] = params[f"{hf_param_prefix}.layer_norm.bias"].clone()

    return weights


def parse_bart_config(args, hf_model):
    config = configparser.ConfigParser()

    config["decoder"] = {}
    for key, val in hf_model.model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["q_scaling"] = "1"
    config["decoder"]["rescale_before_lm_head"] = str(False)
    config["decoder"]["has_model_final_layernorm"] = str(
        args.nougat or isinstance(hf_model, MBartForConditionalGeneration)
    )

    if args.nougat:
        config["decoder"]["normalize_before"] = str(True)
        config["decoder"]["normalize_embeddings"] = str(True)

        config["encoder"] = {}
        encoder_config_keys = [
            "encoder_ffn_dim",
            "encoder_layers",
            "encoder_attention_heads",
            "encoder_layerdrop",
            "d_model",
        ]
        for key in encoder_config_keys:
            config["encoder"][key] = config["decoder"][key]
    else:
        config["encoder"] = {}
        for key, val in hf_model.model.encoder.config.to_dict().items():
            config["encoder"][key] = f"{val}"
        config["encoder"]["q_scaling"] = "1"

        config["encoder"]["has_model_final_layernorm"] = str(
            isinstance(hf_model, MBartForConditionalGeneration)
        )

    config["structure"] = {}
    config["structure"]["t5_with_bias"] = "true"
    config["structure"]["use_gated_activation"] = "false"
    config["structure"]["position_embedding_type"] = "learned_absolute"
    config["structure"]["model_type"] = args.model_type

    def parse_bart_config_by_component(config, component, args):
        component_config = types.SimpleNamespace()
        component_config = copy_args_to_component_config(component_config, args)
        component_config.n_layer = config.getint(component, f"{component}_layers")
        component_config.n_head = config.getint(component, f"{component}_attention_heads")
        component_config.hidden_size = config.getint(component, "d_model")
        component_config.head_size = config.getint(
            component, "d_kv", fallback=component_config.hidden_size // component_config.n_head
        )
        component_config.ffn_hidden_size = config.getint(component, f"{component}_ffn_dim")
        component_config.vocab_size = config.getint(component, "vocab_size")
        component_config.n_positions = config.getint(component, "max_position_embeddings")
        component_config.has_position_embedding = config.getboolean(
            component, "has_position_embedding", fallback=True
        )
        component_config.has_token_type_embedding = config.getboolean(
            component, "has_token_type_embedding", fallback=False
        )
        component_config.has_embedding_layernorm = config.getboolean(
            component, "has_embedding_layernorm", fallback=True
        )
        component_config.has_embedding_scale = config.getboolean(component, "scale_embedding")
        component_config.q_scaling = config.getfloat(component, "q_scaling", fallback=1.0)
        component_config.has_attention_qkvo_bias = config.getboolean(
            "structure", "t5_with_bias", fallback=True
        )
        component_config.has_mlp_bias = config.getboolean(
            "structure", "t5_with_bias", fallback=True
        )
        component_config.has_model_final_layernorm = config.getboolean(
            component, "has_model_final_layernorm"
        )
        component_config.layernorm_eps = config.getfloat(
            component, "layer_norm_epsilon", fallback=1e-6
        )

        normalize_before = config.getboolean(component, "normalize_before")
        component_config.layernorm_position = (
            "pre_layernorm" if normalize_before else "post_layernorm"
        )

        component_config.layernorm_type = config.get(
            component, "layernorm_type", fallback="LayerNorm"
        )
        component_config.hidden_act = config.get(component, "activation_function")
        component_config.gated_act = config.getboolean(component, "is_gated_act")
        component_config.mlp_type = "GatedMLP" if component_config.gated_act else "MLP"
        component_config.relative_attention = (
            config.get("structure", "position_embedding_type") == "relative"
        )

        component_config.num_buckets = config.getint(
            component, "relative_attention_num_buckets", fallback=0
        )
        component_config.max_distance = config.getint(
            component, "relative_attention_max_distance", fallback=0
        )
        component_config.position_embedding_type = config.get(
            "structure", "position_embedding_type"
        )
        component_config.logits_dtype = config.get(component, "logits_dtype", fallback="float32")
        component_config.rescale_before_lm_head = config.getboolean(
            component, "rescale_before_lm_head"
        )
        component_config.encoder_hidden_size = config.getint("encoder", "d_model")
        component_config.encoder_num_heads = config.getint("encoder", "encoder_attention_heads")
        component_config.encoder_head_size = config.getint(
            "encoder",
            "d_kv",
            fallback=component_config.encoder_hidden_size // component_config.encoder_num_heads,
        )
        component_config.decoder_start_token_id = config.getint("decoder", "decoder_start_token_id")
        component_config.eos_token_id = config.getint("decoder", "eos_token_id")
        bos_token_id = config.get("decoder", "bos_token_id")
        component_config.bos_token_id = int(bos_token_id) if bos_token_id != "None" else None
        component_config.pad_token_id = config.getint("decoder", "pad_token_id")

        return component_config

    encoder_config = None
    if not args.nougat:
        encoder_config = parse_bart_config_by_component(config, "encoder", args)
    decoder_config = parse_bart_config_by_component(config, "decoder", args)

    return encoder_config, decoder_config


def convert_bart_weights_to_tllm_safetensors(config, component, params):
    weights = {}

    mapping = config.mapping

    hidden_size = config.hidden_size

    convert_weight_to_dtype(params, config.dtype)
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    hf_param_prefix = f"model.{component}"
    trtllm_layer_name = f"{component}_layers"
    trtllm_attn_layer_name = "attention" if component == "encoder" else "self_attention"
    trtllm_attn_layernorm_name = (
        "self_attention_layernorm" if component == "decoder" else "attention_layernorm"
    )
    embedding_layer_names = {
        "embed_tokens.weight": {
            "name": "embedding.vocab_embedding.weight",
            "shape": (vocab_size, -1),
        },
        "embed_positions.weight": {
            "name": "embedding.position_embedding.weight",
            "shape": (config.max_position_embeddings, hidden_size),
        },
        "layernorm_embedding.weight": {
            "name": "embedding.embedding_layernorm.weight",
            "shape": None,
        },
        "layernorm_embedding.bias": {"name": "embedding.embedding_layernorm.bias", "shape": None},
    }

    hidden_layer_name_split = {
        "self_attn.out_proj.weight": {
            "name": f"{trtllm_attn_layer_name}.dense.weight",
            "shape": (hidden_size, hidden_size // mapping.tp_size),
            "split_dim": -1,
        },
        "fc1.weight": {
            "name": "mlp.fc.weight",
            "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
            "split_dim": 0,
        },
        "fc1.bias": {
            "name": "mlp.fc.bias",
            "shape": (ffn_hidden_size // mapping.tp_size),
            "split_dim": 0,
        },
        "fc2.weight": {
            "name": "mlp.proj.weight",
            "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
            "split_dim": -1,
        },
    }

    hidden_layer_name_no_split = {
        "self_attn.out_proj.bias": {
            "name": f"{trtllm_attn_layer_name}.dense.bias",
            "shape": (hidden_size),
        },
        "self_attn_layer_norm.weight": {
            "name": f"{trtllm_attn_layernorm_name}.weight",
            "shape": None,
        },
        "self_attn_layer_norm.bias": {"name": f"{trtllm_attn_layernorm_name}.bias", "shape": None},
        "fc2.bias": {"name": "mlp.proj.bias", "shape": (hidden_size)},
        "final_layer_norm.weight": {"name": "mlp_layernorm.weight", "shape": None},
        "final_layer_norm.bias": {"name": "mlp_layernorm.bias", "shape": None},
    }

    if config.model_type == "mbart":
        hidden_layer_name_split["layer_norm.weight"] = {
            "name": "final_layernorm.weight",
            "shape": None,
            "split_dim": 0,
        }
        hidden_layer_name_no_split["layer_norm.bias"] = {
            "name": "final_layernorm.bias",
            "shape": None,
            "split_dim": 0,
        }

    if component == "decoder":
        hidden_layer_name_split.update(
            {
                "encoder_attn.out_proj.weight": {
                    "name": "cross_attention.dense.weight",
                    "shape": (hidden_size, hidden_size // mapping.tp_size),
                    "split_dim": -1,
                }
            }
        )
        hidden_layer_name_no_split.update(
            {
                "encoder_attn.out_proj.bias": {
                    "name": "cross_attention.dense.bias",
                    "shape": (hidden_size),
                },
                "encoder_attn_layer_norm.weight": {
                    "name": "cross_attention_layernorm.weight",
                    "shape": None,
                },
                "encoder_attn_layer_norm.bias": {
                    "name": "cross_attention_layernorm.bias",
                    "shape": None,
                },
            }
        )

    def get_attn_module_name(component, layer, attn_type) -> str:
        return f"model.{component}.layers.{int(layer)}.{attn_type}"

    for hf_weight_name, weight_info in embedding_layer_names.items():
        if "position" in hf_weight_name:
            weights[weight_info["name"]] = params[f"{hf_param_prefix}.{hf_weight_name}"][2:].clone()
        else:
            weights[weight_info["name"]] = params[f"{hf_param_prefix}.{hf_weight_name}"].clone()
        weights[weight_info["name"]] = reshape(weights[weight_info["name"]], weight_info["shape"])

    num_layers = config.num_hidden_layers

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        hf_layer_name_prefix = f"{hf_param_prefix}.layers.{layer_idx}"
        trtllm_layer_name_prefix = f"{trtllm_layer_name}.{local_layer_idx}"

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            weights[f"{trtllm_layer_name_prefix}.{weight_info['name']}"] = reshape(
                split(
                    params[f"{hf_layer_name_prefix}.{hf_weight_name}"],
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=weight_info["split_dim"],
                ),
                weight_info["shape"],
            )

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            trtllm_layer_fullname = f"{trtllm_layer_name_prefix}.{weight_info['name']}"
            hf_layer_fullname = f"{hf_layer_name_prefix}.{hf_weight_name}"
            weights[trtllm_layer_fullname] = reshape(
                params[hf_layer_fullname].clone(), shape=weight_info["shape"]
            )

        self_attn_module_name = get_attn_module_name(component, layer_idx, "self_attn")
        weights.update(
            fuse_qkv_one_layer(
                params,
                self_attn_module_name,
                f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}",
                mapping.tp_size,
                mapping.tp_rank,
                config.model_type,
                (hidden_size * 3 // mapping.tp_size, hidden_size),
                None,
            )
        )
        if component == "decoder":
            cross_attn_module_name = get_attn_module_name(component, layer_idx, "encoder_attn")
            weights.update(
                fuse_qkv_one_layer(
                    params,
                    cross_attn_module_name,
                    f"{trtllm_layer_name_prefix}.cross_attention",
                    mapping.tp_size,
                    mapping.tp_rank,
                    config.model_type,
                    (hidden_size * 3 // mapping.tp_size, hidden_size),
                    None,
                )
            )

    if component == "decoder":
        weights["lm_head.weight"] = reshape(
            split(
                params[f"{hf_param_prefix}.output_projection.weight"],
                mapping.tp_size,
                mapping.tp_rank,
                dim=0,
            ),
            (config.vocab_size // mapping.tp_size, hidden_size),
        )

    if config.has_model_final_layernorm:
        weights["final_layernorm.weight"] = params[f"{hf_param_prefix}.layer_norm.weight"].clone()
        weights["final_layernorm.bias"] = params[f"{hf_param_prefix}.layer_norm.bias"].clone()

    return weights


def parse_pix2struct_config(args, hf_model):
    def get_offset_q_scaling(pix2struct_cfg: Any) -> float:
        head_size = getattr(pix2struct_cfg, "d_kv", 0)
        num_heads = getattr(pix2struct_cfg, "num_heads", 1)
        hidden_size = getattr(pix2struct_cfg, "hidden_size", 0)
        if head_size <= 0 and num_heads > 0 and hidden_size > 0:
            head_size = hidden_size / num_heads
        if head_size <= 0:
            return 1.0
        return 1.0 / (head_size**0.5)

    config = configparser.ConfigParser()

    config["decoder"] = {}
    for key, val in hf_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"

    config["decoder"]["q_scaling"] = get_offset_q_scaling(hf_model.decoder.config)

    config["structure"] = {}
    config["structure"]["pix2struct_with_bias"] = "false"
    config["structure"]["use_gated_activation"] = "false"
    config["structure"]["position_embedding_type"] = "relative"
    config["structure"]["model_type"] = args.model_type

    def parse_pix2struct_config_by_component(config, component, args):
        if component == "decoder":
            args.n_layer = config.getint(component, "num_layers")
            args.n_head = config.getint(component, "num_heads")
            args.head_size = config.getint(component, "d_kv")
            args.hidden_size = config.getint(component, "hidden_size")
            args.ffn_hidden_size = config.getint(component, "d_ff")
            args.vocab_size = config.getint(component, "vocab_size")
            args.n_positions = config.getint(component, "n_positions", fallback=512)
            args.has_position_embedding = not config.getboolean(
                component, "no_token_positional_embeddings", fallback=False
            )
            args.has_token_type_embedding = config.getboolean(
                component, "has_token_type_embedding", fallback=False
            )
            args.has_embedding_layernorm = config.getboolean(
                component, "has_embedding_layernorm", fallback=False
            )
            args.has_embedding_scale = config.getboolean(
                component, "has_embedding_scale", fallback=False
            )
            args.q_scaling = config.getfloat(component, "q_scaling", fallback=1.0)
            args.has_attention_qkvo_bias = config.getboolean(
                "structure", "t5_with_bias", fallback=True
            )
            args.has_mlp_bias = config.getboolean("structure", "t5_with_bias", fallback=True)
            args.has_model_final_layernorm = config.getboolean(
                component, "has_model_final_layernorm"
            )
            args.layernorm_eps = config.getfloat(component, "layer_norm_epsilon", fallback=1e-5)

            normalize_before = config.getboolean(component, f"{component}_normalize_before")
            args.layernorm_position = "pre_layernorm" if normalize_before else "post_layernorm"

            args.layernorm_type = config.get(component, "layernorm_type", fallback="LayerNorm")
            args.hidden_act = config.get(component, "activation_fn")
            args.gated_act = config.getboolean(component, "is_gated_act")
            args.mlp_type = "GatedMLP" if args.gated_act else "MLP"
            args.relative_attention = (
                config.get("structure", "position_embedding_type") == "relative"
            )

            args.num_buckets = config.getint(
                component, "relative_attention_num_buckets", fallback=0
            )
            args.max_distance = config.getint(
                component, "relative_attention_max_distance", fallback=0
            )
            args.position_embedding_type = config.get("structure", "position_embedding_type")
            args.logits_dtype = config.get(component, "logits_dtype", fallback="float32")
            args.rescale_before_lm_head = config.getboolean(component, "tie_word_embeddings")
            args.encoder_hidden_size = config.getint("encoder", "d_model")
            args.encoder_num_heads = config.getint("encoder", "encoder_attention_heads")
            args.encoder_head_size = config.getint(
                "encoder",
                "d_kv",
                fallback=args.encoder_hidden_size // args.encoder_num_heads,
            )
            args.decoder_start_token_id = None
            args.eos_token_id = None
            args.bos_token_id = None
            args.pad_token_id = None

        else:
            raise AssertionError("Unsupported component!")
        return args

    decoder_args = parse_pix2struct_config_by_component(config, "decoder", args)
    return None, decoder_args


def convert_pix2struct_weights_to_tllm_safetensors(config, component, params):
    weights = {}

    mapping = config.mapping

    convert_weight_to_dtype(params, config.dtype)
    hidden_size = config.hidden_size
    ffn_hidden_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    n_head = config.num_attention_heads
    head_size = config.head_size
    attention_hidden_size = n_head * head_size

    hf_param_prefix = f"{component}"
    trtllm_layer_name = f"{component}_layers"
    trtllm_attn_layer_name = "attention" if component == "encoder" else "self_attention"
    trtllm_attn_layernorm_name = (
        "self_attention_layernorm" if component == "decoder" else "attention_layernorm"
    )

    def get_attn_module_name(comp: str, block_idx: int, layer_idx: int, attn_type: str) -> str:
        return f"{comp}.layer.{block_idx}.{attn_type}.attention"

    weights["embedding.vocab_embedding.weight"] = reshape(
        params[f"{hf_param_prefix}.embed_tokens.weight"].clone(), (vocab_size, -1)
    )
    weights["embedding.position_embedding.weight"] = reshape(
        sin_pos_embedding, (config.max_position_embeddings, hidden_size)
    )

    num_layers = config.num_hidden_layers

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        hf_layer_name_prefix = f"{hf_param_prefix}.layers.{layer_idx}"
        trtllm_layer_name_prefix = f"{trtllm_layer_name}.{local_layer_idx}"

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            weights[f"{trtllm_layer_name_prefix}.{weight_info['name']}"] = reshape(
                split(
                    params[f"{hf_layer_name_prefix}.{hf_weight_name}"],
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=weight_info["split_dim"],
                ),
                weight_info["shape"],
            )

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            trtllm_layer_fullname = f"{trtllm_layer_name_prefix}.{weight_info['name']}"
            hf_layer_fullname = f"{hf_layer_name_prefix}.{hf_weight_name}"
            weights[trtllm_layer_fullname] = reshape(
                params[hf_layer_fullname].clone(), shape=weight_info["shape"]
            )

        self_attn_module_name = get_attn_module_name(component, layer_idx, "self_attn")
        weights.update(
            fuse_qkv_one_layer(
                params,
                self_attn_module_name,
                f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}",
                mapping.tp_size,
                mapping.tp_rank,
                config.model_type,
                (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                None,
            )
        )

    weights["final_layernorm.weight"] = reshape(
        params[f"{hf_param_prefix}.final_layer_norm.weight"].clone(), None
    )

    weights["lm_head.weight"] = reshape(
        split(params[f"{hf_param_prefix}.lm_head.weight"], mapping.tp_size, mapping.tp_rank, dim=0),
        (config.vocab_size // mapping.tp_size, hidden_size),
    )
    if not config.use_implicit_relative_attention:
        weights["rel_attn_table"] = reshape(
            split(
                params[
                    f"{component}.layer.0.self_attention.attention.relative_attention_bias.weight"
                ].T,
                mapping.tp_size,
                mapping.tp_rank,
                0,
            ),
            (n_head // mapping.tp_size, config.num_buckets),
        )

    return weights


def get_model(args: ArgsNamespace) -> ModelTypeHint:
    model_dir_str = str(args.model_dir)
    if args.model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_dir_str)
    elif args.model_type == "nmt":
        if TransformerModel:
            model = TransformerModel.from_pretrained(model_dir_str)
        else:
            raise ImportError("fairseq not installed, NMT support disabled.")
    elif args.model_type == "bart":
        if args.nougat:
            model = VisionEncoderDecoderModel.from_pretrained(model_dir_str)
            model = model.get_decoder()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir_str)
    elif args.model_type == "pix2struct":
        model = Pix2StructForConditionalGeneration.from_pretrained(model_dir_str)
    elif args.model_type == "blip2":
        blip_model = Blip2ForConditionalGeneration.from_pretrained(model_dir_str)
        model = blip_model.language_model
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    return cast("ModelTypeHint", model)


def convert_checkpoint(args: ArgsNamespace) -> None:
    model: ModelTypeHint = get_model(args)

    saved_dir = Path(args.output_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    encoder_saved_dir = saved_dir / "encoder"
    encoder_saved_dir.mkdir(parents=True, exist_ok=True)
    decoder_saved_dir = saved_dir / "decoder"
    decoder_saved_dir.mkdir(parents=True, exist_ok=True)

    world_size = args.tp_size * args.pp_size

    kv_cache_quant_algo = None
    quant_algo = None

    model_type = args.model_type if args.model_type != "blip2" else "t5"
    encoder_config, decoder_config = globals()[f"parse_{model_type}_config"](args, model)

    additional_settings = ["gated_act"]

    encoder_tllm_config: dict[str, Any] | None = None
    encoder_convert_args: dict[str, Any] | None = None

    if encoder_config and not args.nougat and args.model_type != "pix2struct":
        encoder_tllm_config = {
            "architecture": "EncoderModel",
            "dtype": args.dtype,
            "logits_dtype": encoder_config.logits_dtype,
            "num_hidden_layers": encoder_config.n_layer,
            "num_attention_heads": encoder_config.n_head,
            "hidden_size": encoder_config.hidden_size,
            "norm_epsilon": encoder_config.layernorm_eps,
            "vocab_size": encoder_config.vocab_size,
            "position_embedding_type": encoder_config.position_embedding_type,
            "hidden_act": encoder_config.hidden_act,
            "quantization": {
                "quant_algo": quant_algo,
                "kv_cache_quant_algo": kv_cache_quant_algo,
            },
            "mapping": {
                "world_size": world_size,
                "tp_size": args.tp_size,
                "pp_size": args.pp_size,
            },
            "use_parallel_embedding": args.use_parallel_embedding,
            "embedding_sharding_dim": args.embedding_sharding_dim,
            "max_position_embeddings": encoder_config.n_positions,
            "num_key_value_heads": encoder_config.n_head,
            "head_size": encoder_config.head_size,
            "has_position_embedding": encoder_config.has_position_embedding,
            "layernorm_type": encoder_config.layernorm_type,
            "has_attention_qkvo_bias": encoder_config.has_attention_qkvo_bias,
            "has_mlp_bias": encoder_config.has_mlp_bias,
            "has_model_final_layernorm": encoder_config.has_model_final_layernorm,
            "has_embedding_layernorm": encoder_config.has_embedding_layernorm,
            "has_embedding_scale": encoder_config.has_embedding_scale,
            "intermediate_size": encoder_config.ffn_hidden_size,
            "q_scaling": encoder_config.q_scaling,
            "layernorm_position": encoder_config.layernorm_position,
            "mlp_type": encoder_config.mlp_type,
            "relative_attention": encoder_config.relative_attention,
            "max_distance": encoder_config.max_distance,
            "num_buckets": encoder_config.num_buckets,
            "model_type": encoder_config.model_type,
        }
        for additional_setting in additional_settings:
            if hasattr(encoder_config, additional_setting):
                encoder_tllm_config.update(
                    {additional_setting: getattr(encoder_config, additional_setting)}
                )

        with (encoder_saved_dir / "config.json").open("w") as f:
            json.dump(encoder_tllm_config, f, indent=4, default=str)

        encoder_convert_args = {"params": model.state_dict(), "component": "encoder"}
    tllm_decoder_config: dict[str, Any] = {
        "architecture": "DecoderModel",
        "dtype": args.dtype,
        "logits_dtype": decoder_config.logits_dtype,
        "num_hidden_layers": decoder_config.n_layer,
        "num_attention_heads": decoder_config.n_head,
        "hidden_size": decoder_config.hidden_size,
        "norm_epsilon": decoder_config.layernorm_eps,
        "vocab_size": decoder_config.vocab_size,
        "position_embedding_type": decoder_config.position_embedding_type,
        "hidden_act": decoder_config.hidden_act,
        "quantization": {
            "quant_algo": quant_algo,
            "kv_cache_quant_algo": kv_cache_quant_algo,
        },
        "mapping": {
            "world_size": world_size,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
        },
        "use_parallel_embedding": args.use_parallel_embedding,
        "embedding_sharding_dim": args.embedding_sharding_dim,
        "max_position_embeddings": decoder_config.n_positions,
        "head_size": decoder_config.head_size,
        "has_position_embedding": decoder_config.has_position_embedding,
        "layernorm_type": decoder_config.layernorm_type,
        "has_attention_qkvo_bias": decoder_config.has_attention_qkvo_bias,
        "has_mlp_bias": decoder_config.has_mlp_bias,
        "has_model_final_layernorm": decoder_config.has_model_final_layernorm,
        "has_embedding_layernorm": decoder_config.has_embedding_layernorm,
        "has_embedding_scale": decoder_config.has_embedding_scale,
        "intermediate_size": decoder_config.ffn_hidden_size,
        "q_scaling": decoder_config.q_scaling,
        "layernorm_position": decoder_config.layernorm_position,
        "mlp_type": decoder_config.mlp_type,
        "relative_attention": decoder_config.relative_attention,
        "max_distance": decoder_config.max_distance,
        "num_buckets": decoder_config.num_buckets,
        "model_type": decoder_config.model_type,
        "rescale_before_lm_head": decoder_config.rescale_before_lm_head,
        "encoder_hidden_size": decoder_config.encoder_hidden_size,
        "encoder_num_heads": decoder_config.encoder_num_heads,
        "encoder_head_size": decoder_config.encoder_head_size,
        "skip_cross_kv": args.skip_cross_kv,
        "use_implicit_relative_attention": args.use_implicit_relative_attention,
        "decoder_start_token_id": decoder_config.decoder_start_token_id,
        "eos_token_id": decoder_config.eos_token_id,
        "bos_token_id": decoder_config.bos_token_id,
        "pad_token_id": decoder_config.pad_token_id,
    }
    for additional_setting in additional_settings:
        if hasattr(decoder_config, additional_setting):
            tllm_decoder_config.update(
                {additional_setting: getattr(decoder_config, additional_setting)}
            )

    with (decoder_saved_dir / "config.json").open("w") as f:
        json.dump(tllm_decoder_config, f, indent=4, default=str)

    decoder_convert_args = {"params": model.state_dict(), "component": "decoder"}

    if args.model_type == "nmt" and isinstance(model, TransformerModel):
        if encoder_convert_args and "sin_pos_embedding" in encoder_convert_args:
            decoder_convert_args["sin_pos_embedding"] = encoder_convert_args["sin_pos_embedding"]
        else:
            LOGGER.warning("Recalculating sin_pos_embedding for NMT decoder.")
            fairseq_config = vars(model.cfg.model)
            num_embeddings = fairseq_config["max_source_positions"]
            embedding_dim = fairseq_config["encoder_embed_dim"]
            padding_idx = model.models[0].encoder.embed_tokens.padding_idx
            sin_pos_embedding = model.models[0].encoder.embed_positions.get_embedding(
                padding_idx + 1 + num_embeddings, embedding_dim, padding_idx=padding_idx
            )
            sin_pos_embedding = sin_pos_embedding[2:, :]
            decoder_convert_args["sin_pos_embedding"] = sin_pos_embedding
    elif args.model_type == "nmt":
        LOGGER.error("NMT model type specified but Fairseq model not loaded correctly.")
        return

    workers = args.workers
    if not isinstance(workers, int) or workers <= 0:
        workers = 1

    if workers > 1:
        workers = min(workers, world_size)
        LOGGER.info(f"Convert checkpoint using {workers} workers.")
        import torch.multiprocessing as mp

        spawn_context = mp.get_context("spawn")

        if encoder_tllm_config is not None and encoder_convert_args is not None:
            spawn_context.spawn(
                convert,
                nprocs=workers,
                args=(
                    world_size,
                    args,
                    encoder_tllm_config,
                    encoder_convert_args,
                    encoder_saved_dir,
                    workers,
                ),
                join=True,
            )
        spawn_context.spawn(
            convert,
            nprocs=workers,
            args=(
                world_size,
                args,
                tllm_decoder_config,
                decoder_convert_args,
                decoder_saved_dir,
                workers,
            ),
            join=True,
        )
    else:
        LOGGER.info("Convert checkpoint using 1 worker.")
        if encoder_tllm_config is not None and encoder_convert_args is not None:
            convert(
                0, world_size, args, encoder_tllm_config, encoder_convert_args, encoder_saved_dir, 1
            )
        convert(
            0, world_size, args, tllm_decoder_config, decoder_convert_args, decoder_saved_dir, 1
        )


def convert(
    worker_rank: int,
    world_size: int,
    args: ArgsNamespace,
    model_config: dict[str, Any],
    convert_args: dict[str, Any],
    saved_dir: Path,
    num_workers: int,
) -> None:
    effective_model_type = args.model_type if args.model_type != "blip2" else "t5"
    convert_weights_func_name = f"convert_{effective_model_type}_weights_to_tllm_safetensors"
    convert_weights_func = globals().get(convert_weights_func_name)

    if not callable(convert_weights_func):
        raise NotImplementedError(
            f"Weight conversion func {convert_weights_func_name} not found or not callable."
        )

    for rank in range(worker_rank, world_size, num_workers):
        rank_model_config_copy = copy.deepcopy(model_config)
        rank_trt_llm_config = MinimalConfig.from_dict(rank_model_config_copy)
        rank_trt_llm_config.set_rank(rank)

        LOGGER.info(f"Converting rank {rank}...")
        try:
            rank_convert_args = copy.deepcopy(convert_args)
            weights: WeightDict = convert_weights_func(
                config=rank_trt_llm_config, **rank_convert_args
            )
            output_file = saved_dir / f"rank{rank}.safetensors"
            safetensors.torch.save_file(weights, str(output_file))
            LOGGER.info(f"Saved rank {rank} weights to {output_file}")
        except Exception as e:
            LOGGER.exception(f"Failed to convert weights for rank {rank}: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model_type",
        type=str,
        default="t5",
        choices=["t5", "nmt", "bart", "pix2struct", "blip2"],
        help="Model architecture type.",
    )
    parser.add_argument("--tp_size", type=int, default=1, help="N-way tensor parallelism size")
    parser.add_argument("--pp_size", type=int, default=1, help="N-way pipeline parallelism size")
    parser.add_argument(
        "--model_dir", "-i", type=str, help="Path to the framework checkpoint file", required=True
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Path to the converted TRT-LLM model weight file",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="How many workers to spawn for conversion (default: 1 for safety)",
        default=1,
    )
    parser.add_argument(
        "--nougat", action="store_true", help="Flag for Nougat model (ViT + MBart Decoder)."
    )
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    parser.add_argument(
        "--use_parallel_embedding",
        action="store_true",
        default=False,
        help="Enable embedding parallelism.",
    )
    parser.add_argument(
        "--embedding_sharding_dim",
        type=int,
        default=0,
        choices=[0, 1],
        help="Shard embedding table along vocab (0) or hidden (1) dim.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Target inference dtype.",
    )
    parser.add_argument(
        "--skip_cross_kv",
        action="store_true",
        help="Skip redundant cross QKV computation.",
    )
    parser.add_argument(
        "--use_implicit_relative_attention",
        action="store_true",
        help="Compute relative attention bias on the fly.",
    )
    cli_args: ArgsNamespace = parser.parse_args()

    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    log_level = logging.DEBUG if cli_args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info("\n=============== Argument ===============")
    for key in vars(cli_args):
        LOGGER.info(f"{key}: {vars(cli_args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(cli_args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    LOGGER.info(f"Spend {run_time} (h:m:s) to convert the model")
