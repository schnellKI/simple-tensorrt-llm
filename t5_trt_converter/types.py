"""Type definitions for T5 TensorRT-LLM Converter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

InferencePrecision = Literal["float16", "bfloat16", "float32"]
ModelType = Literal["t5"]  # Extend as needed


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for the TensorRT-LLM conversion process."""

    hf_model_dir: Path
    model_name: str
    model_type: ModelType = "t5"
    inference_precision: InferencePrecision = "bfloat16"
    tp_size: int = 1
    pp_size: int = 1
    max_beam_width: int = 1
    max_encoder_input_len: int = 512
    max_seq_len: int = 512
    max_batch_size: int = 128
    base_dir: Path = Path("/tmp/trt_models")
    engines_base_dir: Path = Path("/tmp/trt_engines2")
    custom_engine_dir: Path | None = None  # User override for engine output
    gpu_arch: str | None = None  # Automatically detected if None


@dataclass(frozen=True)
class DerivedPaths:
    """Paths derived from ConversionConfig."""

    trt_model_dir: Path
    trt_engine_dir: Path
    world_size: int
    max_num_tokens: int
    decoder_max_input_len: int = 1


@dataclass(frozen=True)
class RepoBuildConfig:
    """Configuration for building the Triton repository."""

    script_dir: Path  # Needed to find template files
    triton_repo_dir: Path
    hf_model_dir: Path  # Original HF model dir (potentially overridden for tokenizer)
    trt_engine_dir: Path  # Final engine dir (potentially overridden)
    max_batch_size: int
    max_beam_width: int
    tokenizer_dir: Path | None = None  # User override for tokenizer files
    triton_repo_name: str = "enc_dec_ifb"  # Folder name within triton_repos base
