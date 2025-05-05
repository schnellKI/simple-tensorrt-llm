"""Handles the construction of the Triton Inference Server repository."""

import logging
import shutil
from pathlib import Path

# Import the target function using relative path
from .fill_template import main as fill_template_main
from .types import RepoBuildConfig

logger = logging.getLogger(__name__)


class RepoBuilder:
    """Handles the construction of the Triton Inference Server repository."""

    @staticmethod
    def _resolve_tokenizer_dir(config: RepoBuildConfig) -> Path:
        """Determine the correct tokenizer directory to use."""
        tokenizer_dir = config.tokenizer_dir or config.hf_model_dir
        if not tokenizer_dir.exists():
            # Use lazy formatting
            logger.error("Tokenizer directory not found: %s", tokenizer_dir)
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
        # Use lazy formatting
        logger.info("Using tokenizer directory: %s", tokenizer_dir)
        return tokenizer_dir

    @staticmethod
    def _get_template_path(config: RepoBuildConfig) -> Path:
        """Get the path to the repository template directory."""
        # Assumes repo_template is in the same directory as the original scripts
        # This might need adjustment based on final package structure
        template_path = config.script_dir / "repo_template"
        if not template_path.is_dir():
            # Use lazy formatting
            logger.error("Repository template directory not found: %s", template_path)
            raise FileNotFoundError(f"Repository template directory not found: {template_path}")
        return template_path

    @staticmethod
    def copy_template(config: RepoBuildConfig, template_path: Path) -> None:
        """Copy the template directory to the target Triton repo directory."""
        target_repo_path = config.triton_repo_dir
        # Use lazy formatting
        logger.info("Copying template from %s to %s", template_path, target_repo_path)
        if target_repo_path.exists():
            # Use lazy formatting and shorten line
            logger.warning(
                "Target repository directory %s already exists. Removing.", target_repo_path
            )
            shutil.rmtree(target_repo_path)
        # Create parent directories if they don't exist
        target_repo_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(template_path, target_repo_path)

    @staticmethod
    def fill_templates(config: RepoBuildConfig, tokenizer_dir: Path) -> None:
        """Fill the placeholder values in the copied template config files."""
        logger.info("Generating Triton server configurations...")

        # Define template substitutions (matching the shell script logic)
        substitutions = {
            "tensorrt_llm/config.pbtxt": {
                "triton_backend": "tensorrtllm",
                "triton_max_batch_size": str(config.max_batch_size),
                "decoupled_mode": "False",
                "max_beam_width": str(config.max_beam_width),
                "engine_dir": str(config.trt_engine_dir / "decoder"),
                "encoder_engine_dir": str(config.trt_engine_dir / "encoder"),
                "kv_cache_free_gpu_mem_fraction": "0.8",
                "cross_kv_cache_fraction": "0.5",
                "exclude_input_in_output": "True",
                "enable_kv_cache_reuse": "False",
                "batching_strategy": "inflight_fused_batching",
                "max_queue_delay_microseconds": "0",
                "enable_chunked_context": "False",
                "max_queue_size": "0",
                "encoder_input_features_data_type": "TYPE_FP16",
                "logits_datatype": "TYPE_FP32",
            },
            "preprocessing/config.pbtxt": {
                "tokenizer_dir": str(tokenizer_dir),
                "triton_max_batch_size": str(config.max_batch_size),
                "preprocessing_instance_count": "1",
            },
            "postprocessing/config.pbtxt": {
                "tokenizer_dir": str(tokenizer_dir),
                "triton_max_batch_size": str(config.max_batch_size),
                "postprocessing_instance_count": "1",
            },
            "ensemble/config.pbtxt": {
                "triton_max_batch_size": str(config.max_batch_size),
                "logits_datatype": "TYPE_FP32",
            },
            "tensorrt_llm_bls/config.pbtxt": {
                "triton_max_batch_size": str(config.max_batch_size),
                "decoupled_mode": "False",
                "bls_instance_count": "1",
                "accumulate_tokens": "False",
                "logits_datatype": "TYPE_FP32",
            },
        }

        for template_file, params in substitutions.items():
            target_file_path = config.triton_repo_dir / template_file
            # Format params into the key:value,key:value string expected by fill_template.main
            params_str = ",".join([f"{k}:{v}" for k, v in params.items()])

            # Use lazy formatting
            logger.debug("Filling template for %s with params: %s", target_file_path, params_str)
            try:
                # Call the imported function directly
                fill_template_main(
                    file_path=str(target_file_path), substitutions=params_str, in_place=True
                )
            except FileNotFoundError:
                # This might occur if the target_file_path (from template copy) is missing
                logger.exception(
                    "Error: Template file not found for substitution: %s", target_file_path
                )
                raise
            except AssertionError as e:
                # fill_template.main uses assert key in pbtxt.template
                logger.exception("Error filling template %s: %s", target_file_path, e)
                raise
            except Exception:
                # Catch other potential errors from the script
                logger.exception(
                    "Template filling failed for %s (Params: %s)", target_file_path, params_str
                )
                raise
        logger.info("Template filling complete.")

    @staticmethod
    def delete_unused_files(config: RepoBuildConfig) -> None:
        """Delete unused files/directories like the BLS model and model.py."""
        bls_path = config.triton_repo_dir / "tensorrt_llm_bls"
        unused_model_py = config.triton_repo_dir / "tensorrt_llm" / "model.py"

        # Use lazy formatting
        logger.info("Deleting unused BLS path: %s", bls_path)
        if bls_path.exists():
            shutil.rmtree(bls_path)
        else:
            logger.warning("Path not found, skipping deletion: %s", bls_path)

        # Use lazy formatting
        logger.info("Deleting unused model.py: %s", unused_model_py)
        if unused_model_py.exists():
            unused_model_py.unlink()
        else:
            # Use lazy formatting
            logger.warning("File not found, skipping deletion: %s", unused_model_py)

    @staticmethod
    def build_repository(config: RepoBuildConfig) -> None:
        """Run the full Triton repository build process."""
        logger.info("Building Triton repository...")
        # Use lazy formatting
        logger.info("Target Repository Dir: %s", config.triton_repo_dir)
        logger.info("Using engine directory: %s", config.trt_engine_dir)

        tokenizer_dir = RepoBuilder._resolve_tokenizer_dir(config)
        template_path = RepoBuilder._get_template_path(config)

        RepoBuilder.copy_template(config, template_path)
        RepoBuilder.fill_templates(config, tokenizer_dir)
        RepoBuilder.delete_unused_files(config)

        # Use lazy formatting
        logger.info("Triton repository setup complete in %s", config.triton_repo_dir)
