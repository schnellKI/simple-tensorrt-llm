"""Handles the conversion of Hugging Face T5 models to TensorRT-LLM engines."""

import argparse  # Add argparse import
import logging
import subprocess
from typing import Any

# Assuming convert_checkpoint.py is importable relative to this file's location
# Adjust the import path based on your project structure if needed.
# Example: from trt.t5.convert_checkpoint import convert_checkpoint
# If trt is a top-level directory in your PYTHONPATH or installed package:
from .convert_checkpoint import convert_checkpoint
from .types import ConversionConfig, DerivedPaths

logger = logging.getLogger(__name__)

# Minimum supported compute capability (e.g., Volta)
MIN_COMPUTE_CAPABILITY = 8


class CommandInvoker:
    """Helper class to invoke external commands, allowing for easier mocking."""

    @staticmethod
    def run(
        cmd: list[str],
        *,  # Make subsequent args keyword-only
        check: bool = True,
        capture_output: bool = True,
        text: bool = True,
        **kwargs: Any,  # Add type hint for kwargs
    ) -> subprocess.CompletedProcess[str]:  # Add type arg for CompletedProcess
        """Run a command using subprocess.run and log details.

        Args:
            cmd: The command and its arguments as a list of strings.
            check: If True, raise CalledProcessError on non-zero exit.
            capture_output: If True, capture stdout and stderr.
            text: If True, decode stdout/stderr as text.
            **kwargs: Additional keyword arguments for subprocess.run.

        Returns:
            The CompletedProcess object.

        Raises:
            FileNotFoundError: If the command is not found.
            subprocess.CalledProcessError: If cmd returns non-zero and check=True.
        """
        # Ensure command parts are strings and reasonably safe (basic check)
        safe_cmd = [str(part) for part in cmd]
        cmd_str = " ".join(safe_cmd)
        logger.debug("Running command: %s", cmd_str)
        result: subprocess.CompletedProcess[str] | None = None  # Initialize result
        try:
            # S603: Use safe_cmd; assume inputs forming cmd are trusted or sanitized elsewhere.
            result = subprocess.run(
                safe_cmd, check=check, capture_output=capture_output, text=text, **kwargs
            )
        except FileNotFoundError:
            # Use lazy formatting
            logger.exception(
                "Error: Command not found - %s. Ensure it is installed and in PATH.", safe_cmd[0]
            )
            raise  # Use bare raise (TRY201 fix)
        except subprocess.CalledProcessError:
            # Use lazy formatting, no need to pass exception object explicitly (TRY401 fix)
            logger.exception(
                "Command failed: %s", cmd_str
            )  # Removed return code from msg for simplicity
            # Logging output is handled by run_conversion if needed, keep invoker simple
            raise  # Use bare raise (TRY201 fix)
        else:  # TRY300 fix - return in else block
            logger.debug("Command finished: %s", cmd_str)
            # Log stdout/stderr only if capture_output is True and there was output
            if capture_output and result:
                # Use lazy formatting for logs
                if result.stdout:
                    logger.debug("Stdout:\n%s", result.stdout.strip())
                if result.stderr:
                    logger.debug("Stderr:\n%s", result.stderr.strip())
            # We can be sure result is not None here because if subprocess.run failed,
            # an exception would have been raised.
            assert result is not None
            return result


class Converter:
    """Handles the conversion of Hugging Face T5 models to TensorRT-LLM engines."""

    @staticmethod
    def _get_gpu_arch() -> str:
        """Detect the GPU architecture using nvidia-smi."""
        compute_cap_str: str | None = None
        try:
            result = CommandInvoker.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
            )
            compute_cap_str = result.stdout.strip().split(".")[0]
            compute_cap = int(compute_cap_str)
            # Use constant for magic value
            if compute_cap >= MIN_COMPUTE_CAPABILITY:
                # Use lazy formatting
                logger.debug("Detected GPU Compute Capability: %s", compute_cap)
            else:
                logger.warning(
                    "Unsupported GPU Arch (Compute Capability %s). Defaulting sm80.",
                    compute_cap_str,
                )
                return "sm80"
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            # Use lazy formatting, no need for exception object (TRY401 fix)
            logger.exception("Failed to detect GPU architecture. Defaulting to sm80.")
            return "sm80"
        else:
            # TRY300 fix: Return success case in else block
            # compute_cap_str cannot be None here if no exception occurred
            assert compute_cap_str is not None
            return f"sm{compute_cap_str}"

    @staticmethod
    def calculate_derived_paths(config: ConversionConfig) -> DerivedPaths:
        """Calculate derived paths based on the conversion configuration."""
        gpu_arch = config.gpu_arch or Converter._get_gpu_arch()
        world_size = config.tp_size * config.pp_size
        max_num_tokens = config.max_seq_len * config.max_batch_size

        base_model_path = (
            config.base_dir
            / config.model_name
            / gpu_arch
            / config.inference_precision
            / f"{world_size}-gpu"
        )
        base_engine_path = (
            config.engines_base_dir
            / config.model_name
            / gpu_arch
            / config.inference_precision
            / f"{world_size}-gpu"
        )

        trt_model_dir = base_model_path
        trt_engine_dir = config.custom_engine_dir or base_engine_path

        # Use lazy formatting
        logger.info("Calculated TRT Model Dir: %s", trt_model_dir)
        logger.info("Calculated TRT Engine Dir: %s", trt_engine_dir)

        return DerivedPaths(
            trt_model_dir=trt_model_dir,
            trt_engine_dir=trt_engine_dir,
            world_size=world_size,
            max_num_tokens=max_num_tokens,
        )

    @staticmethod
    def create_directories(paths: DerivedPaths) -> None:
        """Create the necessary output directories for models and engines."""
        logger.info("Creating output directories...")
        dirs_to_create = [
            paths.trt_model_dir / "encoder",
            paths.trt_model_dir / "decoder",
            paths.trt_engine_dir / "encoder",
            paths.trt_engine_dir / "decoder",
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Use lazy formatting
            logger.debug("Ensured directory exists: %s", dir_path)

    @staticmethod
    def _invoke_convert_checkpoint(config: ConversionConfig, paths: DerivedPaths) -> None:
        """Invoke the convert_checkpoint function directly."""
        logger.info("Calling convert_checkpoint function directly...")

        # --- Construct args namespace mimicking argparse ---
        # Defaults from convert_checkpoint.py's parser (as of review)
        # We might need to refine these or make them configurable if they change
        args = argparse.Namespace(
            model_type=config.model_type,
            model_dir=str(config.hf_model_dir),
            output_dir=str(paths.trt_model_dir),
            tp_size=config.tp_size,
            pp_size=config.pp_size,
            dtype=config.inference_precision,
            workers=4,  # Default from script, make configurable?
            nougat=False,  # Default from script
            verbose=False,  # Default from script, link to our logger level?
            use_parallel_embedding=False,  # Default
            embedding_sharding_dim=0,  # Default
            use_weight_only=False,  # Default
            weight_only_precision="int8",  # Default
            skip_cross_kv=False,  # Default
            use_implicit_relative_attention=False,  # Default
            # Add any other args expected by convert_checkpoint if needed
        )
        # TODO: Consider how to handle 'verbose' - map from our logger?
        # TODO: Consider making 'workers' configurable in ConversionConfig?

        try:
            # Call the function directly
            convert_checkpoint(args)
        except Exception:  # Catch a broader range of potential issues
            # Use logging.exception here
            logger.exception(
                "Checkpoint conversion function call failed (args: %s).",
                vars(args),  # Log the args used
            )
            raise  # Re-raise the caught exception
        else:
            logger.info("Checkpoint conversion function call successful.")

    @staticmethod
    def convert_checkpoint(config: ConversionConfig, paths: DerivedPaths) -> None:
        """Convert the Hugging Face checkpoint, skipping if already done."""
        # Use lazy formatting
        logger.info(
            "Attempting checkpoint conversion from %s to %s",
            config.hf_model_dir,
            paths.trt_model_dir,
        )
        output_encoder_config = paths.trt_model_dir / "encoder" / "config.json"
        output_decoder_config = paths.trt_model_dir / "decoder" / "config.json"

        if output_encoder_config.exists() and output_decoder_config.exists():
            # Use lazy formatting and shorten line
            logger.info(
                "TensorRT-LLM checkpoint already exists in %s. Skipping.", paths.trt_model_dir
            )
            return

        Converter._invoke_convert_checkpoint(config, paths)

    @staticmethod
    def _invoke_trtllm_build(cmd_args: list[str]) -> None:
        """Invoke the trtllm-build command with specific arguments."""
        # Use list unpacking (RUF005 fix)
        cmd = ["trtllm-build", *cmd_args]
        try:
            CommandInvoker.run(cmd)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Use logging.exception (TRY400 fix)
            logger.exception("trtllm-build command failed.")
            raise  # Use bare raise (TRY201 fix)

    @staticmethod
    def build_encoder_engine(config: ConversionConfig, paths: DerivedPaths) -> None:
        """Build the TensorRT engine for the encoder."""
        # Use lazy formatting
        engine_path = paths.trt_engine_dir / "encoder"
        logger.info("Building encoder engine in %s", engine_path)
        encoder_engine_file = engine_path / "rank0.engine"

        if encoder_engine_file.exists():
            logger.info("Encoder engine already exists. Skipping build.")
            return

        cmd_args = [
            "--checkpoint_dir",
            str(paths.trt_model_dir / "encoder"),
            "--output_dir",
            str(engine_path),
            "--paged_kv_cache",
            "disable",
            "--moe_plugin",
            "disable",
            "--max_beam_width",
            str(config.max_beam_width),
            "--max_batch_size",
            str(config.max_batch_size),
            "--max_num_tokens",
            str(paths.max_num_tokens),
            "--max_input_len",
            str(config.max_encoder_input_len),
            "--gemm_plugin",
            config.inference_precision,
            "--bert_attention_plugin",
            config.inference_precision,
            "--gpt_attention_plugin",
            config.inference_precision,
            "--remove_input_padding",
            "enable",
            "--multiple_profiles",
            "enable",
            "--context_fmha",
            "disable",
        ]
        # Simplified error handling (TRY203 fix): rely on _invoke method
        Converter._invoke_trtllm_build(cmd_args)
        logger.info("Encoder engine build successful.")

    @staticmethod
    def build_decoder_engine(config: ConversionConfig, paths: DerivedPaths) -> None:
        """Build the TensorRT engine for the decoder."""
        # Use lazy formatting
        engine_path = paths.trt_engine_dir / "decoder"
        logger.info("Building decoder engine in %s", engine_path)
        decoder_engine_file = engine_path / "rank0.engine"

        if decoder_engine_file.exists():
            logger.info("Decoder engine already exists. Skipping build.")
            return

        cmd_args = [
            "--checkpoint_dir",
            str(paths.trt_model_dir / "decoder"),
            "--output_dir",
            str(engine_path),
            "--moe_plugin",
            "disable",
            "--max_beam_width",
            str(config.max_beam_width),
            "--max_batch_size",
            str(config.max_batch_size),
            "--max_input_len",
            str(paths.decoder_max_input_len),
            "--max_seq_len",
            str(config.max_seq_len),
            "--max_encoder_input_len",
            str(config.max_encoder_input_len),
            "--gemm_plugin",
            config.inference_precision,
            "--bert_attention_plugin",
            config.inference_precision,
            "--gpt_attention_plugin",
            config.inference_precision,
            "--remove_input_padding",
            "enable",
            "--multiple_profiles",
            "enable",
            "--context_fmha",
            "disable",
        ]
        # Simplified error handling (TRY203 fix): rely on _invoke method
        Converter._invoke_trtllm_build(cmd_args)
        logger.info("Decoder engine build successful.")

    @staticmethod
    def run_conversion(config: ConversionConfig) -> DerivedPaths:
        """Run the full conversion pipeline."""
        logger.info("Starting T5 Conversion and Build Process")
        logger.info("----------------------------------------")
        # Use lazy formatting
        logger.info("HF Model Dir:        %s", config.hf_model_dir)
        logger.info("Model Name:          %s", config.model_name)
        # TODO: Log other config details similarly

        derived_paths = Converter.calculate_derived_paths(config)
        logger.info("TRT Model Dir:       %s", derived_paths.trt_model_dir)
        logger.info("TRT Engine Dir:      %s", derived_paths.trt_engine_dir)
        logger.info("----------------------------------------")

        Converter.create_directories(derived_paths)
        # These methods now handle skipping and invoking the isolated calls
        Converter.convert_checkpoint(config, derived_paths)
        Converter.build_encoder_engine(config, derived_paths)
        Converter.build_decoder_engine(config, derived_paths)

        logger.info("----------------------------------------")
        logger.info("T5 Conversion and Build Process Completed Successfully!")
        logger.info("Engines are located in: %s", derived_paths.trt_engine_dir)
        logger.info("----------------------------------------")
        return derived_paths
