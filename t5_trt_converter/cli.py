"""Command Line Interface for T5 TensorRT-LLM Converter."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .converter import Converter
from .repo_builder import RepoBuilder
from .types import ConversionConfig, InferencePrecision, ModelType, RepoBuildConfig

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to multiple subcommands."""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level.",
    )


def _add_conversion_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the 'convert' command."""
    parser.add_argument(
        "--hf-model-dir", type=Path, required=True, help="Path to the Hugging Face model directory."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="A name for the model used in output paths (e.g., t5-small).",
    )
    parser.add_argument(
        "--model-type", type=str, default="t5", choices=["t5"], help="Model architecture type."
    )
    parser.add_argument(
        "--inference-precision",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Target inference precision.",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallelism size.")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline Parallelism size.")
    parser.add_argument(
        "--max-beam-width", type=int, default=1, help="Maximum beam width for search."
    )
    parser.add_argument(
        "--max-encoder-input-len", type=int, default=512, help="Max encoder input sequence length."
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max decoder total sequence length (input + output).",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=128, help="Max batch size for engine build."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/tmp/trt_models"),
        help="Base directory for intermediate TensorRT models.",
    )
    parser.add_argument(
        "--engines-base-dir",
        type=Path,
        default=Path("/tmp/trt_engines2"),
        help="Base directory for final TensorRT engines.",
    )
    # Shorten help text line
    parser.add_argument(
        "--custom-engine-dir",
        type=Path,
        help="Optional: Use a specific directory for the final engines instead of default.",
    )
    # gpu_arch is detected automatically, no CLI arg for now


def _add_repo_build_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the 'build-repo' command."""
    # Some args overlap with conversion but are needed here too
    # Shorten help text line
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        required=True,
        help="Path to the original HF model dir (used for tokenizer unless overridden).",
    )
    parser.add_argument(
        "--trt-engine-dir",
        type=Path,
        required=True,
        help="Path to the directory containing the built TensorRT engines.",
    )
    parser.add_argument(
        "--triton-repo-dir",
        type=Path,
        required=True,
        help="Output directory for the Triton repository.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        required=True,
        help="Max batch size used during engine build (must match!).",
    )
    parser.add_argument(
        "--max-beam-width",
        type=int,
        required=True,
        help="Max beam width used during engine build (must match!).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Optional: Use a custom tokenizer directory instead of the HF model directory.",
    )
    parser.add_argument(
        "--triton-repo-name",
        type=str,
        default="enc_dec_ifb",
        help="Name for the Triton repository folder within the base path.",
    )
    # script_dir is derived, not a CLI arg


def handle_convert(args: argparse.Namespace) -> None:
    """Handle the 'convert' subcommand."""
    logger.info("Handling 'convert' command...")
    # Validate args before passing to dataclass
    model_type: ModelType = args.model_type
    inference_precision: InferencePrecision = args.inference_precision

    config = ConversionConfig(
        hf_model_dir=args.hf_model_dir,
        model_name=args.model_name,
        model_type=model_type,
        inference_precision=inference_precision,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        max_beam_width=args.max_beam_width,
        max_encoder_input_len=args.max_encoder_input_len,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        base_dir=args.base_dir,
        engines_base_dir=args.engines_base_dir,
        custom_engine_dir=args.custom_engine_dir,
    )
    try:
        Converter.run_conversion(config)
        logger.info("Conversion process completed successfully.")
    except Exception:
        # Use lazy logging, no need for exception object
        logger.exception("Conversion process failed")
        sys.exit(1)


def handle_build_repo(args: argparse.Namespace) -> None:
    """Handle the 'build-repo' subcommand."""
    logger.info("Handling 'build-repo' command...")
    # Determine the script directory (location of this file)
    # This assumes cli.py is within the t5_trt_converter package
    # and the original scripts/templates are relative to the package's parent
    script_file_dir = Path(__file__).parent
    # Adjust this path if repo_template/fill_template are elsewhere
    assumed_scripts_base_dir = script_file_dir.parent / "trt" / "t5"
    # Use lazy logging
    logger.debug("Assuming script base directory: %s", assumed_scripts_base_dir)

    config = RepoBuildConfig(
        script_dir=assumed_scripts_base_dir,  # Relative path to find templates/scripts
        triton_repo_dir=args.triton_repo_dir,
        hf_model_dir=args.hf_model_dir,
        trt_engine_dir=args.trt_engine_dir,
        max_batch_size=args.max_batch_size,
        max_beam_width=args.max_beam_width,
        tokenizer_dir=args.tokenizer_dir,
        triton_repo_name=args.triton_repo_name,
    )
    try:
        RepoBuilder.build_repository(config)
        logger.info("Triton repository build completed successfully.")
    except Exception:
        # Use lazy logging, no need for exception object
        logger.exception("Triton repository build failed")
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and execute the selected subcommand."""
    parser = argparse.ArgumentParser(
        description="Convert T5 models to TensorRT-LLM and build Triton repos."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # --- Convert Subcommand ---
    parser_convert = subparsers.add_parser(
        "convert", help="Convert a Hugging Face T5 model to TensorRT-LLM engines."
    )
    _add_common_args(parser_convert)
    _add_conversion_args(parser_convert)
    parser_convert.set_defaults(func=handle_convert)

    # --- Build Repo Subcommand ---
    parser_build = subparsers.add_parser(
        "build-repo", help="Build a Triton repository from existing TensorRT-LLM engines."
    )
    _add_common_args(parser_build)
    _add_repo_build_args(parser_build)
    parser_build.set_defaults(func=handle_build_repo)

    args = parser.parse_args(argv)  # Pass argv for testability

    # Set logging level based on args
    log_level_str = args.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Get root logger and set level
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Also set level for existing handlers
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    # Use lazy logging
    logger.info("Set log level to: %s", log_level_str)

    # Call the handler function associated with the subcommand
    args.func(args)


if __name__ == "__main__":
    main()
