# Placeholder for converter tests
import argparse  # Import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the package root to the path
PACKAGE_ROOT = Path(__file__).parent.parent
SRC_PATH = PACKAGE_ROOT / "t5_trt_converter"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from t5_trt_converter.converter import Converter
from t5_trt_converter.types import ConversionConfig, DerivedPaths

# --- Fixtures ---


@pytest.fixture
def base_conversion_config(tmp_path: Path) -> ConversionConfig:
    """Provides a base ConversionConfig using temporary paths."""
    hf_dir = tmp_path / "hf_model"
    hf_dir.mkdir()
    return ConversionConfig(
        hf_model_dir=hf_dir,
        model_name="test-t5",
        base_dir=tmp_path / "trt_models",
        engines_base_dir=tmp_path / "trt_engines",
        # Keep other defaults
    )


@pytest.fixture
def derived_paths_for_config(base_conversion_config: ConversionConfig) -> DerivedPaths:
    """Calculates DerivedPaths based on the base_conversion_config fixture."""
    # Manually calculate for testing, assuming default GPU arch if needed
    gpu_arch = "sm80"  # Assume default for testing path calculation
    world_size = base_conversion_config.tp_size * base_conversion_config.pp_size
    max_num_tokens = base_conversion_config.max_seq_len * base_conversion_config.max_batch_size
    # Add default decoder_max_input_len from DerivedPaths
    decoder_max_input_len = 1  # Default from DerivedPaths

    trt_model_dir = (
        base_conversion_config.base_dir
        / base_conversion_config.model_name
        / gpu_arch
        / base_conversion_config.inference_precision
        / f"{world_size}-gpu"
    )
    trt_engine_dir = (
        base_conversion_config.engines_base_dir
        / base_conversion_config.model_name
        / gpu_arch
        / base_conversion_config.inference_precision
        / f"{world_size}-gpu"
    )

    return DerivedPaths(
        trt_model_dir=trt_model_dir,
        trt_engine_dir=trt_engine_dir,
        world_size=world_size,
        max_num_tokens=max_num_tokens,
        decoder_max_input_len=decoder_max_input_len,  # Add this
    )


# --- Test Cases ---


# Patch the directly imported convert_checkpoint function within the converter module
@patch("t5_trt_converter.converter.convert_checkpoint")
def test_convert_checkpoint_invoked_when_files_missing(
    mock_convert_checkpoint_func: MagicMock,  # Updated mock name
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that convert_checkpoint function is called when config files don't exist."""
    # Arrange
    # Ensure target config files *don't* exist (default for tmp_path)
    paths = derived_paths_for_config
    paths.trt_model_dir.mkdir(parents=True)  # Create parent but not files
    (paths.trt_model_dir / "encoder").mkdir()
    (paths.trt_model_dir / "decoder").mkdir()

    # Act
    Converter.convert_checkpoint(base_conversion_config, paths)

    # Assert
    mock_convert_checkpoint_func.assert_called_once()  # Check the imported function mock
    call_args = mock_convert_checkpoint_func.call_args[0][0]  # Get the args namespace
    assert isinstance(call_args, argparse.Namespace)
    assert call_args.model_type == base_conversion_config.model_type
    assert call_args.model_dir == str(base_conversion_config.hf_model_dir)
    assert call_args.output_dir == str(paths.trt_model_dir)
    assert call_args.tp_size == base_conversion_config.tp_size
    assert call_args.pp_size == base_conversion_config.pp_size
    assert call_args.dtype == base_conversion_config.inference_precision
    # Check a default arg example
    assert call_args.workers == 4


# Patch the directly imported function
@patch("t5_trt_converter.converter.convert_checkpoint")
def test_convert_checkpoint_skipped_when_files_exist(
    mock_convert_checkpoint_func: MagicMock,  # Updated mock name
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that convert_checkpoint function is NOT called when config files exist."""
    # Arrange
    paths = derived_paths_for_config
    # Create the target config files
    (paths.trt_model_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (paths.trt_model_dir / "decoder").mkdir(parents=True, exist_ok=True)
    (paths.trt_model_dir / "encoder" / "config.json").touch()
    (paths.trt_model_dir / "decoder" / "config.json").touch()

    # Act
    Converter.convert_checkpoint(base_conversion_config, paths)

    # Assert
    mock_convert_checkpoint_func.assert_not_called()  # Check the imported function mock


# Tests for build_encoder/decoder still use CommandInvoker.run for trtllm-build
@patch("t5_trt_converter.converter.CommandInvoker.run")
def test_build_encoder_engine_invoked_when_file_missing(
    mock_run: MagicMock,
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that _invoke_trtllm_build is called for encoder when engine file doesn't exist."""
    # Arrange
    paths = derived_paths_for_config
    (paths.trt_engine_dir / "encoder").mkdir(parents=True, exist_ok=True)
    # Ensure engine file does NOT exist

    # Act
    Converter.build_encoder_engine(base_conversion_config, paths)

    # Assert
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]  # Get the cmd list
    assert call_args[0] == "trtllm-build"
    assert "--output_dir" in call_args
    assert str(paths.trt_engine_dir / "encoder") in call_args
    assert "--checkpoint_dir" in call_args
    assert str(paths.trt_model_dir / "encoder") in call_args


@patch("t5_trt_converter.converter.CommandInvoker.run")
def test_build_encoder_engine_skipped_when_file_exists(
    mock_run: MagicMock,
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that _invoke_trtllm_build is NOT called for encoder when engine file exists."""
    # Arrange
    paths = derived_paths_for_config
    # Create the target engine file
    (paths.trt_engine_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (paths.trt_engine_dir / "encoder" / "rank0.engine").touch()

    # Act
    Converter.build_encoder_engine(base_conversion_config, paths)

    # Assert
    mock_run.assert_not_called()


@patch("t5_trt_converter.converter.CommandInvoker.run")
def test_build_decoder_engine_invoked_when_file_missing(
    mock_run: MagicMock,
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that _invoke_trtllm_build is called for decoder when engine file doesn't exist."""
    # Arrange
    paths = derived_paths_for_config
    (paths.trt_engine_dir / "decoder").mkdir(parents=True, exist_ok=True)
    # Ensure engine file does NOT exist

    # Act
    Converter.build_decoder_engine(base_conversion_config, paths)

    # Assert
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]  # Get the cmd list
    assert call_args[0] == "trtllm-build"
    assert "--output_dir" in call_args
    assert str(paths.trt_engine_dir / "decoder") in call_args
    assert "--checkpoint_dir" in call_args
    assert str(paths.trt_model_dir / "decoder") in call_args


@patch("t5_trt_converter.converter.CommandInvoker.run")
def test_build_decoder_engine_skipped_when_file_exists(
    mock_run: MagicMock,
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that _invoke_trtllm_build is NOT called for decoder when engine file exists."""
    # Arrange
    paths = derived_paths_for_config
    # Create the target engine file
    (paths.trt_engine_dir / "decoder").mkdir(parents=True, exist_ok=True)
    (paths.trt_engine_dir / "decoder" / "rank0.engine").touch()

    # Act
    Converter.build_decoder_engine(base_conversion_config, paths)

    # Assert
    mock_run.assert_not_called()


# Test error handling for trtllm-build call
@patch("t5_trt_converter.converter.CommandInvoker.run")
def test_build_encoder_engine_raises_error(
    mock_run: MagicMock,
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that errors from CommandInvoker.run are propagated."""
    # Arrange
    paths = derived_paths_for_config
    (paths.trt_engine_dir / "encoder").mkdir(parents=True, exist_ok=True)
    # Simulate CommandInvoker.run failing
    mock_run.side_effect = subprocess.CalledProcessError(1, ["trtllm-build", "..."])

    # Act & Assert
    with pytest.raises(subprocess.CalledProcessError):
        Converter.build_encoder_engine(base_conversion_config, paths)
    mock_run.assert_called_once()  # Ensure it was actually called


# Test the main run_conversion orchestrator
@patch("t5_trt_converter.converter.Converter.create_directories")
@patch("t5_trt_converter.converter.Converter.convert_checkpoint")
@patch("t5_trt_converter.converter.Converter.build_encoder_engine")
@patch("t5_trt_converter.converter.Converter.build_decoder_engine")
@patch("t5_trt_converter.converter.Converter.calculate_derived_paths")
def test_run_conversion_calls_steps(
    mock_calc_paths: MagicMock,
    mock_build_decoder: MagicMock,
    mock_build_encoder: MagicMock,
    mock_convert_ckpt: MagicMock,
    mock_create_dirs: MagicMock,
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
) -> None:
    """Test that run_conversion calls all the required steps in order."""
    # Arrange
    # Make calculate_derived_paths return our fixture value
    mock_calc_paths.return_value = derived_paths_for_config

    # Act
    result_paths = Converter.run_conversion(base_conversion_config)

    # Assert
    assert result_paths == derived_paths_for_config
    mock_calc_paths.assert_called_once_with(base_conversion_config)
    mock_create_dirs.assert_called_once_with(derived_paths_for_config)
    mock_convert_ckpt.assert_called_once_with(base_conversion_config, derived_paths_for_config)
    mock_build_encoder.assert_called_once_with(base_conversion_config, derived_paths_for_config)
    mock_build_decoder.assert_called_once_with(base_conversion_config, derived_paths_for_config)


# Add a test specifically for the error handling of the direct function call
@patch("t5_trt_converter.converter.convert_checkpoint")
def test_convert_checkpoint_function_raises_error(
    mock_convert_checkpoint_func: MagicMock,
    base_conversion_config: ConversionConfig,
    derived_paths_for_config: DerivedPaths,
    tmp_path: Path,
) -> None:
    """Test that errors from the imported convert_checkpoint function are propagated."""
    # Arrange
    paths = derived_paths_for_config
    (paths.trt_model_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (paths.trt_model_dir / "decoder").mkdir(parents=True, exist_ok=True)
    # Simulate the imported function failing
    mock_convert_checkpoint_func.side_effect = ValueError("Simulated function error")

    # Act & Assert
    with pytest.raises(ValueError, match="Simulated function error"):
        Converter.convert_checkpoint(base_conversion_config, paths)
    mock_convert_checkpoint_func.assert_called_once()  # Ensure it was actually called
