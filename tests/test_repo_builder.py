# Placeholder for repo builder tests
import sys
from pathlib import Path

import pytest

# Add the package root to the path for imports if tests are run directly
# This might be handled by pytest config or IDE settings as well
PACKAGE_ROOT = Path(__file__).parent.parent
SRC_PATH = PACKAGE_ROOT / "t5_trt_converter"
TRT_T5_PATH = PACKAGE_ROOT / "trt" / "t5"  # Path to actual scripts/templates
if str(PACKAGE_ROOT) not in sys.path:
    # Add package root for t5_trt_converter imports
    sys.path.insert(0, str(PACKAGE_ROOT))
# Add trt/t5 path if fill_template.py needs to be imported directly (unlikely now)
# if str(TRT_T5_PATH) not in sys.path:
#     sys.path.insert(0, str(TRT_T5_PATH))

from t5_trt_converter.repo_builder import RepoBuilder
from t5_trt_converter.types import RepoBuildConfig

# Import the function that is now called directly
# from .fill_template import main as fill_template_main # Now imported via RepoBuilder

# --- Fixtures ---

# Removed mock_scripts_dir fixture


@pytest.fixture
def base_repo_config(tmp_path: Path) -> RepoBuildConfig:
    """Provides a base RepoBuildConfig using temporary paths and actual script/template path."""
    hf_dir = tmp_path / "hf_model"
    engine_dir = tmp_path / "trt_engines"
    repo_dir = tmp_path / "triton_repo_out"
    hf_dir.mkdir()
    (engine_dir / "encoder").mkdir(parents=True)
    (engine_dir / "decoder").mkdir(parents=True)
    # Create dummy engine files if needed for path checks
    (engine_dir / "encoder" / "rank0.engine").touch()
    (engine_dir / "decoder" / "rank0.engine").touch()

    # --- Use actual script path ---
    actual_script_dir = TRT_T5_PATH
    # Basic check that expected items exist in the actual script dir
    assert (actual_script_dir / "repo_template").is_dir(), (
        f"repo_template not found in {actual_script_dir}"
    )
    # Removed check for fill_template.py in the original script_dir
    # assert (actual_script_dir / "fill_template.py").is_file(), (
    #     f"fill_template.py not found in {actual_script_dir}"
    # )

    return RepoBuildConfig(
        script_dir=actual_script_dir,  # Use the real path
        triton_repo_dir=repo_dir,
        hf_model_dir=hf_dir,
        trt_engine_dir=engine_dir,
        max_batch_size=128,
        max_beam_width=4,
    )


# --- Test Cases ---
# Since these tests check the final output files, they might not need mocks
# for the fill_template call unless we want to isolate RepoBuilder logic.
# For now, we assume calling the real fill_template is acceptable for these tests.


def test_build_repository_creates_structure(base_repo_config: RepoBuildConfig) -> None:
    """Test that the basic repository structure is created and cleanup happens."""
    # Removed chmod call as fill_template is imported directly
    # fill_script = base_repo_config.script_dir / "fill_template.py"
    # fill_script.chmod(0o755)  # No longer needed

    RepoBuilder.build_repository(base_repo_config)

    repo_dir = base_repo_config.triton_repo_dir
    assert repo_dir.exists()
    # Check expected directories exist
    assert (repo_dir / "preprocessing").is_dir()
    assert (repo_dir / "postprocessing").is_dir()
    assert (repo_dir / "tensorrt_llm").is_dir()
    assert (repo_dir / "ensemble").is_dir()

    # Check that cleaned-up items do NOT exist
    assert not (repo_dir / "tensorrt_llm_bls").exists()
    assert not (repo_dir / "tensorrt_llm" / "model.py").exists()

    # Check that only the expected directories exist at the top level
    expected_dirs = {"preprocessing", "postprocessing", "tensorrt_llm", "ensemble"}
    actual_items = {item.name for item in repo_dir.iterdir() if item.is_dir() or item.is_file()}
    # Allow for potential hidden files like .DS_Store, filter them if necessary
    actual_items = {item for item in actual_items if not item.startswith(".")}
    assert actual_items == expected_dirs


def test_config_paths_default(base_repo_config: RepoBuildConfig) -> None:
    """Test that default engine and tokenizer paths are correctly set in configs."""
    # Removed chmod call
    # fill_script = base_repo_config.script_dir / "fill_template.py"
    # fill_script.chmod(0o755)
    RepoBuilder.build_repository(base_repo_config)

    repo_dir = base_repo_config.triton_repo_dir
    trt_llm_config = repo_dir / "tensorrt_llm" / "config.pbtxt"
    pre_config = repo_dir / "preprocessing" / "config.pbtxt"
    post_config = repo_dir / "postprocessing" / "config.pbtxt"

    # Check engine paths
    expected_decoder_path = base_repo_config.trt_engine_dir / "decoder"
    expected_encoder_path = base_repo_config.trt_engine_dir / "encoder"
    trt_config_content = trt_llm_config.read_text()
    assert f'string_value: "{expected_decoder_path}"' in trt_config_content
    assert f'string_value: "{expected_encoder_path}"' in trt_config_content

    # Check tokenizer path (should be hf_model_dir by default)
    expected_tokenizer_path = base_repo_config.hf_model_dir
    pre_config_content = pre_config.read_text()
    post_config_content = post_config.read_text()
    assert f'string_value: "{expected_tokenizer_path}"' in pre_config_content
    assert f'string_value: "{expected_tokenizer_path}"' in post_config_content


def test_config_paths_custom_engine_dir(tmp_path: Path, base_repo_config: RepoBuildConfig) -> None:
    """Test that custom engine paths are correctly set in the config."""
    # Removed chmod call
    # fill_script = base_repo_config.script_dir / "fill_template.py"
    # fill_script.chmod(0o755)
    custom_engine_dir = tmp_path / "custom_engines"
    (custom_engine_dir / "encoder").mkdir(parents=True)
    (custom_engine_dir / "decoder").mkdir(parents=True)

    # Create a new config with the custom engine dir
    config = RepoBuildConfig(
        script_dir=base_repo_config.script_dir,
        triton_repo_dir=base_repo_config.triton_repo_dir,  # Use same output dir
        hf_model_dir=base_repo_config.hf_model_dir,
        trt_engine_dir=custom_engine_dir,  # <-- Override engine dir
        max_batch_size=base_repo_config.max_batch_size,
        max_beam_width=base_repo_config.max_beam_width,
    )

    RepoBuilder.build_repository(config)

    repo_dir = config.triton_repo_dir
    trt_llm_config = repo_dir / "tensorrt_llm" / "config.pbtxt"

    # Check engine paths point to custom dir
    expected_decoder_path = custom_engine_dir / "decoder"
    expected_encoder_path = custom_engine_dir / "encoder"
    trt_config_content = trt_llm_config.read_text()
    assert f'string_value: "{expected_decoder_path}"' in trt_config_content
    assert f'string_value: "{expected_encoder_path}"' in trt_config_content


def test_config_paths_custom_tokenizer_dir(
    tmp_path: Path, base_repo_config: RepoBuildConfig
) -> None:
    """Test that custom tokenizer paths are correctly set in configs."""
    # Removed chmod call
    # fill_script = base_repo_config.script_dir / "fill_template.py"
    # fill_script.chmod(0o755)
    custom_tokenizer_dir = tmp_path / "custom_tokenizer"
    custom_tokenizer_dir.mkdir()

    config = RepoBuildConfig(
        script_dir=base_repo_config.script_dir,
        triton_repo_dir=base_repo_config.triton_repo_dir,
        hf_model_dir=base_repo_config.hf_model_dir,
        trt_engine_dir=base_repo_config.trt_engine_dir,
        max_batch_size=base_repo_config.max_batch_size,
        max_beam_width=base_repo_config.max_beam_width,
        tokenizer_dir=custom_tokenizer_dir,  # <-- Override tokenizer dir
    )

    RepoBuilder.build_repository(config)

    repo_dir = config.triton_repo_dir
    pre_config = repo_dir / "preprocessing" / "config.pbtxt"
    post_config = repo_dir / "postprocessing" / "config.pbtxt"

    # Check tokenizer path points to custom dir
    expected_tokenizer_path = custom_tokenizer_dir
    pre_config_content = pre_config.read_text()
    post_config_content = post_config.read_text()
    assert f'string_value: "{expected_tokenizer_path}"' in pre_config_content
    assert f'string_value: "{expected_tokenizer_path}"' in post_config_content


# Optional: Add a test to mock fill_template.main if needed to isolate RepoBuilder
# @patch("t5_trt_converter.repo_builder.fill_template_main")
# def test_build_repository_calls_fill_template(mock_fill: MagicMock, ...):
#    ...
#    RepoBuilder.build_repository(config)
#    assert mock_fill.call_count == expected_number_of_templates
#    # Check call args if necessary
