Here's a clearer, more structured version of your original document, suitable for an LLM (or human developer) to confidently implement the requested Python module:

⸻

📘 Project Requirements: t5_trt_converter Python Module

This document outlines the requirements for a Python CLI tool named t5_trt_converter. This module is responsible for automating the conversion of Hugging Face T5 models into optimized TensorRT-LLM engines, and for building corresponding Triton Inference Server repository entries.

⸻

✅ Overview

The module should:
	•	Replicate the logic found in conv_t5.sh and build_triton_repo.sh
	•	Re-implement the tests from test_build_triton_repo.bats as pytest test cases
	•	Be packaged as a standalone Python module named: t5_trt_converter

⸻

🧩 Core Requirements

1. Module Structure
	•	All logic must be encapsulated in Python classes
	•	All methods should be implemented as @staticmethods (no instance or class state)
	•	Group functionality logically into separate classes (e.g., Converter, RepoBuilder, ConfigGenerator)

2. Strict Typing
	•	Use strict type annotations, especially for numerical operations and data transformation
	•	Prefer NumPy typing (numpy.typing.NDArray, etc.) for arrays
	•	Use dataclasses for all structured data passing between components
	•	Define custom TypedDicts or Pydantic models where needed for clarity

3. Functionality to Implement

The module must perform the following:

🔁 Conversion Pipeline
	•	Accept a Hugging Face T5 model directory
	•	Convert the model into TensorRT-LLM engine format using the steps in conv_t5.sh
	•	Support options such as:
	•	Precision mode (e.g. fp16, bf16)
	•	Weight-only quantization (e.g. int8)
	•	Sharding or multi-GPU configurations

🧱 Triton Repository Builder
	•	Build a valid Triton model repository structure, including:
	•	config.pbtxt files
	•	Engine file placement (e.g., plan or engine files)
	•	Tokenizer and model-related auxiliary files
	•	Match logic from build_triton_repo.sh

🧪 Testing
	•	Translate test_build_triton_repo.bats into equivalent pytest tests
	•	Use tmp_path or tempfile for temporary directories in tests
	•	Validate outputs: engine structure, config correctness, tokenizer presence

⸻

⚙️ CLI Design
	•	The module must expose a CLI entry point, e.g.:

t5_trt_converter convert --model-dir /path/to/model --output-dir /out --dtype bf16 --workers 4
t5_trt_converter build-repo --engine-dir /out --repo-dir /triton_repo

Use argparse or typer to build the CLI interface.

⸻

📁 File Organization Suggestion

t5_trt_converter/
├── __init__.py
├── converter.py        # Model conversion logic
├── repo_builder.py     # Triton repo construction logic
├── types.py            # Typed dataclasses and numpy types
├── cli.py              # CLI entry point
tests/
├── test_converter.py
├── test_repo_builder.py



⸻

🛠️ Additional Requirements
	•	Python 3.10+
	•	Follow PEP8 formatting, and use tools like black and mypy
	•	Document all classes and public methods with concise docstrings

⸻

🧪 Mocking External Tools

The module should support unit testing without depending on the execution of actual external tools like `convert_checkpoint.py` and `trtllm-build`, which may be slow or hardware-dependent.

4. External Command Mocking Strategy

To ensure testability and decoupling from system resources, use the following strategy:

4.1 Command Isolation
	•	All external command invocations (e.g., via `subprocess.run`) must be isolated within dedicated, mockable methods (like the implemented `CommandInvoker.run` or specific `_invoke_` methods).
	•	Avoid direct inline `subprocess.run(...)` calls in the main logic flow of `Converter` or `RepoBuilder`.

4.2 Unit Test Mocking
	•	Use `pytest` and Python's built-in `unittest.mock.patch` (or the `pytest-mock` fixture `mocker`) to mock the isolated command invocation methods during tests.
	•	Specifically, patch targets like `t5_trt_converter.converter.CommandInvoker.run` or the specific `_invoke_trtllm_build` / `_invoke_convert_checkpoint` methods.
	•	Verify:
		•	That the mocked invoker method is called.
		•	That it's called with the expected command list (`cmd`).
		•	That the rest of the code behaves correctly based on the mocked return value (e.g., simulating success or raising `FileNotFoundError` / `CalledProcessError`).

⸻

🧪 Example Test Case using `unittest.mock.patch`

```python
from unittest.mock import patch, MagicMock
import subprocess

# Inside a test function or class...
@patch('t5_trt_converter.converter.CommandInvoker.run')
def test_build_encoder_engine_invokes_correctly(mock_run: MagicMock, /* other fixtures */):
    # Arrange
    config = ... # Set up ConversionConfig
    paths = ... # Set up DerivedPaths
    expected_cmd_args = [ # Expected args passed to _invoke_trtllm_build
        "--checkpoint_dir", str(paths.trt_model_dir / "encoder"),
        # ... other expected args ...
    ]
    expected_full_cmd = ["trtllm-build"] + expected_cmd_args
    
    # Act
    Converter.build_encoder_engine(config, paths)

    # Assert
    mock_run.assert_called_once()
    # Check the 'cmd' argument passed to the mocked CommandInvoker.run
    actual_cmd_call = mock_run.call_args[0][0] # First positional arg of the call
    assert actual_cmd_call == expected_full_cmd

    # Example: Test error handling
    mock_run.side_effect = subprocess.CalledProcessError(1, expected_full_cmd)
    with pytest.raises(subprocess.CalledProcessError):
         Converter.build_encoder_engine(config, paths) # Should re-raise
```

⸻

✅ Result

This allows developers to:
	•	Write fast, isolated unit tests for the `Converter` logic.
	•	Avoid reliance on system state (installed binaries, PATH).
	•	Verify correct argument construction and error propagation.

⸻

