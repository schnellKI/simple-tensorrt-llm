# T5 TensorRT-LLM Converter & Triton Repo Builder

This project provides tools to convert Hugging Face T5-based models into the TensorRT-LLM format and build the necessary Triton Inference Server repository structure.

## Recent Refactoring (Summary)

The `t5_trt_converter` package underwent significant refactoring with the following key goals and changes:

1.  **Elimination of Subprocess Calls:**
    *   Replaced `subprocess` calls to external Python scripts (`convert_checkpoint.py`, `fill_template.py`) with direct function imports within the package (`t5_trt_converter.converter`, `t5_trt_converter.repo_builder`).
    *   This improves security (mitigating risks like `S603`) and makes the conversion process more robust and easier to debug.
    *   **Note:** Calls to the external `trtllm-build` command remain in `converter.py` for engine building.

2.  **Dependency Removal (`tensorrt_llm` library):**
    *   The direct dependency on the `tensorrt_llm` Python library has been removed from the core conversion logic (`convert_checkpoint.py`, `helper.py`).
    *   Necessary functionalities (like dtype conversion, config handling, and mapping) were replaced with standard PyTorch operations or minimal placeholder classes (`MinimalConfig`, `MinimalMapping`) defined within `convert_checkpoint.py`.
    *   This makes the conversion package potentially usable in environments where the full `tensorrt_llm` library is not installed or easily accessible.

3.  **Code Integration & Structure:**
    *   The `helper.py` script was moved from its original location (`trt/t5/helper.py`) into the `t5_trt_converter` package directory to reside alongside the modules that use it.
    *   Relative imports were adjusted accordingly.

4.  **Testing:**
    *   Import errors arising during the refactoring were fixed.
    *   All tests within the `tests/` directory now pass, confirming the core functionality after the changes.

## Current Status

*   The primary conversion path for T5 models and the Triton repository building process function correctly using direct imports and without the `tensorrt_llm` library dependency (except for the `trtllm-build` calls).
*   The `convert_checkpoint.py` script still contains conversion logic and configuration parsing for other model types (NMT, BART, Pix2Struct, BLIP2) inherited from its original source. This code has numerous linting and typing errors and may require significant further refactoring or removal if only T5 support is desired long-term.
*   Similarly, `helper.py` contains functions (`fairseq_sin_pos_embedding`, model-specific QKV names) related to non-T5 models that may need review.

## Usage

(TODO: Add details on how to run the conversion, likely via `t5_trt_converter/cli.py`)

## Testing

To run the test suite:

```bash
pytest
```

# Simple TensorTrT-LLM 

TRT makes using Nvidia's TensorRT LLM easier, because it's f*(_ing hard. 

Using this project you should be able to
1. Point to an HF format model (Generative or not, including encoder-decoder)
2. Compile it with TensorRT 
2.1. Sweep over TensorRT params like MacBatch size and tokens
3. Create a Triton server model repositoy for the built model.
4. Benchmark the deployed model 


## Nvidia's Tools 

Generally Nvidia provides many tools and scripts for these, but they are hard to compose together, or even discover. 

- [Triton Inference Server][triton-inference-server]
- [Triton Server GitHub repo][triton-server]
- [Triton TensorRT-LLM backend][tensorrtllm-backend]
- [TensorRT-LLM GitHub repo][tensorrt-llm]
- [TensorRT-LLM enc_dec examples][tensorrt-llm-enc-dec]
- [Encoder-Decoder Backend Docs][encoder-decoder-docs]
- [GenAI Perf Analyzer README][genai-perf-readme]

## References

[triton-inference-server]: https://developer.nvidia.com/triton-inference-server  
[triton-server]: https://github.com/triton-inference-server/server  
[tensorrtllm-backend]: https://github.com/triton-inference-server/tensorrtllm_backend  
[tensorrt-llm]: https://github.com/NVIDIA/TensorRT-LLM  
[tensorrt-llm-enc-dec]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec  
[encoder-decoder-docs]: https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/encoder_decoder.md  
[genai-perf-readme]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html  

# Steps and current work
So far I've
* Made a script that does the conversion from HF to Tensorrt for Flan
* Loaded that manually into Triton 
## I learned

* The [tensortrt-llm-backend exmaple](tensorrt-llm-enc-dec) is misleading because it makes a model repo with a python entry but it doesn't use it. e.g. it does for the preprocessor and postprocesser,,but the tensortrtllm direcotry has a model.py that one can delete 

* The triton dynamic batcher isn't relevant because the tensorrtllm backend is doing inflight batching . 
* It seems to be version sensitive 
