#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# --- Helper Functions ---
source utils.sh
usage() {
    echo "Usage: $0 <hf_model_dir> <model_name> [--engine-load-dir=PATH] [--tokenizer-dir=PATH]"
    echo "Example: $0 ./tmp/hf_models/t5-small t5-small"
    echo "Example with custom engine dir: $0 ./tmp/hf_models/t5-small t5-small --engine-load-dir=/path/to/custom/engines"
    echo "Example with custom tokenizer dir: $0 ./tmp/hf_models/t5-small t5-small --tokenizer-dir=/path/to/custom/tokenizer"
    echo ""
    echo "Required Arguments:"
    echo "  hf_model_dir: Path to the directory containing the Hugging Face model files (e.g., downloaded via git clone)."
    echo "  model_name:   A name for the model used in output paths (e.g., t5-small, flan-t5-large)."
    echo ""
    echo "Optional Arguments:"
    echo "  --engine-load-dir=PATH: Use a custom engine directory instead of the default."
    echo "  --tokenizer-dir=PATH:   Use a custom tokenizer directory instead of the HF model directory."
    echo ""
    echo "Optional Environment Variables (Defaults):"
    echo "  MODEL_TYPE (t5):             Model architecture type."
    echo "  INFERENCE_PRECISION (bfloat16): Target inference precision (float16, bfloat16, float32)."
    echo "  TP_SIZE (1):                 Tensor Parallelism size."
    echo "  PP_SIZE (1):                 Pipeline Parallelism size."
    echo "  MAX_BEAM_WIDTH (1):          Maximum beam width for search."
    echo "  MAX_ENCODER_INPUT_LEN (512): Max encoder input sequence length."
    echo "  MAX_SEQ_LEN (512):          Max decoder total sequence length (input + output)."
    echo "  MAX_BATCH_SIZE (128):        Max batch size for engine build."
    echo "  BASE_DIR (/tmp/trt_models):  Base directory for TensorRT models."
    echo "  ENGINES_BASE_DIR (/tmp/trt_engines2): Base directory for TensorRT engines."
    echo "  TRITON_REPO_NAME (enc_dec_ifb): Name for the Triton repository folder."
    exit 1
}

# Parse command line arguments
parse_args() {
    # Need at least two arguments
    if [ $# -lt 2 ]; then
        usage
    fi
    
    # Set the first two arguments
    HF_MODEL_DIR="$1"
    MODEL_NAME="$2"
    shift 2
    
    # Check if HF_MODEL_DIR exists
    if [ ! -d "$HF_MODEL_DIR" ]; then
        echo "Error: Hugging Face model directory not found: ${HF_MODEL_DIR}"
        usage
    fi
    
    # Check for additional arguments
    CUSTOM_ENGINE_DIR=""
    CUSTOM_TOKENIZER_DIR=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --engine-load-dir=*)
                CUSTOM_ENGINE_DIR="${1#*=}"
                ;;
            --tokenizer-dir=*)
                CUSTOM_TOKENIZER_DIR="${1#*=}"
                ;;
            --help|-h)
                usage
                ;;
            *)
                echo "Error: Unknown argument: $1"
                usage
                ;;
        esac
        shift
    done
}

# Get the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Parse command line arguments
parse_args "$@"

# --- Configuration Variables (Defaults can be overridden by environment variables) ---
: "${MODEL_TYPE:=t5}"                # Model type ("t5")
: "${INFERENCE_PRECISION:=bfloat16}" # Inference precision ("float16", "bfloat16", "float32")
: "${TP_SIZE:=1}"                    # Tensor Parallelism size
: "${PP_SIZE:=1}"                    # Pipeline Parallelism size
: "${MAX_BEAM_WIDTH:=1}"             # Maximum beam width for beam search
: "${MAX_ENCODER_INPUT_LEN:=512}"    # Maximum input length for the encoder
: "${MAX_SEQ_LEN:=512}"              # Maximum total sequence length (input + output) for the decoder
: "${MAX_BATCH_SIZE:=128}"           # Maximum batch size for inference engines
: "${TRITON_REPO_NAME:=enc_dec_ifb}" # Name for the Triton repository folder

# --- Derived Variables ---
GPU_ARCH=$(get_gpu_arch)
WORLD_SIZE=$((TP_SIZE * PP_SIZE))
MAX_NUM_TOKENS=$((MAX_SEQ_LEN * MAX_BATCH_SIZE))
# Include GPU Arch in the path
: "${BASE_DIR:=/tmp/trt_models}"           # Base directory for TensorRT models
: "${ENGINES_BASE_DIR:=/tmp/trt_engines2}" # Base directory for TensorRT engines
TRT_MODEL_DIR="${BASE_DIR}/${MODEL_NAME}/${GPU_ARCH}/${INFERENCE_PRECISION}/${WORLD_SIZE}-gpu"
TRT_ENGINE_DIR="${ENGINES_BASE_DIR}/${MODEL_NAME}/${GPU_ARCH}/${INFERENCE_PRECISION}/${WORLD_SIZE}-gpu"
TRITON_REPO_DIR="${BASE_DIR}/triton_repos/${MODEL_NAME}/${TRITON_REPO_NAME}"
DECODER_MAX_INPUT_LEN=1 # Default decoder start token length for enc-dec models

# Use custom engine directory if provided
if [ -n "${CUSTOM_ENGINE_DIR:-}" ]; then
    TRT_ENGINE_DIR="${CUSTOM_ENGINE_DIR}"
    echo "Using custom engine directory: ${TRT_ENGINE_DIR}"
fi

# Optimization profiles for batch size (min, opt, max)

# --- Helper Functions ---


create_dirs() {
    echo "Creating directories..."
    # Don't create HF_MODEL_DIR, it's required input
    mkdir -p "${TRT_MODEL_DIR}/encoder"
    mkdir -p "${TRT_MODEL_DIR}/decoder"
    mkdir -p "${TRT_ENGINE_DIR}/encoder"
    mkdir -p "${TRT_ENGINE_DIR}/decoder"
}

convert_checkpoint() {
    echo "Converting Hugging Face checkpoint from ${HF_MODEL_DIR} to TensorRT-LLM format..."
    # Check if config files exist in the *specific* target directory
    if [ ! -f "${TRT_MODEL_DIR}/encoder/config.json" ] || [ ! -f "${TRT_MODEL_DIR}/decoder/config.json" ]; then
        # Ensure the convert_checkpoint script path is correct relative to this script's location
        local convert_script_path="${SCRIPT_DIR}/convert_checkpoint.py"
        if [ ! -f "$convert_script_path" ]; then
            echo "Error: convert_checkpoint.py not found at expected path: $convert_script_path"
            echo "Please ensure the script is run from the correct directory or adjust the path."
            exit 1
        fi
        python3 "$convert_script_path" \
            --model_type "${MODEL_TYPE}" \
            --model_dir "${HF_MODEL_DIR}" \
            --output_dir "${TRT_MODEL_DIR}" \
            --tp_size "${TP_SIZE}" \
            --pp_size "${PP_SIZE}" \
            --dtype "${INFERENCE_PRECISION}"
    else
        echo "TensorRT-LLM checkpoint already exists in ${TRT_MODEL_DIR}. Skipping conversion."
    fi
}

build_encoder_engine() {
    local encoder_engine_file="${TRT_ENGINE_DIR}/encoder/rank0.engine"
    echo "Building TensorRT engine for Encoder..."
    if [ ! -f "${encoder_engine_file}" ]; then
        trtllm-build \
            --checkpoint_dir "${TRT_MODEL_DIR}/encoder" \
            --output_dir "${TRT_ENGINE_DIR}/encoder" \
            --paged_kv_cache disable \
            --moe_plugin disable \
            --max_beam_width "${MAX_BEAM_WIDTH}" \
            --max_batch_size "${MAX_BATCH_SIZE}" \
            --max_num_tokens "${MAX_NUM_TOKENS}" \
            --max_input_len "${MAX_ENCODER_INPUT_LEN}" \
            --gemm_plugin "${INFERENCE_PRECISION}" \
            --bert_attention_plugin "${INFERENCE_PRECISION}" \
            --gpt_attention_plugin "${INFERENCE_PRECISION}" \
            --remove_input_padding enable \
            --multiple_profiles enable \
            --context_fmha disable # T5 relative attention bias not compatible with FMHA
    else
        echo "Encoder engine already exists. Skipping build."
    fi
}

build_decoder_engine() {
    local decoder_engine_file="${TRT_ENGINE_DIR}/decoder/rank0.engine"
    echo "Building TensorRT engine for Decoder..."
    if [ ! -f "${decoder_engine_file}" ]; then
        trtllm-build \
            --checkpoint_dir "${TRT_MODEL_DIR}/decoder" \
            --output_dir "${TRT_ENGINE_DIR}/decoder" \
            --moe_plugin disable \
            --max_beam_width "${MAX_BEAM_WIDTH}" \
            --max_batch_size "${MAX_BATCH_SIZE}" \
            --max_input_len "${DECODER_MAX_INPUT_LEN}" \
            --max_seq_len "${MAX_SEQ_LEN}" \
            --max_encoder_input_len "${MAX_ENCODER_INPUT_LEN}" \
            --gemm_plugin "${INFERENCE_PRECISION}" \
            --bert_attention_plugin "${INFERENCE_PRECISION}" \
            --gpt_attention_plugin "${INFERENCE_PRECISION}" \
            --remove_input_padding enable \
            --multiple_profiles enable \
            --context_fmha disable # T5 relative attention bias not compatible with FMHA
        # --use_implicit_relative_attention # Add if max_seq_len is very large causing OOM
    else
        echo "Decoder engine already exists. Skipping build."
    fi
}

# --- Main Script ---

echo "Starting T5 Conversion and Build Process"
echo "----------------------------------------"
echo "HF Model Dir:        ${HF_MODEL_DIR}"
echo "Model Name:          ${MODEL_NAME}"
echo "Model Type:          ${MODEL_TYPE}"
echo "GPU Architecture:    ${GPU_ARCH}"
echo "Inference Precision: ${INFERENCE_PRECISION}"
echo "Tensor Parallelism:  ${TP_SIZE}"
echo "Pipeline Parallelism:${PP_SIZE}"
echo "World Size:          ${WORLD_SIZE}"
echo "Max Beam Width:      ${MAX_BEAM_WIDTH}"
echo "Max Encoder Input:   ${MAX_ENCODER_INPUT_LEN}"
echo "Max Decoder Seq Len: ${MAX_SEQ_LEN}"
echo "Max Batch Size:      ${MAX_BATCH_SIZE}"
echo "TRT Model Dir:       ${TRT_MODEL_DIR}"
echo "TRT Engine Dir:      ${TRT_ENGINE_DIR}"
echo "----------------------------------------"

# 1. Ensure output directories exist
create_dirs

# 2. Convert checkpoint (Input HF Model Dir already validated)
convert_checkpoint

# 4. Build Encoder Engine
build_encoder_engine

# 5. Build Decoder Engine
build_decoder_engine

echo "----------------------------------------"
echo "T5 Conversion and Build Process Completed Successfully!"
echo "Engines are located in: ${TRT_ENGINE_DIR}"
echo "----------------------------------------"

echo ""
echo "To run inference with the generated engines, use the following command from the project root:"
echo "python3 run.py --model_name ${MODEL_NAME} --engine_dir ${TRT_ENGINE_DIR} --hf_model_dir ${HF_MODEL_DIR}"
echo ""
echo "To build a Triton inference server repository with these engines, use the separate script:"
echo "source $(basename "$0") ${HF_MODEL_DIR} ${MODEL_NAME}"
# Construct build_triton_repo.sh command with appropriate parameters
build_cmd="./build_triton_repo.sh"
if [ -n "${CUSTOM_ENGINE_DIR:-}" ]; then
    build_cmd+=" --engine-load-dir=${CUSTOM_ENGINE_DIR}"
fi
if [ -n "${CUSTOM_TOKENIZER_DIR:-}" ]; then
    build_cmd+=" --tokenizer-dir=${CUSTOM_TOKENIZER_DIR}"
fi
echo "$build_cmd"
echo "This will create a Triton repository at: ${TRITON_REPO_DIR}"
