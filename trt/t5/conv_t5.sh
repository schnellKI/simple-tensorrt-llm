#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Helper Functions ---

usage() {
    echo "Usage: $0 <hf_model_dir> <model_name>"
    echo "Example: $0 ./tmp/hf_models/t5-small t5-small"
    echo ""
    echo "Required Arguments:"
    echo "  hf_model_dir: Path to the directory containing the Hugging Face model files (e.g., downloaded via git clone)."
    echo "  model_name:   A name for the model used in output paths (e.g., t5-small, flan-t5-large)."
    echo ""
    echo "Optional Environment Variables (Defaults):"
    echo "  MODEL_TYPE (t5):             Model architecture type."
    echo "  INFERENCE_PRECISION (bfloat16): Target inference precision (float16, bfloat16, float32)."
    echo "  TP_SIZE (1):                 Tensor Parallelism size."
    echo "  PP_SIZE (1):                 Pipeline Parallelism size."
    echo "  MAX_BEAM_WIDTH (1):          Maximum beam width for search."
    echo "  MAX_ENCODER_INPUT_LEN (1024): Max encoder input sequence length."
    echo "  MAX_SEQ_LEN (1024):          Max decoder total sequence length (input + output)."
    echo "  MAX_BATCH_SIZE (128):        Max batch size for engine build."
    exit 1
}

get_gpu_arch() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi command not found. Cannot determine GPU architecture."
        exit 1
    fi
    # Get the name of the first GPU, replace spaces with underscores
    local gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1 | sed 's/ /_/g')
    if [ -z "$gpu_name" ]; then
        echo "Error: Could not determine GPU name using nvidia-smi."
        exit 1
    fi
    # Basic mapping (can be extended)
    if [[ $gpu_name == *"A100"* ]] || [[ $gpu_name == *"A"[0-9]* ]] || [[ $gpu_name == *"RTX_40"* ]] || [[ $gpu_name == *"RTX_A"* ]]; then # Includes A40, A6000, RTX 40 series etc.
        echo "ampere"
    elif [[ $gpu_name == *"H100"* ]] || [[ $gpu_name == *"H"[0-9]* ]]; then # Includes H800 etc.
        echo "hopper"
    elif [[ $gpu_name == *"V100"* ]]; then
        echo "volta"
    elif [[ $gpu_name == *"T4"* ]] || [[ $gpu_name == *"RTX_20"* ]] || [[ $gpu_name == *"RTX_30"* ]] || [[ $gpu_name == *"Quadro_RTX"* ]] || [[ $gpu_name == *"TITAN_RTX"* ]]; then # Includes most Turing GPUs
        echo "turing"
    elif [[ $gpu_name == *"P100"* ]] || [[ $gpu_name == *"GP100"* ]] || [[ $gpu_name == *"GTX_10"* ]] || [[ $gpu_name == *"Quadro_P"* ]] || [[ $gpu_name == *"TITAN_Xp"* ]] || [[ $gpu_name == *"TITAN_X_(Pascal)"* ]]; then # Includes Pascal GPUs
        echo "pascal"
    elif [[ $gpu_name == *"GTX_9"* ]] || [[ $gpu_name == *"GTX_TITAN_X"* ]] || [[ $gpu_name == *"Quadro_M"* ]]; then # Includes Maxwell GPUs
        echo "maxwell"
    elif [[ $gpu_name == *"K80"* ]] || [[ $gpu_name == *"K40"* ]] || [[ $gpu_name == *"GTX_7"* ]] || [[ $gpu_name == *"GTX_TITAN"* ]] || [[ $gpu_name == *"Quadro_K"* ]]; then # Includes Kepler GPUs
        echo "kepler"
    else
        # Fallback to lowercase gpu name if unknown
        echo "unknown_$(echo "$gpu_name" | tr '[:upper:]' '[:lower:]')"
    fi
}


# --- Argument Parsing ---
if [ "$#" -ne 2 ]; then
    usage
fi

HF_MODEL_DIR="$1"
MODEL_NAME="$2"

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "Error: Hugging Face model directory not found: ${HF_MODEL_DIR}"
    usage
fi

# --- Configuration Variables (Defaults can be overridden by environment variables) ---
: ${MODEL_TYPE:="t5"}       # Model type ("t5")
: ${INFERENCE_PRECISION:="bfloat16"} # Inference precision ("float16", "bfloat16", "float32")
: ${TP_SIZE:=1}             # Tensor Parallelism size
: ${PP_SIZE:=1}             # Pipeline Parallelism size
: ${MAX_BEAM_WIDTH:=1}      # Maximum beam width for beam search
: ${MAX_ENCODER_INPUT_LEN:=512} # Maximum input length for the encoder
: ${MAX_SEQ_LEN:=512}      # Maximum total sequence length (input + output) for the decoder
: ${MAX_BATCH_SIZE:=128}    # Maximum batch size for inference engines

# --- Derived Variables ---
GPU_ARCH=$(get_gpu_arch)
WORLD_SIZE=$((TP_SIZE * PP_SIZE))
MAX_NUM_TOKENS=$((MAX_SEQ_LEN * MAX_BATCH_SIZE))
# Include GPU Arch in the path
TRT_MODEL_DIR="tmp/trt_models/${MODEL_NAME}/${GPU_ARCH}/${INFERENCE_PRECISION}/${WORLD_SIZE}-gpu"
TRT_ENGINE_DIR="tmp/trt_engines2/${MODEL_NAME}/${GPU_ARCH}/${INFERENCE_PRECISION}/${WORLD_SIZE}-gpu"
DECODER_MAX_INPUT_LEN=1 # Default decoder start token length for enc-dec models

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
            --tp_size ${TP_SIZE} \
            --pp_size ${PP_SIZE} \
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
            --max_beam_width ${MAX_BEAM_WIDTH} \
            --max_batch_size ${MAX_BATCH_SIZE} \
            --max_num_tokens ${MAX_NUM_TOKENS} \
            --max_input_len ${MAX_ENCODER_INPUT_LEN} \
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
            --max_beam_width ${MAX_BEAM_WIDTH} \
            --max_batch_size ${MAX_BATCH_SIZE} \
            --max_input_len ${DECODER_MAX_INPUT_LEN} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --max_encoder_input_len ${MAX_ENCODER_INPUT_LEN} \
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
