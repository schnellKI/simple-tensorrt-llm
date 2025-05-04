#!/bin/bash

# build_triton_repo.sh
# This script creates a Triton Inference Server repository for T5/Flan-T5 models
# converted with conv_t5.sh

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# Usage information for this script
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "This script requires environment variables typically set by conv_t5.sh:"
    echo ""
    echo "Options:"
    echo "  --engine-load-dir=PATH  Override the TRT_ENGINE_DIR with a custom engine directory"
    echo ""
    echo "Required Environment Variables:"
    echo "  SCRIPT_DIR:          Directory containing repo_template and tools"
    echo "  TRITON_REPO_DIR:     Output directory for the Triton repository"
    echo "  HF_MODEL_DIR:        Path to the HuggingFace model"
    echo "  TRT_ENGINE_DIR:      Path to the TensorRT engines (can be overridden with --engine-load-dir)"
    echo "  MAX_BATCH_SIZE:      Maximum batch size for inference"
    echo "  MAX_BEAM_WIDTH:      Maximum beam width for beam search"
    echo ""
    echo "Example usage:"
    echo "  1. First run conv_t5.sh to set up the environment:"
    echo "     ./conv_t5.sh <hf_model_dir> <model_name>"
    echo "  2. Then run this script:"
    echo "     ./build_triton_repo.sh"
    echo "     or with custom engine directory:"
    echo "     ./build_triton_repo.sh --engine-load-dir=/path/to/engines"
    echo ""
    echo "Or source conv_t5.sh and run this script:"
    echo "  source ./conv_t5.sh <hf_model_dir> <model_name>"
    echo "  ./build_triton_repo.sh"
    exit 1
}

# Parse command line arguments
parse_args() {
    local custom_engine_dir=""
    
    # Parse command line arguments
    for arg in "$@"; do
        case $arg in
            --engine-load-dir=*)
                custom_engine_dir="${arg#*=}"
                # Override TRT_ENGINE_DIR if custom engine directory is provided
                if [ -n "$custom_engine_dir" ]; then
                    echo "Using custom engine directory: $custom_engine_dir"
                    export TRT_ENGINE_DIR="$custom_engine_dir"
                fi
                ;;
            --help|-h)
                usage
                ;;
            *)
                # Ignore other arguments for backward compatibility
                ;;
        esac
    done
}

# Verify that required environment variables are set
check_env_vars() {
    local missing_vars=0
    for var in SCRIPT_DIR TRITON_REPO_DIR HF_MODEL_DIR TRT_ENGINE_DIR MAX_BATCH_SIZE MAX_BEAM_WIDTH; do
        if [ -z "${!var:-}" ]; then
            echo "Error: Required environment variable $var is not set"
            missing_vars=1
        fi
    done
    
    if [ $missing_vars -eq 1 ]; then
        echo ""
        echo "Please ensure you've run or sourced conv_t5.sh before running this script."
        usage
    fi
}

# Main function to build the Triton repository
build_triton_repo() {
    local template_path="${SCRIPT_DIR}/repo_template"
    local fill_template_script="${SCRIPT_DIR}/fill_template.py"

    echo "Building Triton repository for the model..."
    echo "Triton repository will be created at: ${TRITON_REPO_DIR}"
    echo "Using engine directory: ${TRT_ENGINE_DIR}"

    if [ ! -f "${fill_template_script}" ]; then
        echo "Error: fill_template.py script not found at ${fill_template_script}"
        echo "Please ensure the tools directory exists in the script location."
        return 1
    fi

    # Create the directory
    mkdir -p "$(dirname "${TRITON_REPO_DIR}")"

    # Copy template to destination
    echo "Copying template from ${template_path}/ to ${TRITON_REPO_DIR}"
    cp -r "${template_path}/" "${TRITON_REPO_DIR}"

    # Generate configurations with template fill
    echo "Generating Triton server configurations..."

    # Set environment variables for template filling
    local ENGINE_PATH="${TRT_ENGINE_DIR}"
    local HF_MODEL_PATH="${HF_MODEL_DIR}"

    # Fill templates for different components
    python3 "${fill_template_script}" -i "${TRITON_REPO_DIR}/tensorrt_llm/config.pbtxt" \
        "triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:${MAX_BEAM_WIDTH},engine_dir:${ENGINE_PATH}/decoder,encoder_engine_dir:${ENGINE_PATH}/encoder,kv_cache_free_gpu_mem_fraction:0.8,cross_kv_cache_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,max_queue_size:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"

    python3 "${fill_template_script}" -i "${TRITON_REPO_DIR}/preprocessing/config.pbtxt" \
        "tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:1"

    python3 "${fill_template_script}" -i "${TRITON_REPO_DIR}/postprocessing/config.pbtxt" \
        "tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1"

    python3 "${fill_template_script}" -i "${TRITON_REPO_DIR}/ensemble/config.pbtxt" \
        "triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:TYPE_FP32"

    python3 "${fill_template_script}" -i "${TRITON_REPO_DIR}/tensorrt_llm_bls/config.pbtxt" \
        "triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32"

    echo "Triton repository setup complete in ${TRITON_REPO_DIR} directory"
}

delete_unused_files() {
    #deletes the bls model and the model.py that is in the the tensorrt_llm directory

    echo "Deleting bls path ${TRITON_REPO_DIR}/tensorrt_llm_bls"
    rm -rf "${TRITON_REPO_DIR}/tensorrt_llm_bls"
    echo "Deleting unused model.py  ${TRITON_REPO_DIR}/tensorrt_llm/model.py"
    rm -rf "${TRITON_REPO_DIR}/tensorrt_llm/model.py"
}

# Parse command line arguments
parse_args "$@"

# Check for required environment variables before proceeding
check_env_vars

# Run the main function
build_triton_repo 
delete_unused_files