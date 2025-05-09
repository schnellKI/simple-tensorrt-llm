TEST_BREW_PREFIX="$(brew --prefix)"
load "${TEST_BREW_PREFIX}/lib/bats-support/load.bash"
load "${TEST_BREW_PREFIX}/lib/bats-assert/load.bash"
load "${TEST_BREW_PREFIX}/lib/bats-file/load.bash"
# Test the conversion of T5 models to TensorRT
function setup() {
    export BATS_TMPDIR="$(temp_make)"
    export HF_MODEL_DIR="${BATS_TMPDIR}/hf_models"
    export SCRIPT_DIR="$(pwd)"
    export TRITON_REPO_DIR="${BATS_TMPDIR}/triton_repo"
    export TRT_ENGINE_DIR="${BATS_TMPDIR}/trt_engines"
    export MAX_BATCH_SIZE=128
    export MAX_BEAM_WIDTH=4
    
}
@test "test_script_and_deps_is_on_path" {
    assert [ -e build_triton_repo.sh ]
    assert [ -d repo_template ]
    assert [ -e fill_template.py ]
}



@test "test_build_triton_repo" {
    run ./build_triton_repo.sh ${HF_MODEL_DIR} t5-small
    assert_success
    assert_output --partial "Deleting bls path ${TRITON_REPO_DIR}/tensorrt_llm_bls"
    #Assert the the bls dir is deleted
    assert_output --partial "Deleting unused model.py  ${TRITON_REPO_DIR}/tensorrt_llm/model.py"
    # Noe we assert that the preprocessor,postprocessor and tensorrt_llm,ensembles directories exist 
    
    # Now assert that those are the only directories in the triton_repo directory
    assert [ $(ls -1 ${TRITON_REPO_DIR} | wc -l) -eq 4 ]
}
@test "repo_structure_is_correct" {
    run ./build_triton_repo.sh ${HF_MODEL_DIR} t5-small
    assert_success

    assert [ ! -d "${TRITON_REPO_DIR}/tensorrt_llm_bls" ]

    assert_output --partial "Deleting bls path ${TRITON_REPO_DIR}/tensorrt_llm_bls"
    assert_output --partial "Deleting unused model.py  ${TRITON_REPO_DIR}/tensorrt_llm/model.py"

    # The resulting repo should have the following directories
    assert [ -d "${TRITON_REPO_DIR}/preprocessing" ]
    assert [ -d "${TRITON_REPO_DIR}/postprocessing" ]
    assert [ -d "${TRITON_REPO_DIR}/tensorrt_llm" ]
    assert [ -d "${TRITON_REPO_DIR}/ensemble" ]
    
}

@test "tensorrt_llm_default_engine_dirs" {
    # The default fill_template assumes it run inside the triton container, which is maybe correct. 
    # But if we ran it on the host, we need to allow the user to set where the engine files are stored.
    # The template has two sections that look like this, and we need to check that the values are set correctly.
#         parameters: {
#       key: "gpt_model_path"
#       value: {
#         string_value: "${engine_dir}"
#       }
#     }
#     parameters: {
#       key: "encoder_model_path"
#       value: {
#         string_value: "${encoder_engine_dir}"
#       }
#     }   
    EXPECTED_DECODER_DIR="${TRT_ENGINE_DIR}/decoder"
    EXPECTED_ENCODER_DIR="${TRT_ENGINE_DIR}/encoder"

    run ./build_triton_repo.sh ${HF_MODEL_DIR} t5-small
    assert_success
    # Check that the gpt_model_path and encoder_model_path are set correctly
    run grep -A2      "gpt_model_path" ${TRITON_REPO_DIR}/tensorrt_llm/config.pbtxt 
    assert_success

    assert_output --partial "string_value: \"${EXPECTED_DECODER_DIR}\""

    run grep -A 2  "encoder_model_path" ${TRITON_REPO_DIR}/tensorrt_llm/config.pbtxt 
    assert_success
    assert_output --partial "string_value: \"${EXPECTED_ENCODER_DIR}\""
}

@test "tensorrt_llm_custom_engine_dirs" {
    # Create a custom engine directory path
    export CUSTOM_ENGINE_DIR="${BATS_TMPDIR}/custom_engines"
    EXPECTED_DECODER_DIR="${CUSTOM_ENGINE_DIR}/decoder"
    EXPECTED_ENCODER_DIR="${CUSTOM_ENGINE_DIR}/encoder"
    
    # Run the build_triton_repo script with the custom engine directory
    run ./build_triton_repo.sh --engine-load-dir="${CUSTOM_ENGINE_DIR}" ${HF_MODEL_DIR} t5-small
    assert_success
    
    # Check that the gpt_model_path is set to the custom decoder directory
    run grep -A2 "gpt_model_path" ${TRITON_REPO_DIR}/tensorrt_llm/config.pbtxt 
    assert_success
    assert_output --partial "string_value: \"${EXPECTED_DECODER_DIR}\""
    
    # Check that the encoder_model_path is set to the custom encoder directory
    run grep -A2 "encoder_model_path" ${TRITON_REPO_DIR}/tensorrt_llm/config.pbtxt 
    assert_success
    assert_output --partial "string_value: \"${EXPECTED_ENCODER_DIR}\""
}

@test "default_tokenizer_dir_in_preprocessing" {
    # Check that the tokenizer_dir parameter in preprocessing/config.pbtxt points to the default HF model directory
    run ./build_triton_repo.sh ${HF_MODEL_DIR} t5-small
    assert_success
    
    # Check the tokenizer_dir parameter in preprocessing/config.pbtxt
    run grep -A2 "tokenizer_dir" ${TRITON_REPO_DIR}/preprocessing/config.pbtxt
    assert_success
    assert_output --partial "string_value: \"${HF_MODEL_DIR}\""
}

@test "default_tokenizer_dir_in_postprocessing" {
    # Check that the tokenizer_dir parameter in postprocessing/config.pbtxt points to the default HF model directory
    run ./build_triton_repo.sh ${HF_MODEL_DIR} t5-small
    assert_success
    
    # Check the tokenizer_dir parameter in postprocessing/config.pbtxt
    run grep -A2 "tokenizer_dir" ${TRITON_REPO_DIR}/postprocessing/config.pbtxt
    assert_success
    assert_output --partial "string_value: \"${HF_MODEL_DIR}\""
}

@test "custom_tokenizer_dir_in_preprocessing" {
    # Test with a custom tokenizer directory
    export CUSTOM_TOKENIZER_DIR="${BATS_TMPDIR}/custom_tokenizer"
    
    # Run the build_triton_repo script with a custom tokenizer directory
    run ./build_triton_repo.sh --tokenizer-dir="${CUSTOM_TOKENIZER_DIR}" ${HF_MODEL_DIR} t5-small
    assert_success
    
    # Check the tokenizer_dir parameter in preprocessing/config.pbtxt
    run grep -A2 "tokenizer_dir" ${TRITON_REPO_DIR}/preprocessing/config.pbtxt
    assert_success
    assert_output --partial "string_value: \"${CUSTOM_TOKENIZER_DIR}\""
}

@test "custom_tokenizer_dir_in_postprocessing" {
    # Test with a custom tokenizer directory
    export CUSTOM_TOKENIZER_DIR="${BATS_TMPDIR}/custom_tokenizer"
    
    # Run the build_triton_repo script with a custom tokenizer directory
    run ./build_triton_repo.sh --tokenizer-dir="${CUSTOM_TOKENIZER_DIR}" ${HF_MODEL_DIR} t5-small
    assert_success
    
    # Check the tokenizer_dir parameter in postprocessing/config.pbtxt
    run grep -A2 "tokenizer_dir" ${TRITON_REPO_DIR}/postprocessing/config.pbtxt
    assert_success
    assert_output --partial "string_value: \"${CUSTOM_TOKENIZER_DIR}\""
}