get_gpu_arch() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: nvidia-smi command not found. Cannot determine GPU architecture."
        exit 1
    fi
    # Get the name of the first GPU, replace spaces with underscores
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1 | sed 's/ /_/g')
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
