.PHONY: install-dev venv

VENV_DIR := .venv

venv: $(VENV_DIR)/pyvenv.cfg

$(VENV_DIR)/pyvenv.cfg:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	uv venv $(VENV_DIR)
	@# We touch the pyvenv.cfg file only after uv venv succeeds

install-dev: venv
	@echo "Installing development dependencies into $(VENV_DIR)..."
	uv pip install -e .[dev]
	@echo "Installation complete."

# === NVIDIA Dependency Installation ===

# Install NVIDIA dependencies WITHOUT GPU support
# This uses uv pip install --no-deps to skip GPU-specific packages.
.PHONY: install-nvidia-deps-no-gpu
install-nvidia-deps-no-gpu:
	@echo "Installing NVIDIA dependencies (no GPU)..."
	uv pip install --no-deps --prerelease=allow

# Install NVIDIA dependencies WITH GPU support
# This installs necessary system packages (libopenmpi-dev) and tensorrt_llm from NVIDIA's index.
# Requires sudo privileges for apt-get.
.PHONY: install-nvidia-deps-gpu
install-nvidia-deps-gpu:
	@echo "Installing NVIDIA dependencies (GPU)..."
	sudo apt-get -y install libopenmpi-dev
	uv pip install tensorrt_llm

# Check for GPU and install appropriate NVIDIA dependencies
# Runs nvidia-smi to detect a GPU. If found, runs install-nvidia-deps-gpu. Otherwise, runs install-nvidia-deps-no-gpu.
.PHONY: install-nvidia-deps
install-nvidia-deps:
	@if nvidia-smi > /dev/null 2>&1; then \\
	    echo "NVIDIA GPU detected. Installing GPU dependencies..."; \\
	    $(MAKE) install-nvidia-deps-gpu; \\
	else \\
	    echo "No NVIDIA GPU detected or nvidia-smi not found. Installing non-GPU dependencies..."; \\
	    $(MAKE) install-nvidia-deps-no-gpu; \\
	fi 