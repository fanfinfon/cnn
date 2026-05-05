#!/bin/bash

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as an error, and fail on pipeline errors.
set -euo pipefail

# ==========================================
# 0. Helper Functions
# ==========================================
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

# ==========================================
# 1. Check and Install Docker if missing
# ==========================================
if ! command -v docker > /dev/null 2>&1; then
    log "Docker is not installed. Downloading and installing Docker now..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm -f get-docker.sh
    log "Docker installed successfully."
fi

# ==========================================
# 2. Check and Install NVIDIA Container Toolkit 
# (Required to use GPUs inside Docker)
# ==========================================
if ! command -v nvidia-container-cli > /dev/null 2>&1; then
    log "NVIDIA Container Toolkit is not installed. Installing now..."
    
    # Check if we're on an apt-based system before attempting to run apt-get
    if ! command -v apt-get > /dev/null 2>&1; then
        error "This script currently only supports automatic installation on apt-based systems (Debian/Ubuntu). Please install the NVIDIA Container Toolkit manually."
    fi

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    log "NVIDIA Container Toolkit installed successfully."
fi

# ==========================================
# 3. Check Permissions and Daemon State
# ==========================================
DOCKER_CMD="docker"

# First, check if docker daemon is reachable without sudo
if ! docker ps >/dev/null 2>&1; then
    # If not, try with sudo
    if sudo docker ps >/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    else
        error "Docker daemon is not running, or the current user does not have permissions to access it even with sudo."
    fi
fi

# ==========================================
# 4. Verify Host GPU Health & Docker GPU Access
# ==========================================
log "Verifying host GPU health..."

# 1. Check if NVIDIA driver is installed (nvidia-smi exists)
if ! command -v nvidia-smi > /dev/null 2>&1; then
    error "nvidia-smi not found. The NVIDIA driver is not installed on the host OS."
fi

# 2. Check if GPU is visible and driver is healthy at host level
if ! nvidia-smi > /dev/null 2>&1; then
    log "Unprivileged nvidia-smi failed. Attempting with sudo..."
    if ! sudo nvidia-smi > /dev/null 2>&1; then
        error "nvidia-smi execution failed even with sudo. The GPU is not visible to the system, or the host NVIDIA driver/CUDA stack is broken."
    fi
fi

# 3. Test if Docker is correctly configured to pass GPUs to containers
log "Testing Docker GPU access..."
if ! $DOCKER_CMD run --rm --gpus all nvcr.io/nvidia/tensorflow:24.03-tf2-py3 nvidia-smi > /dev/null 2>&1; then
    error "Docker cannot access the GPUs. The NVIDIA Container Toolkit might be misconfigured, or the Docker daemon needs a restart (sudo systemctl restart docker)."
fi

log "GPU health checks passed successfully!"

# ==========================================
# 5. Start the Training
# ==========================================
# Stop and remove any existing container with the same name
# Use || true so the script doesn't fail if the container doesn't exist
$DOCKER_CMD rm -f pso_training >/dev/null 2>&1 || true

echo "=========================================================="
echo " Starting PSO Training across 3 RTX A4000 GPUs"
echo "=========================================================="

# Run the docker container in the background.
# Note: --rm is deliberately omitted so that if the container crashes,
# you can still inspect its logs for debugging.
$DOCKER_CMD run --gpus all -d \
  --name pso_training \
  -e PYTHONUNBUFFERED=1 \
  -v "$(pwd):/root/project" \
  -w /root/project \
  nvcr.io/nvidia/tensorflow:24.03-tf2-py3 \
  bash -c "pip install --no-cache-dir xgboost seaborn ray tqdm && python pso_patience100.py"

# Briefly pause to allow Docker to initialize the container and verify it didn't crash immediately
sleep 2

# Verify that the container is actually running
if ! $DOCKER_CMD ps --format '{{.Names}}' | grep -q "^pso_training$"; then
    echo "=========================================================="
    error "Container 'pso_training' failed to start or crashed immediately. Showing recent logs:"
    $DOCKER_CMD logs pso_training || true
    exit 1
fi

echo "Container successfully started in the background!"
echo "Tailing live logs now..."
echo "----------------------------------------------------------"
echo "(Note: You can safely press Ctrl+C at any time to exit the log viewer. The training will continue running in the background.)"
echo "----------------------------------------------------------"
echo ""

# Follow the container logs in real-time
$DOCKER_CMD logs -f pso_training
