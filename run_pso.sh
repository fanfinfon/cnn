#!/bin/bash

# ==========================================
# 1. Check and Install Docker if missing
# ==========================================
if ! command -v docker > /dev/null 2>&1; then
    echo "Docker is not installed. Downloading and installing Docker now..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    echo "Docker installed successfully."
fi

# ==========================================
# 2. Check and Install NVIDIA Container Toolkit 
# (Required to use GPUs inside Docker)
# ==========================================
if ! command -v nvidia-container-cli > /dev/null 2>&1; then
    echo "NVIDIA Container Toolkit is not installed. Installing now..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed successfully."
fi

# ==========================================
# 3. Check Permissions
# ==========================================
DOCKER_CMD="docker"
# If standard docker command fails due to permissions, fallback to sudo
if ! docker ps >/dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
fi

# ==========================================
# 4. Start the Training
# ==========================================
# Stop and remove any existing container with the same name
$DOCKER_CMD rm -f pso_training >/dev/null 2>&1

echo "=========================================================="
echo " Starting PSO Training across 3 RTX A4000 GPUs"
echo "=========================================================="

# Run the docker container in the background
$DOCKER_CMD run --gpus all -d --rm \
  --name pso_training \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd):/root/project \
  -w /root/project \
  nvcr.io/nvidia/tensorflow:24.03-tf2-py3 \
  bash -c "pip install xgboost seaborn ray tqdm && python pso_patience100.py"

echo "Container successfully started in the background!"
echo "Tailing live logs now..."
echo "----------------------------------------------------------"
echo "(Note: You can safely press Ctrl+C at any time to exit the log viewer. The training will continue running in the background.)"
echo "----------------------------------------------------------"
echo ""

# Follow the container logs in real-time
$DOCKER_CMD logs -f pso_training
