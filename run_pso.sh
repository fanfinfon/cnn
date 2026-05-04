#!/bin/bash

# Stop and remove any existing container with the same name to avoid conflicts
docker rm -f pso_training >/dev/null 2>&1

echo "=========================================================="
echo " Starting PSO Training across 3 RTX A4000 GPUs"
echo "=========================================================="

# Run the docker container in the background
docker run --gpus all -d --rm \
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
docker logs -f pso_training
