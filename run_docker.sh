#!/bin/bash
set -e

echo "Building recbole-env image (including datasets, excluding saved and log)..."
docker build -t recbole-env .

echo "Removing previous container instance if exists..."
docker rm -f recbole-exp 2>/dev/null || true

echo "Starting recbole-env container (results will PERSIST in container after exit)..."
docker run -it --name recbole-exp --gpus all recbole-env
