$ErrorActionPreference = "Stop"

Write-Host "Building recbole-env image (including datasets, excluding saved and log)..." -ForegroundColor Cyan
docker build -t recbole-env .

# Remove previous container instance if it exists so state starts fresh for new run
docker rm -f recbole-exp 2>$null

Write-Host "Starting recbole-env container (results will PERSIST in container after exit)..." -ForegroundColor Green
docker run -it --name recbole-exp --gpus all recbole-env
