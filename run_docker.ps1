$ErrorActionPreference = "Stop"

Write-Host "Building rcat *Xsadaecbole-env image (including datasets & all saved models, excluding log)..." -ForegroundColor Cyan
docker build -t recbole-env .

# Remove previous container instance if it exists so state starts fresh for new run
$existingContainer = docker ps -a -q -f "name=recbole-exp"
if ($existingContainer) {
    docker rm -f recbole-exp | Out-Null
}

Write-Host "Starting recbole-env container (results will PERSIST in container after exit)..." -ForegroundColor Green
docker run -it --name recbole-exp --gpus all recbole-env
