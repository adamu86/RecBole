@echo off

echo Building recbole-env image (including datasets, excluding saved and log)...
docker build -t recbole-env .

if %errorlevel% neq 0 (
    echo Docker build failed!
    exit /b %errorlevel%
)

echo Removing previous container instance if exists...
docker rm -f recbole-exp >nul 2>&1

echo Starting recbole-env container (results will PERSIST in container after exit)...
docker run -it --name recbole-exp --gpus all recbole-env