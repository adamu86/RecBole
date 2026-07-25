@echo off

docker image inspect recbole-env >nul 2>&1
if %errorlevel% neq 0 (
    echo Building recbole-env image...
    docker build -t recbole-env .
)

echo Starting recbole-env image...
docker run -p 8000:8000 --rm --name recbole-env -it --gpus all -v %cd%:/workspace recbole-env

@REM docker run -p 8000:8000 --rm --name recbole-env -it --gpus all -v "${PWD}:/workspace" recbole-env