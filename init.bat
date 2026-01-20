@echo off

docker build -t recbole-env .

docker run -it --rm --name recbole-env --gpus all recbole-env

pause