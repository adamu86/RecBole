@echo off

docker build -t recbole-env .

docker run --rm --name recbole-env -it --gpus all -v %cd%:/workspace recbole-env