@echo off

docker run --rm --name recbole-env -it --gpus all -v %cd%:/workspace recbole-env