@echo off
set PROJECT_PATH=C:\Users\User\Documents\AdamkaRzeczy\MasterDegree\RecBole

docker run --memory=24g --memory-swap=48g --gpus all -it --rm -v "%PROJECT_PATH%:/root/RecBole" recbole-env