@echo off

docker build -t recbole-env .

@REM docker run -it --rm --name recbole-env --gpus all recbole-env

docker run --memory=24g --memory-swap=48g --gpus all -it --rm -v "C:\Users\User\Documents\AdamkaRzeczy\GitHub\RecBole:/root/RecBole" recbole-env

pause