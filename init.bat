@echo off

docker rm -f recbole-env 2>nul

docker build -t recbole-env .

docker run -it --rm recbole-env