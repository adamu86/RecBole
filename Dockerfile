FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV CONDA_DIR=/root/miniforge3
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        software-properties-common \
        curl \
        wget \
        git \
        vim \
        unzip \
        htop \
        man \
        byobu \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN wget -q "https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-25.11.0-1-Linux-x86_64.sh" -O miniforge.sh && \
    bash miniforge.sh -b -p ${CONDA_DIR} && \
    rm miniforge.sh && \
    conda clean -afy && \
    conda init bash

RUN git clone --branch session-based https://github.com/adamu86/RecBole.git

WORKDIR /root/RecBole

RUN mamba env update -n base -f conda/environment.yml -y

CMD ["/bin/bash"]