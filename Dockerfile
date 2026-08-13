FROM ubuntu:24.04

ARG ENV_NAME=recbole-env
ARG WORKSPACE=/workspace

ENV ENV_NAME=${ENV_NAME}
ENV WORKSPACE=${WORKSPACE}

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV CONDA_DIR=/root/miniforge3
ENV PATH=${CONDA_DIR}/bin:${PATH}

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

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
    ca-certificates \
    zsh \
    tree \
    fonts-powerline \
    fzf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN wget -q "https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-25.11.0-1-Linux-x86_64.sh" -O miniforge.sh && \
    bash miniforge.sh -b -p ${CONDA_DIR} && \
    rm miniforge.sh && \
    conda clean -afy && \
    conda init bash && \
    conda init zsh

WORKDIR ${WORKSPACE}

# Copy repository files (excluding paths defined in .dockerignore: log/, saved/, dataset_raw/, etc.)
COPY . ${WORKSPACE}

RUN mamba env create -f conda/environment.yml -y

RUN bash -c "source ${CONDA_DIR}/bin/activate recbole && \
    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
    pip install numpy==1.24.4 && \
    pip install -e ."

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

RUN git clone https://github.com/zsh-users/zsh-autosuggestions \
    ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git \
    ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' /root/.zshrc

RUN echo 'ZSH_THEME=""' >> /root/.zshrc && \
    echo 'setopt PROMPT_SUBST' >> /root/.zshrc && \
    echo 'PROMPT="%F{green}${ENV_NAME}%f:%F{blue}\${PWD#\${WORKSPACE}}%f$ "' >> /root/.zshrc && \
    echo "source ${CONDA_DIR}/bin/activate recbole" >> /root/.zshrc && \
    echo "conda activate recbole" >> /root/.zshrc

WORKDIR ${WORKSPACE}

CMD ["zsh"]

# ==============================================================================
# Instructions for running experiments purely inside container (No bind mounts)
# ==============================================================================
# 1. Build the self-contained container image:
#    docker build -t recbole-env .
#
# 2. Run experiments inside container (standalone, no host volume mounts):
#    docker run --rm -it --gpus all recbole-env
#
# 3. Run a specific script directly (non-interactive):
#    docker run --rm --gpus all recbole-env python run_recbole.py
#
# 4. Extract logs/results from container if ever needed:
#    docker cp <container_name_or_id>:/workspace/log ./log
# ==============================================================================