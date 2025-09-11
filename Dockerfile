# Set the base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set the shell and enable pipefail for better error handling
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set basic environment variables
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG CUDA_VERSION

# Set basic environment variables
ENV SHELL=/bin/bash 
ENV PYTHONUNBUFFERED=True 
ENV DEBIAN_FRONTEND=noninteractive

# Set the default workspace directory
ENV RP_WORKSPACE=/workspace

# Override the default huggingface cache directory.
ENV HF_HOME="${RP_WORKSPACE}/.cache/huggingface/"

# Faster transfer of models from the hub to the container
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1

# Shared python package cache
ENV VIRTUALENV_OVERRIDE_APP_DATA="${RP_WORKSPACE}/.cache/virtualenv/"
ENV PIP_CACHE_DIR="${RP_WORKSPACE}/.cache/pip/"
ENV UV_CACHE_DIR="${RP_WORKSPACE}/.cache/uv/"

# modern pip workarounds
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore

# Set TZ and Locale
ENV TZ=Etc/UTC

# Set working directory
WORKDIR /

# Update and upgrade
RUN apt-get update --yes && \
    apt-get upgrade --yes

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Install essential packages
RUN apt-get install --yes --no-install-recommends \
        git wget curl bash nginx-light rsync sudo binutils ffmpeg lshw nano tzdata file build-essential cmake nvtop \
        libgl1 libglib2.0-0 clang libomp-dev ninja-build \
        openssh-server ca-certificates && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# NGINX Proxy
COPY proxy/nginx.conf /etc/nginx/nginx.conf
COPY proxy/snippets /etc/nginx/snippets
COPY proxy/readme.html /usr/share/nginx/html/readme.html

# Copy setup files
RUN mkdir -p /setup/app
COPY custom_nodes.txt /setup/custom_nodes.txt

# app
COPY app/app.py /setup/app/
COPY app/requirements.txt /setup/app/
# RUN cd /workspace/ComfyUI && \
#     pip install --no-cache-dir -r requirements.txt

# Copy the README.md
COPY README.md /usr/share/nginx/html/README.md

# Start Scripts
COPY --chmod=755 scripts/start.sh /setup/
COPY --chmod=755 scripts/pre_start.sh /setup/
COPY --chmod=755 scripts/post_start.sh /setup/

COPY --chmod=755 scripts/download_models.sh /setup/
# COPY --chmod=755 scripts/install_custom_nodes.sh /

# Welcome Message
COPY logo/am05mhz.txt /etc/am05mhz.txt
RUN echo 'cat /etc/am05mhz.txt' >> /root/.bashrc
RUN echo 'echo -e "\nFor detailed documentation and guides, please visit:\n\033[1;34mhttps://docs.runpod.io/\033[0m and \033[1;34mhttps://blog.runpod.io/\033[0m\n\n"' >> /root/.bashrc

# install miniconda3
# RUN cd /workspace && \
#     curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     chmod a+x Miniconda3-latest-Linux-x86_64.sh && \
#     ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
# ENV PATH="/workspace/miniconda3/:$PATH"
# RUN conda create -n comfy python=${PYTHON_VERSION} -y && \
#     conda activate comfy

# # Install essential Python packages and dependencies
# RUN pip install --no-cache-dir -U \
#     pip setuptools wheel \
#     huggingface_hub hf_transfer \
#     numpy scipy matplotlib pandas scikit-learn seaborn requests tqdm pillow pyyaml \
#     triton \
#     torch==${TORCH_VERSION} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install ComfyUI and ComfyUI Manager
# RUN cd /workspace && \
#     git clone https://github.com/comfyanonymous/ComfyUI.git && \
#     cd ComfyUI && \
#     pip install --no-cache-dir -r requirements.txt && \
#     git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager && \
#     cd custom_nodes/ComfyUI-Manager && \
#     pip install --no-cache-dir -r requirements.txt && \
#     cd ../.. && \
#     git clone https://github.com/QuietNoise/ComfyUI-Queue-Manager.git custom_nodes/ComfyUI-Queue-Manager && \
#     cd custom_nodes/ComfyUI-Manager && \
#     pip install --no-cache-dir -r requirements.txt

# RUN cd /workspace/ComfyUI/custom_nodes && \
#     xargs -n 1 git clone --recursive < /workspace/custom_nodes.txt && \
#     find /workspace/ComfyUI/custom_nodes -name "requirements.txt" -exec pip install --no-cache-dir -r {} \; && \
#     find /workspace/ComfyUI/custom_nodes -name "install.py" -exec python {} \;

# Install Runpod CLI
RUN wget -qO- cli.runpod.net | sudo bash

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

EXPOSE 22 3000 5000 8080

# Set entrypoint to the start script
CMD ["/setup/start.sh"]
