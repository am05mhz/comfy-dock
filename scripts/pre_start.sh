#!/bin/bash

export PYTHONUNBUFFERED=1

echo "**** Setting the timezone based on the TIME_ZONE environment variable. If not set, it defaults to Etc/UTC. ****"
export TZ=${TIME_ZONE:-"Etc/UTC"}
echo "**** Timezone set to $TZ ****"
echo "$TZ" | sudo tee /etc/timezone > /dev/null
sudo ln -sf "/usr/share/zoneinfo/$TZ" /etc/localtime
sudo dpkg-reconfigure -f noninteractive tzdata

setup_miniconda() {
    cd /workspace
    curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod a+x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
    if [[ ! "$PATH" == *"/workspace/miniconda3/bin"* ]]; then
        export PATH="/workspace/miniconda3/bin:$PATH"
    fi
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda create -n comfy python=$ARG_PYTHON_VERSION -y
    source /workspace/miniconda3/bin/activate comfy
    pip install --no-cache-dir -U \
        pip setuptools wheel \
        huggingface_hub hf_transfer \
        numpy scipy matplotlib pandas scikit-learn seaborn requests tqdm pillow pyyaml \
        triton \
        torch==$ARG_TORCH_VERSION torchvision torchaudio --extra-index-url "https://download.pytorch.org/whl/$ARG_CUDA_VERSION"
}

setup_comfy() {
    if [[ ! "$PATH" == *"/workspace/miniconda3/bin"* ]]; then
        export PATH="/workspace/miniconda3/bin:$PATH"
    fi
    source /workspace/miniconda3/bin/activate comfy
    cd /workspace
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    pip install --no-cache-dir -r requirements.txt
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
    cd custom_nodes/ComfyUI-Manager
    pip install --no-cache-dir -r requirements.txt
    cd ../..
    git clone https://github.com/QuietNoise/ComfyUI-Queue-Manager.git custom_nodes/ComfyUI-Queue-Manager
    cd custom_nodes/ComfyUI-Manager
    pip install --no-cache-dir -r requirements.txt
}

if [ -d "/workspace" ]; then
    if [ ! -d "/workspace/miniconda3" ]; then
        echo "*** installing miniconda ***"
        setup_miniconda
    fi    
    if [ ! -d "/workspace/ComfyUI" ]; then
        echo "*** installing ComfyUI ***"
        setup_comfy
        /setup/download_models.sh --quiet "${PRESET_DOWNLOAD}"
    fi    
    if [ ! -f "/workspace/custom_nodes.txt" ]; then
        echo "*** installing custom nodes ***"
        cp /setup/custom_nodes.txt /workspace/custom_nodes.txt
        cd /workspace/ComfyUI/custom_nodes
        xargs -n 1 git clone --recursive < /workspace/custom_nodes.txt
        find /workspace/ComfyUI/custom_nodes -name "requirements.txt" -exec pip install --no-cache-dir -r {} \;
        find /workspace/ComfyUI/custom_nodes -name "install.py" -exec python {} \;
    fi
    if [ ! -d "/workspace/app" ]; then
        echo "*** installing app ***"
        cp -r /setup/app /workspace/app
        cd /workspace/app
        pip install --no-cache-dir -r requirements.txt    
    fi
fi
