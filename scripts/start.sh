#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #

# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh

        ssh-keygen -A       # regenerate new keys
        service ssh start
    fi
}

# Start nginx service
start_nginx() {
    echo "Starting Nginx service..."
    service nginx start
}

# Execute script if exists
execute_script() {
    local script_path=$1
    local script_msg=$2
    if [[ -f ${script_path} ]]; then
        echo "${script_msg}"
        bash ${script_path}
    fi
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# Start code-server
start_code_server() {
    if [[ "${RUN_APP}" ]]; then
        echo "Starting code-server..."
        mkdir -p /workspace/logs
        # Allow a password to be set by providing the ACCESS_PASSWORD environment variable
        if [[ -n "${ACCESS_PASSWORD}" ]]; then
            echo "Starting code-server with the provided password..."
            export PASSWORD="${ACCESS_PASSWORD}"
            nohup code-server /workspcae --bind-addr 0.0.0.0:8080 \
                --auth password \
                --ignore-last-opened \
                --disable-workspace-trust \
                &> /workspace/logs/code-server.log &
        else
            echo "Starting code-server without a password... (ACCESS_PASSWORD environment variable is not set.)"
            nohup code-server /workspace --bind-addr 0.0.0.0:8080 \
                --auth none \
                --ignore-last-opened \
                --disable-workspace-trust \
                &> /workspace/logs/code-server.log &
        fi
        echo "code-server started"
    fi
}

start_comfyui() {
    echo "Starting ComfyUI..."
    source /workspace/miniconda3/bin/activate comfy
    cd /workspace/ComfyUI
    nohup python main.py --listen --port 3000 $COMFYUI_EXTRA_ARGS > /proc/self/fd/1 2>&1 &
}

start_app() {
    if [[ "${RUN_APP}" ]]; then
        echo "Starting app..."
        if [[ ! "$PATH" == *"/workspace/miniconda3/bin"* ]]; then
            export PATH="/workspace/miniconda3/bin:$PATH"
        fi
        source /workspace/miniconda3/bin/activate comfy
        cd /workspace/app
        nohup python app.py &> /workspace/logs/app.log &
    fi
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

setup_ssh
start_nginx

execute_script "/setup/pre_start.sh" "Running pre-start script..."
echo "fix huggingface_hub"
source /workspace/miniconda3/bin/activate comfy
pip uninstall -y huggingface_hub
pip install "huggingface_hub<1.0"
echo "Pod Started"

export_env_vars
start_code_server
start_comfyui
start_app

echo "Start script(s) finished, pod is ready to use."

sleep infinity
