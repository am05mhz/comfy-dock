#!/bin/bash

export PYTHONUNBUFFERED=1

echo "**** Setting the timezone based on the TIME_ZONE environment variable. If not set, it defaults to Etc/UTC. ****"
export TZ=${TIME_ZONE:-"Etc/UTC"}
echo "**** Timezone set to $TZ ****"
echo "$TZ" | sudo tee /etc/timezone > /dev/null
sudo ln -sf "/usr/share/zoneinfo/$TZ" /etc/localtime
sudo dpkg-reconfigure -f noninteractive tzdata

# echo "**** syncing ComfyUI to workspace, please wait ****"
# if [ -d /ComfyUI ]; then

#     SRC_MODELS="/ComfyUI/models"
#     DST_MODELS="/workspace/ComfyUI/models"

#     EXCLUDE_MODELS=""

#     if [ -d "$DST_MODELS" ] && [ "$(ls -A "$DST_MODELS")" ]; then
#         for d in "$DST_MODELS"/*/; do
#             [ -d "$d" ] || continue
#             folder_name=$(basename "$d")
#             EXCLUDE_MODELS="$EXCLUDE_MODELS --exclude='models/$folder_name/**'"
#         done
#         echo "**** Excluding existing model folders: $EXCLUDE_MODELS ****"
#     fi

#     if [ -d /workspace/ComfyUI/output ]; then
#         EXCLUDE_MODELS="$EXCLUDE_MODELS --exclude='output/'"
#         echo "**** Excluding existing output folder ****"
#     fi

#     rsync -au --remove-source-files $EXCLUDE_MODELS /ComfyUI/ /workspace/ComfyUI/ && rm -rf /ComfyUI

# else
#     echo "Skip: /ComfyUI does not exist."
# fi

# if [ "${INSTALL_SAGEATTENTION,,}" = "true" ]; then
#     if pip show sageattention > /dev/null 2>&1; then
#         echo "**** SageAttention2 is already installed. Skipping installation. ****"
#     else
#         echo "**** SageAttention2 is not installed. Installing, please wait.... (This may take a long time, approximately 5+ minutes.) ****"
#         git clone https://github.com/thu-ml/SageAttention.git /SageAttention
#         cd /SageAttention
#         python setup.py install
#         echo "**** SageAttention2 installation completed. ****"
#     fi
# fi

# if [ "${INSTALL_CUSTOM_NODES,,}" = "true" ]; then
#     if [ -f /install_custom_nodes.sh ]; then
#         echo "**** INSTALL_CUSTOM_NODES is set. Running /install_custom_nodes.sh ****"
#         /install_custom_nodes.sh
#     else
#         echo "**** /install_custom_nodes.sh not found. Skipping. ****"
#     fi
# fi

/workspace/download_models.sh --quiet "${PRESET_DOWNLOAD}"
