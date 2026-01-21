#!/bin/bash

WGET_OPTS="--show-progress"

if [[ "$1" == "--quiet" ]]; then
    WGET_OPTS="-q"
    shift
fi

WGET_HF_TOKEN=""
if [[ $HF_TOKEN ]]; then
    WGET_HF_TOKEN="--header='Authorization: Bearer ${HF_TOKEN}"
fi

# download_if_missing <URL> <TARGET_DIR>
download_if_missing() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"

    if [ -z "$filename" ]; then
        filename=$(basename "$url")
    fi
    local filepath="$dest_dir/$filename"

    mkdir -p "$dest_dir"

    if [ -f "$filepath" ]; then
        echo "File already exists: $filepath (skipping)"
        return
    fi

    echo "Downloading: $filename → $dest_dir"
    
    local tmpdir="/workspace/tmp"
    mkdir -p "$tmpdir"
    local tmpfile="$tmpdir/${filename}.part"

    if wget $WGET_OPTS $WGET_HF_TOKEN -O "$tmpfile" "$url"; then
        mv -f "$tmpfile" "$filepath"
        echo "Download completed: $filepath"
    else
        echo "Download failed: $url"
        rm -f "$tmpfile"
        return 1
    fi
}

IFS=',' read -ra PRESETS <<< "$1"

echo "**** Downloading default models ****"
# upscale model
download_if_missing "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors" "/workspace/ComfyUI/models/upscale_models" # 66.9MB
download_if_missing "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_RCAN.safetensors" "/workspace/ComfyUI/models/upscale_models"       # 31.1MB


download_if_missing "https://huggingface.co/gitgato/EpicReal/resolve/fd2b9d09eb5dbd5fb5a0aedfb9c24c3e0196c214/epicrealism_naturalSinRC1VAE.safetensors" "/workspace/ComfyUI/models/checkpoints"
download_if_missing "https://huggingface.co/xxiaogui/hongchao/resolve/main/juggernautXL_ragnarokBy.safetensors" "/workspace/ComfyUI/models/checkpoints"
download_if_missing "https://huggingface.co/AiWise/Juggernaut-XL-V9-GE-RDPhoto2-Lightning_4S/resolve/main/juggernautXL_v9Rdphoto2Lightning.safetensors" "/workspace/ComfyUI/models/checkpoints"
download_if_missing "https://huggingface.co/khaimd123/realisticVisionV60B1_v51VAE/resolve/5314c8485d833085571162c6bd3664eabdc6b25f/realisticVisionV60B1_v51VAE.safetensors" "/workspace/ComfyUI/models/checkpoints"

download_if_missing "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "/workspace/ComfyUI/models/clip"

download_if_missing "https://huggingface.co/fofr/comfyui/resolve/a25ad5613692b9593ea6d126b0451191cf420765/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" "/workspace/ComfyUI/models/clip_vision"
download_if_missing "https://huggingface.co/fofr/comfyui/resolve/a25ad5613692b9593ea6d126b0451191cf420765/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" "/workspace/ComfyUI/models/clip_vision"
download_if_missing "https://huggingface.co/fofr/comfyui/resolve/a25ad5613692b9593ea6d126b0451191cf420765/clip_vision/clip-vit-large-patch14.bin" "/workspace/ComfyUI/models/clip_vision"

download_if_missing "https://huggingface.co/ckpt/controlnet-sdxl-1.0/resolve/main/control-lora-depth-rank256.safetensors" "/workspace/ComfyUI/models/controlnet"

download_if_missing "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors" "/workspace/ComfyUI/models/diffusion_models"
download_if_missing "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors" "/workspace/ComfyUI/models/diffusion_models"
download_if_missing "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors" "/workspace/ComfyUI/models/diffusion_models"
download_if_missing "https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_layered_fp8mixed.safetensors" "/workspace/ComfyUI/models/diffusion_models"
download_if_missing "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "/workspace/ComfyUI/models/diffusion_models"
download_if_missing "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors" "/workspace/ComfyUI/models/diffusion_models"

download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors" "/workspace/ComfyUI/models/ipadapter"

download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin" "/workspace/ComfyUI/models/ipadapter"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl_unnorm.bin" "/workspace/ComfyUI/models/ipadapter"

download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.0.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/FoxBaze/Try_On_Qwen_Edit_Lora_Alpha/resolve/main/Try_On_Qwen_Edit_Lora.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors" "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors " "/workspace/ComfyUI/models/loras"
download_if_missing "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0.safetensors" "/workspace/ComfyUI/models/loras/qwen-image"
download_if_missing "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors" "/workspace/ComfyUI/models/loras/qwen-edit/2511/"
download_if_missing "https://toot.bot.nu/joy/ckpt/blending.safetensors" "/workspace/ComfyUI/models/loras/qwen-edit"
download_if_missing "https://toot.bot.nu/joy/ckpt/wh1t3bg.safetensors" "/workspace/ComfyUI/models/loras/training"

download_if_missing "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors" "/workspace/ComfyUI/models/text_encoders"
download_if_missing "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" "/workspace/ComfyUI/models/text_encoders"

download_if_missing "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors" "/workspace/ComfyUI/models/unet"
download_if_missing "https://huggingface.co/theunlikely/Qwen-Image-Edit-2509/resolve/main/qwen_image_edit_2509_fp8_e4m3fn.safetensors" "/workspace/ComfyUI/models/unet"

download_if_missing "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" "/workspace/ComfyUI/models/vae"
download_if_missing "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" "/workspace/ComfyUI/models/vae"
download_if_missing "https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/resolve/main/split_files/vae/qwen_image_layered_vae.safetensors" "/workspace/ComfyUI/models/vae"
download_if_missing "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "/workspace/ComfyUI/models/vae"
download_if_missing "https://toot.bot.nu/joy/ckpt/ultrafluxv1.safetensors" "/workspace/ComfyUI/models/vae"

echo "done"
