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

    local filename
    filename=$(basename "$url")
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

echo "**** Downloading default files ****"
# upscale model
download_if_missing "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors" "/workspace/ComfyUI/models/upscale_models" # 66.9MB
download_if_missing "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_RCAN.safetensors" "/workspace/ComfyUI/models/upscale_models"       # 31.1MB

# flux
download_if_missing "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder/model.safetensors" "/workspace/ComfyUI/models/text_encoders"    # 246MB
download_if_missing "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors" "/workspace/ComfyUI/models/vae"     # 168MB
download_if_missing "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors" "/workspace/ComfyUI/models/unet"                  # 23.8GB

# sd 3
download_if_missing "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/text_encoders/t5xxl_fp16.safetensors" "/workspace/ComfyUI/models/text_encoders"   # 4.89GB
download_if_missing "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors" "/workspace/ComfyUI/models/checkpoints"              # 5.97GB

download_if_missing "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors" "/workspace/ComfyUI/models/checkpoints"  # 2.14GB

echo "**** Checking presets and downloading corresponding files ****"

for preset in "${PRESETS[@]}"; do
    case "${preset}" in
        WAINSFW_V140)
            echo "Preset: WAINSFW_V140"
            download_if_missing "https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/waiNSFWIllustrious_v140.safetensors" "/workspace/ComfyUI/models/checkpoints"
            ;;
        NTRMIX_V40)
            echo "Preset: NTRMIX_V40"
            download_if_missing "https://huggingface.co/personal1802/NTRMIXillustrious-XLNoob-XL4.0/resolve/main/ntrMIXIllustriousXL_v40.safetensors" "/workspace/ComfyUI/models/checkpoints"
            ;;
        WAN22_TI2V_5B)
            echo "Preset: WAN22_TI2V_5B"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_T2V_A14B)
            echo "Preset: WAN22_T2V_A14B"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_FP8_SCALED)
            echo "Preset: WAN22_I2V_A14B_FP8_SCALED"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_FP8_E4M3FN_SCALED_KJ)
            echo "Preset: WAN22_I2V_A14B_FP8_E4M3FN_SCALED_KJ"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_FP8_E5M2_SCALED_KJ)
            echo "Preset: WAN22_I2V_A14B_FP8_E5M2_SCALED_KJ"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_GGUF_Q8_0)
            echo "Preset: WAN22_I2V_A14B_GGUF_Q8_0"
            download_if_missing "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q8_0.gguf" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q8_0.gguf" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_GGUF_Q6_K)
            echo "Preset: WAN22_I2V_A14B_GGUF_Q6_K"
            download_if_missing "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q6_K.gguf" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_GGUF_Q5_K_S)
            echo "Preset: WAN22_I2V_A14B_GGUF_Q5_K_S"
            download_if_missing "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_S.gguf" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q5_K_S.gguf" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q5_K_S.gguf" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_GGUF_Q5_K_M)
            echo "Preset: WAN22_I2V_A14B_GGUF_Q5_K_M"
            download_if_missing "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q5_K_M.gguf" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q5_K_M.gguf" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_GGUF_Q4_K_S)
            echo "Preset: WAN22_I2V_A14B_GGUF_Q4_K_S"
            download_if_missing "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_S.gguf" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_S.gguf" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q4_K_S.gguf" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_I2V_A14B_GGUF_Q4_K_M)
            echo "Preset: WAN22_I2V_A14B_GGUF_Q4_K_M"
            download_if_missing "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_M.gguf" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_M.gguf" "/workspace/ComfyUI/models/diffusion_models"
            download_if_missing "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q4_K_M.gguf" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        WAN22_LIGHTNING_LORA)
            echo "Preset: WAN22_LIGHTNING_LORA"
            download_if_missing "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors" "/workspace/ComfyUI/models/loras"
            download_if_missing "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors" "/workspace/ComfyUI/models/loras"
            ;;
        WAN22_NSFW_LORA)
            echo "Preset: WAN22_NSFW_LORA"
            download_if_missing "https://huggingface.co/sombi/comfyui_models/resolve/main/Wan2.2_nsfw_lora_v0.08a/NSFW-22-H-e8.safetensors" "/workspace/ComfyUI/models/loras"
            download_if_missing "https://huggingface.co/sombi/comfyui_models/resolve/main/Wan2.2_nsfw_lora_v0.08a/NSFW-22-L-e8.safetensors" "/workspace/ComfyUI/models/loras"
            ;;
        # UPSCALE_MODELS)
        #     echo "Preset: UPSCALE_MODELS"
        #     download_if_missing "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors" "/workspace/ComfyUI/models/upscale_models"
        #     download_if_missing "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_RCAN.safetensors" "/workspace/ComfyUI/models/upscale_models"
        #     ;;
        WAN22_S2V_FP8_SCALED)
            echo "Preset: WAN22_S2V_FP8_SCALED"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "/workspace/ComfyUI/models/text_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors" "/workspace/ComfyUI/models/audio_encoders"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "/workspace/ComfyUI/models/vae"
            download_if_missing "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors" "/workspace/ComfyUI/models/diffusion_models"
            ;;
        *)
            echo "No matching preset for '${preset}', skipping."
            ;;
    esac
done
