[![Build and Push Docker Images](https://github.com/am05mhz/comfy-dock/actions/workflows/build.yml/badge.svg)](https://github.com/am05mhz/comfy-dock/actions/workflows/build.yml)

## 🔌 Exposed Ports

| Port | Type | Service     |
| ---- | ---- | ----------- |
| 22   | TCP  | SSH         |
| 3000 | HTTP | ComfyUI     |
| 5000 | HTTP | App         |
| 8080 | HTTP | code-server |

---

## 🏷️ Tag Format

```text
sombi/comfyui:(A)-torch2.8.0-(B)
```

* **(A)**: `slim` or `base`
  * `slim`: ComfyUI + Manager only
  * `base`: slim + pre-installed custom nodes
* **(B)**: CUDA version → `cu124`, `cu126`, `cu128`

---

## 🧱 Image Variants

| Image Name                                  | Custom Nodes | CUDA |
| ------------------------------------------- | ------------ | ---- |
| `am05mhz/pebble-test:base-torch2.8.0-cu126` | ✅ Yes       | 12.6 |
| `am05mhz/pebble-test:base-torch2.8.0-cu128` | ✅ Yes       | 12.8 |

> 👉 To switch: **Edit Pod/Template** → set `Container Image`.

---

## ⚙️ Environment Variables

| Variable                | Description                                                                | Default   |
| ----------------------- | -------------------------------------------------------------------------- | --------- |
| `ACCESS_PASSWORD`       | Password for JupyterLab & code-server                                      | (unset)   |
| `TIME_ZONE`             | [Timezone](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) (e.g., `Asia/Seoul`)   | `Etc/UTC` |
| `COMFYUI_EXTRA_ARGS`    | Extra ComfyUI options (e.g. `--fast`)                        | (unset)   |
| `INSTALL_SAGEATTENTION` | Install [SageAttention2](https://github.com/thu-ml/SageAttention) on start (`True`/`False`) | `False`    |
| `PRESET_DOWNLOAD`       | Download model presets at startup (comma-separated list). **See below**.                  | (unset)   |

> 👉 To set: **Edit Pod/Template** → **Add Environment Variable** (Key/Value).

> ⚠️ SageAttention2 requires **Ampere+ GPUs** and ~5 minutes to install.

---

## 🔧 Preset Downloads

> The `PRESET_DOWNLOAD` environment variable accepts either a **single preset** or **multiple presets** separated by commas.\
> (e.g. `WAINSFW_V140` or `WAN22_I2V_A14B_GGUF_Q8_0,WAN22_LIGHTNING_LORA,WAN22_NSFW_LORA`) \
> When set, the container will automatically download the corresponding models on startup.

> You can also manually run the preset download script **inside JupyterLab or code-server**:

```bash
bash /download_presets.sh PRESET1,PRESET2,...
```

> 👉 To see which presets are available and view the download list for each, check the [Wiki](https://github.com/somb1/ComfyUI-Docker/wiki/PRESET_DOWNLOAD).

---

## 📁 Logs

| App         | Log Path                                   |
| ----------- | ------------------------------------------ |
| ComfyUI     | `/workspace/ComfyUI/user/comfyui_3000.log` |
| code-server | `/workspace/logs/code-server.log`          |

---

## 🧩 Pre-installed Components

### System

* **OS**: Ubuntu 24.04 (22.02 for CUDA 12.4)
* **Python**: 3.13
* **Framework**: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) + [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager) + [JupyterLab](https://jupyter.org/) + [code-server](https://github.com/coder/code-server)
* **Libraries**: PyTorch 2.8.0, CUDA (12.4–12.8), Triton, [hf\_hub](https://huggingface.co/docs/huggingface_hub), [nvtop](https://github.com/Syllo/nvtop)

#### Custom Nodes (only in **base** image)

* ComfyUI-KJNodes
* ComfyUI-WanVideoWrapper
* ComfyUI-GGUF
* ComfyUI-Easy-Use
* ComfyUI-Frame-Interpolation
* ComfyUI-mxToolkit
* ComfyUI-MultiGPU
* ComfyUI_TensorRT
* ComfyUI_UltimateSDUpscale
* comfyui-prompt-reader-node
* ComfyUI_essentials
* ComfyUI-Impact-Pack
* ComfyUI-Impact-Subpack
* efficiency-nodes-comfyui
* ComfyUI-Custom-Scripts
* ComfyUI_JPS-Nodes
* cg-use-everywhere
* ComfyUI-Crystools
* rgthree-comfy
* ComfyUI-Image-Saver
* comfy-ex-tagcomplete
* ComfyUI-VideoHelperSuite
* ComfyUI-wanBlockswap
* ComfyUI-Manager
* ComfyUI-Queue-Manager

> 👉 More details in the [Wiki](https://github.com/somb1/ComfyUI-Docker/wiki/Custom-Nodes).
