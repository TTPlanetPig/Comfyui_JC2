# 🧠 ComfyUI Joy Caption Wrapper (Alpha Two & Beta One)

> 💡 支持 Alpha Two 与全新 Beta One 模型  
> 🎮 一键部署 / 自动下载（Beta One 模型无需手动放置）  
> 📦 GitHub Repo: https://github.com/TTPlanetPig/Comfyui_JC2
>
> Comfyui workflow example：
> https://github.com/TTPlanetPig/Comfyui_JC2/blob/main/example/JoyCaption%20Beta_One_example.png

---

## 🌟 简介(这里为旧的Joy Caption Alpha Two介绍，已过时）

这是为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 封装的 Joy Caption 节点：

- ✅ 支持 `joy-caption-alpha-two` 以及 **全新版本** [`joy-caption-beta-one`](https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one)
- 🧊 对低显存卡用户推荐使用 `nf4` 模式，兼顾速度与效果
- 🔁 参考实现来自：
  - [chflame163/ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)
  - [John6666/joy-caption-alpha-two-cli-mod](https://huggingface.co/John6666/joy-caption-alpha-two-cli-mod)

---

## ⚠️ VRAM 要求

| 模式 | 最低显存 | 说明 |
|------|-----------|------|
| `bf16` | ≥ 19GB | 推荐给 3090 / 4090 用户 |
| `nf4`  | ≥ 10GB | 推荐低于 19GB 显存时使用 |

> 显存不足将导致 ComfyUI 报错或运行失败。

---

## 🚀 安装方式

### ✅ 安装节点：

方法一：通过 ComfyUI 内置 Manager 安装  
方法二：手动克隆

```bash
cd ./comfyui/custom_nodes
git clone https://github.com/TTPlanetPig/Comfyui_JC2
```

### ✅ 安装依赖（适用于 `python_embedded`）：

```bash
cd ./comfyui/custom_nodes/Comfyui_JC2
../../../python_embeded/python.exe -m pip install -r requirements.txt
```

### ✅ 安装 PyTorch（如果未预装）

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚅 提速建议：开启 Liger Kernel

若需要进一步**提升运行速度**，推荐启用 **liger-kernel**：

- 📁 请点击节点目录下的 `安装liger-kernel.bat`
- ✅ 适用于 ComfyUI 官方一键包（`python_embeded` 构建）

---

## 📥 模型准备

| 模型 | 下载链接 | 放置路径 |
|------|-----------|----------|
| `clip_vision` | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | `ComfyUI/models/clip_vision/google--siglip-so400m-patch14-384` |
| `LLM` | [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct) | `ComfyUI/models/LLM/unsloth--Meta-Llama-3.1-8B-Instruct` |
| `Joy Caption LoRA` (alpha two) | [joy-caption-alpha-two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two) | `ComfyUI/models/Joy_caption/cgrkzexw-599808` |

📦 推荐使用 `huggingface-cli` 下载以避免路径或名称出错。

或者使用百度网盘打包下载：

> 链接: https://pan.baidu.com/s/1yYRlDKclehSPv-tUVwfVHw 提取码: `b84c`

---

## 🆕 新增支持：joy-caption-beta-one 🎉

- ✅ 已集成 [joy-caption-beta-one](https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one)
- ✅ **无需用户手动下载模型**，ComfyUI 节点会自动拉取 HuggingFace 资源
- ✅ 保持与 Alpha Two 同样的调用逻辑，支持 GPU 显存检测与模式选择

---

## 📸 界面 & 使用要点

### 🎛 关键参数介绍：

1. 模式选择（`bf16` / `nf4`）  
   推荐 3090 / 4090 使用 `bf16`，其余使用 `nf4`  
   ![bf16 vs nf4](https://github.com/user-attachments/assets/8001e70b-cea3-4971-a8c2-f483a2c4f91c)

2. 提示词模式选择（多种任务类型）  
   ![prompt type](https://github.com/user-attachments/assets/110f25f6-ea25-4395-b698-c0ec358940ae)

3. 文本长度选择（不总是越长越好）  
   ![length not always better](https://github.com/user-attachments/assets/05e8cfbe-f983-4c8e-813a-761779d0ba4e)

4. 模型 offload 开关（决定是否将模型保留在显存）  
   ![offload setting](https://github.com/user-attachments/assets/804d3326-0f44-4cd2-98c9-56e174e552c1)

5. 控制附加选项是否生效（需搭配使用）  
   ![extra enable](https://github.com/user-attachments/assets/6cb00a63-a1e6-4502-87ff-b99800d37912)

6. 联动选项，需同时启用才有效果  
   ![combo 1](https://github.com/user-attachments/assets/16d11016-6ff1-4d62-90ca-c3d820af4cd3)  
   ![combo 2](https://github.com/user-attachments/assets/6fe8dbd4-affe-4753-b10e-aa4120ab5149)

---

## 🖼 文件夹结构示意

确保模型文件正确放置，如图所示：

![结构1](https://github.com/user-attachments/assets/4675b67c-38f8-4d6a-9785-607215038337)  
![结构2](https://github.com/user-attachments/assets/9ae0a410-539e-49c5-a1b4-4434da02dc28)  
![结构3](https://github.com/user-attachments/assets/2d17e8d2-42af-4040-9cf9-019eb25464e0)  
![结构4](https://github.com/user-attachments/assets/aeba0145-81c7-4c86-a31c-bbb9c317cad8)

---

## ⭐ Star History

<a href="https://star-history.com/#TTPlanetPig/Comfyui_JC2&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_JC2&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_JC2&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_JC2&type=Date" />
 </picture>
</a>

---

🧪 欢迎测试并反馈问题，Enjoy!
