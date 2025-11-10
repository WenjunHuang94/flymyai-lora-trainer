import torch
import time
from diffusers import (
    AutoencoderKLQwenImage,  # 这是 VAE
    QwenImageTransformer2DModel,  # 这是 DiT (flux_transformer)
    QwenImagePipeline  # 这是加载 Qwen-VL (text_encoder) 最简单的方式
)
from transformers import logging as hf_logging

# 压制加载模型时的非必要警告
hf_logging.set_verbosity_error()

# --- 配置 ---
model_path = "Qwen/Qwen-Image"  # 基础模型
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在从 {model_path} 加载模型到 {device}...")
print("=" * 60 + "\n")

# --- 1. VAE (变分自编码器) ---
# 作用：将 1024x1024 的图片压缩成潜向量 (Latent)
print("=" * 25 + " 1. VAE 结构 (AutoencoderKLQwenImage) " + "=" * 25)
load_start = time.time()
try:
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype
    ).to(device)

    print(f"VAE 加载完毕 (耗时: {time.time() - load_start:.2f}s)\n")

    # 打印 VAE 的完整结构
    print(vae)

    print("\n\n" + "=" * 80 + "\n\n")

except Exception as e:
    print(f"加载 VAE 失败: {e}")

# 释放内存
del vae
torch.cuda.empty_cache()

# --- 2. DiT (扩散 Transformer) ---
# 作用：“主大脑”，在潜空间中进行“去噪”
print("=" * 20 + " 2. DiT 结构 (QwenImageTransformer2DModel / flux_transformer) " + "=" * 20)
load_start = time.time()
try:
    dit = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype
    ).to(device)

    print(f"DiT 加载完毕 (耗时: {time.time() - load_start:.2f}s)\n")

    # 打印 DiT 的完整结构
    print(dit)

    print("\n\n" + "=" * 80 + "\n\n")

except Exception as e:
    print(f"加载 DiT 失败: {e}")

# 释放内存
del dit
torch.cuda.empty_cache()

# --- 3. Qwen-VL (文本编码器) ---
# 作用：将您的文本提示（"ohwhwj man..."）转换成 DiT 能理解的数学向量
print("=" * 20 + " 3. Qwen-VL 结构 (pipe.text_encoder) " + "=" * 20)
load_start = time.time()
try:
    # 我们加载 Pipeline 只是为了方便地访问它内部的 .text_encoder
    # 我们通过 transformer=None 和 vae=None 告诉它“不要”重复加载
    pipe = QwenImagePipeline.from_pretrained(
        model_path,
        transformer=None,  # 不加载 DiT
        vae=None,  # 不加载 VAE
        torch_dtype=dtype
    ).to(device)

    qwen_vl = pipe.text_encoder

    print(f"Qwen-VL (Text Encoder) 加载完毕 (耗时: {time.time() - load_start:.2f}s)\n")

    # 打印 Qwen-VL (Qwen2_5_VLForConditionalGeneration) 的完整结构
    print(qwen_vl)

    print("\n\n" + "=" * 80 + "\n\n")

except Exception as e:
    print(f"加载 Qwen-VL 失败: {e}")

# 释放内存
del pipe, qwen_vl
torch.cuda.empty_cache()

print("模型结构审查完毕。")