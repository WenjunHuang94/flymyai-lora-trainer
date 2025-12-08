import torch
import time
# 【【【 修改1：我们现在需要导入模型类本身, 但只用它们的 .load_config 方法 】】】
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
# 【【【 修改2：我们仍然需要 transformers.AutoConfig, 专门给 VL 用 】】】
from transformers import AutoConfig
from transformers import logging as hf_logging

# 压制加载模型时的非必要警告
hf_logging.set_verbosity_error()

# --- 配置 ---
model_path = "Qwen/Qwen-Image"  # 基础模型
print(f"正在从 {model_path} 仅加载 'config.json' 文件...")
print("=" * 60 + "\n")

# --- 1. VAE (变分自编码器) ---
print("=" * 25 + " 1. VAE 配置 (AutoencoderKLQwenImageConfig) " + "=" * 25)
load_start = time.time()
try:
    # 【【【 关键修改：调用 .load_config() 而不是 AutoConfig 】】】
    # 这只会加载 config.json, 不会加载权重
    vae_config = AutoencoderKLQwenImage.load_config(
        model_path,
        subfolder="vae"
    )

    print(f"VAE 配置加载完毕 (耗时: {time.time() - load_start:.2f}s)\n")

    # 打印 VAE 的完整结构配置
    print(vae_config)

    print("\n\n" + "=" * 80 + "\n\n")

except Exception as e:
    print(f"加载 VAE 配置失败: {e}")

# --- 2. DiT (扩散 Transformer) ---
print("=" * 20 + " 2. DiT 配置 (QwenImageTransformer2DModel Config) " + "=" * 20)
load_start = time.time()
try:
    # 【【【 关键修改：调用 .load_config() 而不是 AutoConfig 】】】
    dit_config = QwenImageTransformer2DModel.load_config(
        model_path,
        subfolder="transformer"
    )

    print(f"DiT 配置加载完毕 (耗时: {time.time() - load_start:.2f}s)\n")

    # 打印 DiT 的完整结构配置
    print(dit_config)

    print("\n\n" + "=" * 80 + "\n\n")

except Exception as e:
    print(f"加载 DiT 配置失败: {e}")

# --- 3. Qwen-VL (文本编码器) ---
print("=" * 20 + " 3. Qwen-VL 结构 (Text Encoder Config) " + "=" * 20)
load_start = time.time()
try:
    # 【【【 保持不变：AutoConfig 适用于 transformers 库的模型 】】】
    vl_config = AutoConfig.from_pretrained(
        model_path,
        subfolder="text_encoder"
    )

    print(f"Qwen-VL (Text Encoder) 配置加载完毕 (耗时: {time.time() - load_start:.2f}s)\n")

    # 打印 Qwen-VL 的完整结构配置
    print(vl_config)

    print("\n\n" + "=" * 80 + "\n\n")

except Exception as e:
    print(f"加载 Qwen-VL 配置失败: {e}")

print("模型配置审查完毕。没有加载任何模型权重。")


