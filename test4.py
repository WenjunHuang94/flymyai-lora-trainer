import torch
import time
# 【【【 修改1：导入模型类本身 】】】
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
# 【【【 修改2：导入transformers的AutoConfig和Qwen-VL的具体模型类 】】】
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
from transformers import logging as hf_logging

# 压制加载模型时的非必要警告
hf_logging.set_verbosity_error()

# --- 配置 ---
# model_path = "Qwen/Qwen-Image"
model_path = "Qwen/Qwen-Image-Edit" # <--- 【【请修改成这个】】

print(f"正在从 {model_path} 加载配置, 并使用 'meta' device 快速实例化架构...")
print("=" * 60 + "\n")

# --- 使用 'meta' device 上下文 ---
# 这样创建的模型只有架构，没有权重，不占内存，速度极快
with torch.device("meta"):
    # --- 1. VAE (变分自编码器) ---
    print("=" * 25 + " 1. VAE 架构 (AutoencoderKLQwenImage) " + "=" * 25)
    load_start = time.time()
    try:
        # 第1步：加载配置 (同脚本1)
        vae_config = AutoencoderKLQwenImage.load_config(
            model_path,
            subfolder="vae"
        )

        print('vae_config = ', vae_config)

        # 【【【 关键修改：使用 .from_config() 从配置实例化模型 】】】
        # .from_config() 会根据配置蓝图，搭建出模型的“空壳”
        vae_model = AutoencoderKLQwenImage.from_config(vae_config)

        print(f"VAE 架构实例化完毕 (耗时: {time.time() - load_start:.4f}s)\n")

        # 打印模型实例，你将看到完整的架构
        print(vae_model)

        # 计算参数量
        params = sum(p.numel() for p in vae_model.parameters())
        print(f"vae_model  总参数量: {params / 1e9:.2f}B")

        print("\n\n" + "=" * 80 + "\n\n")

    except Exception as e:
        print(f"实例化 VAE 架构失败: {e}")

    # --- 2. DiT (扩散 Transformer) ---
    print("=" * 20 + " 2. DiT 架构 (QwenImageTransformer2DModel) " + "=" * 20)
    load_start = time.time()
    try:
        # 第1步：加载配置
        dit_config = QwenImageTransformer2DModel.load_config(
            model_path,
            subfolder="transformer"
        )

        print('dit_config = ', dit_config)

        # 【【【 关键修改：使用 .from_config() 实例化模型 】】】
        dit_model = QwenImageTransformer2DModel.from_config(dit_config)

        print(f"DiT 架构实例化完毕 (耗时: {time.time() - load_start:.4f}s)\n")

        # 打印模型实例
        print(dit_model)

        # 计算参数量
        params = sum(p.numel() for p in dit_model.parameters())
        print(f"dit_model  总参数量: {params / 1e9:.2f}B")

        print("\n\n" + "=" * 80 + "\n\n")

    except Exception as e:
        print(f"加载 DiT 架构失败: {e}")

    # --- 3. Qwen-VL (文本编码器) ---
    print("=" * 20 + " 3. Qwen-VL 架构 (Text Encoder) " + "=" * 20)
    load_start = time.time()
    try:
        # 第1步：加载配置
        vl_config = AutoConfig.from_pretrained(
            model_path,
            subfolder="text_encoder"
        )

        print('vl_config = ', vl_config)

        # 【【【 关键修改：使用 Qwen2_5_VLForConditionalGeneration.from_config() 实例化 】】】
        # 注意：我们这里用了具体的类名，这是最标准的方法
        from transformers import AutoModel

        vl_model = AutoModel.from_config(vl_config)

        print(f"Qwen-VL (Text Encoder) 架构实例化完毕 (耗时: {time.time() - load_start:.4f}s)\n")

        # 打印模型实例
        print(vl_model)

        # 计算参数量
        params = sum(p.numel() for p in vl_model.parameters())
        print(f"vl_model  总参数量: {params / 1e9:.2f}B")

        print("\n\n" + "=" * 80 + "\n\n")

    except Exception as e:
        print(f"加载 Qwen-VL 架构失败: {e}")

print("模型架构审查完毕。没有加载任何模型权重。")