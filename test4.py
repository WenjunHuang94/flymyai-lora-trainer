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
model_path = "Qwen/Qwen-Image"
# model_path = "Qwen/Qwen-Image-Edit"  # <--- 【【请修改成这个】】

print(f"正在从 {model_path} 加载配置, 并使用 'meta' device 快速实例化架构...")
print("=" * 60 + "\n")

# 记录总参数
total_params = 0
model_components = []

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

        # 【【【 关键修改：使用 .from_config() 从配置实例化模型 】】】
        vae_model = AutoencoderKLQwenImage.from_config(vae_config)
        print(f"VAE 架构实例化完毕 (耗时: {time.time() - load_start:.4f}s)\n")

        # 计算参数量
        params = sum(p.numel() for p in vae_model.parameters())
        total_params += params

        # 估算显存占用
        # 假设使用 fp16 (float16)，每个参数 2 字节
        mem_fp16 = params * 2 / (1024 ** 3)  # 转换为 GB
        # 假设使用 fp32 (float32)，每个参数 4 字节
        mem_fp32 = params * 4 / (1024 ** 3)  # 转换为 GB
        # 假设使用 bf16 (bfloat16)，每个参数 2 字节
        mem_bf16 = params * 2 / (1024 ** 3)  # 转换为 GB

        print(f"VAE 参数量: {params:,}")
        print(f"  ≈ {params / 1e6:.2f}M 参数")
        print(f"  ≈ {params / 1e9:.2f}B 参数")
        print("\n显存占用估算 (仅模型权重):")
        print(f"  - FP16: {mem_fp16:.2f} GB")
        print(f"  - BF16: {mem_bf16:.2f} GB")
        print(f"  - FP32: {mem_fp32:.2f} GB")
        print("\n实际训练时估算 (AdamW优化器 + 梯度):")
        print(f"  - FP16混合精度: {mem_fp16 * 4:.2f} GB  (参数+梯度+动量+方差)")
        print(f"  - FP32训练: {mem_fp32 * 4:.2f} GB  (参数+梯度+动量+方差)")

        model_components.append(("VAE", params, mem_fp16, mem_fp32))

        print("\n" + "=" * 80 + "\n")

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

        # 实例化模型
        dit_model = QwenImageTransformer2DModel.from_config(dit_config)
        print(f"DiT 架构实例化完毕 (耗时: {time.time() - load_start:.4f}s)\n")

        # 计算参数量
        params = sum(p.numel() for p in dit_model.parameters())
        total_params += params

        # 估算显存占用
        mem_fp16 = params * 2 / (1024 ** 3)
        mem_fp32 = params * 4 / (1024 ** 3)
        mem_bf16 = params * 2 / (1024 ** 3)

        print(f"DiT 参数量: {params:,}")
        print(f"  ≈ {params / 1e6:.2f}M 参数")
        print(f"  ≈ {params / 1e9:.2f}B 参数")
        print("\n显存占用估算 (仅模型权重):")
        print(f"  - FP16: {mem_fp16:.2f} GB")
        print(f"  - BF16: {mem_bf16:.2f} GB")
        print(f"  - FP32: {mem_fp32:.2f} GB")
        print("\n实际训练时估算 (AdamW优化器 + 梯度):")
        print(f"  - FP16混合精度: {mem_fp16 * 4:.2f} GB  (参数+梯度+动量+方差)")
        print(f"  - FP32训练: {mem_fp32 * 4:.2f} GB  (参数+梯度+动量+方差)")

        model_components.append(("DiT", params, mem_fp16, mem_fp32))

        print("\n" + "=" * 80 + "\n")

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

        # 实例化模型
        from transformers import AutoModel

        vl_model = AutoModel.from_config(vl_config)
        print(f"Qwen-VL (Text Encoder) 架构实例化完毕 (耗时: {time.time() - load_start:.2f}s)\n")

        # 计算参数量
        params = sum(p.numel() for p in vl_model.parameters())
        total_params += params

        # 估算显存占用
        mem_fp16 = params * 2 / (1024 ** 3)
        mem_fp32 = params * 4 / (1024 ** 3)
        mem_bf16 = params * 2 / (1024 ** 3)

        print(f"Qwen-VL 参数量: {params:,}")
        print(f"  ≈ {params / 1e6:.2f}M 参数")
        print(f"  ≈ {params / 1e9:.2f}B 参数")
        print("\n显存占用估算 (仅模型权重):")
        print(f"  - FP16: {mem_fp16:.2f} GB")
        print(f"  - BF16: {mem_bf16:.2f} GB")
        print(f"  - FP32: {mem_fp32:.2f} GB")
        print("\n实际训练时估算 (AdamW优化器 + 梯度):")
        print(f"  - FP16混合精度: {mem_fp16 * 4:.2f} GB  (参数+梯度+动量+方差)")
        print(f"  - FP32训练: {mem_fp32 * 4:.2f} GB  (参数+梯度+动量+方差)")

        model_components.append(("Qwen-VL", params, mem_fp16, mem_fp32))

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"加载 Qwen-VL 架构失败: {e}")

# 打印汇总信息
print("\n" + "=" * 60)
print("模型架构审查完成 - 显存占用总结")
print("=" * 60)

print(f"\n总参数量: {total_params:,}")
print(f"  ≈ {total_params / 1e6:.2f}M 参数")
print(f"  ≈ {total_params / 1e9:.2f}B 参数")

# 计算总计显存
total_fp16 = sum(mem[2] for mem in model_components)  # fp16权重
total_fp32 = sum(mem[3] for mem in model_components)  # fp32权重

print("\n" + "=" * 60)
print("显存占用估算总结:")
print("=" * 60)

print("\n🔵 推理模式 (仅加载权重):")
print(f"  - FP16精度: {total_fp16:.2f} GB")
print(f"  - BF16精度: {total_fp16:.2f} GB")
print(f"  - FP32精度: {total_fp32:.2f} GB")

print("\n🟢 训练模式 (AdamW优化器, 需要存储梯度+动量+方差):")
print(f"  - FP16混合精度: {total_fp16 * 4:.2f} GB  (参数×4)")
print(f"  - FP32训练: {total_fp32 * 4:.2f} GB  (参数×4)")

print("\n⚠️  注意: 以上估算仅包含模型参数本身。实际使用时还需要考虑:")
print("  - 激活值 (activations) 的显存占用")
print("  - 输入/输出张量的显存占用")
print("  - 批次大小 (batch size) 的影响")
print("  - 梯度检查点 (gradient checkpointing) 可减少激活值占用")
print("  - CUDA上下文和其他系统开销")

print("\n📊 各组件详细统计:")
for name, params, mem_fp16, mem_fp32 in model_components:
    print(f"  - {name}: {params / 1e9:.2f}B 参数, FP16: {mem_fp16:.2f}GB, FP32: {mem_fp32:.2f}GB")

print(f"\n✅ 模型架构审查完毕。总计 {total_params / 1e9:.2f}B 参数。")
print("   没有加载任何模型权重，显存占用为 0 MB (meta device)。")
print(f"   如果加载权重，估计需要:")
print(f"     - 推理: {total_fp16:.1f}-{total_fp32:.1f} GB 显存")
print(f"     - 训练: {total_fp16 * 4:.1f}-{total_fp32 * 4:.1f} GB 显存")