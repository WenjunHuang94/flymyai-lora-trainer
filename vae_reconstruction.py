import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor

# 1. 设置路径
IMAGE_PATH = "/home/disk2/hwj/flymyai-lora-trainer/my_training_data2/001.jpg"  # 你的图片路径
MODEL_PATH = "Qwen/Qwen-Image"  # 你的模型路径

print(f"正在加载 VAE 从: {MODEL_PATH}")
# 加载 VAE
vae = AutoencoderKLQwenImage.from_pretrained(
    MODEL_PATH,
    subfolder="vae"
).to("cuda", dtype=torch.bfloat16)

# 2. 读取并预处理原图
print(f"正在读取图片: {IMAGE_PATH}")
try:
    img = Image.open(IMAGE_PATH).convert("RGB")
    img = img.resize((1024, 1024))  # 强制缩放
except Exception as e:
    print(f"错误: 无法读取图片. {e}")
    exit()

# 3. 编码 -> 解码
print("正在执行 VAE 编码与解码...")
with torch.no_grad():
    # --- 预处理 ---
    # 1. 转 Tensor: (H, W, C)
    pixel_values = torch.tensor(np.array(img))
    # 2. Permute: (C, H, W)
    pixel_values = pixel_values.permute(2, 0, 1)
    # 3. Batch 维度: (1, C, H, W)
    pixel_values = pixel_values.unsqueeze(0)
    # 4. 【关键修复】增加 Frame/Time 维度: (1, C, 1, H, W)
    pixel_values = pixel_values.unsqueeze(2)

    # 5. 归一化并送入 GPU
    pixel_values = pixel_values.to("cuda", dtype=torch.bfloat16) / 127.5 - 1.0

    print(f"输入 Tensor 形状: {pixel_values.shape}")  # 应该是 [1, 3, 1, 1024, 1024]

    # --- 编码 (Encoding) ---
    # 此时 pixel_values 是 5D 的，encode 不会报错了
    latents = vae.encode(pixel_values).latent_dist.sample()

    # --- 解码 (Decoding) ---
    # 解码出来的 decoded 也是 5D 的: [1, 3, 1, 1024, 1024]
    decoded = vae.decode(latents).sample

    # 【关键修复】去掉 Frame 维度，变回 4D 图片: [1, 3, 1024, 1024]
    # 我们取第 0 帧
    decoded = decoded[:, :, 0, :, :]

# 4. 后处理并保存
print("正在保存重建结果...")
processor = VaeImageProcessor()
# postprocess 期望的是 4D 数据
recon_image = processor.postprocess(decoded, output_type="pil")[0]

save_path = "vae_reconstruction.png"
recon_image.save(save_path)
print(f"✅ VAE 重建完成！结果已保存至: {save_path}")
print("请对比 '原图' 和 'vae_reconstruction.png' 的清晰度。")