# inference_small_dit.py
import torch
from diffusers import QwenImagePipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
import os

# --- 1. 定义您的模型路径 ---
# 原始模型的（本地）路径 (或者 "Qwen/Qwen-Image")
ORIGINAL_MODEL_PATH = "Qwen/Qwen-Image"
# 您新创建的小型 DiT 的路径
# SMALL_DIT_PATH = "./my_small_dit_1_7B"
SMALL_DIT_PATH = "./output_medium_dit_training/checkpoint-20/transformer"

# --- 2. 分别加载所有组件 ---
print(f"Loading VAE from: {ORIGINAL_MODEL_PATH}")
vae = AutoencoderKLQwenImage.from_pretrained(
    ORIGINAL_MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16
)

print('vae = ', vae)

print(f"Loading Text Encoder from: {ORIGINAL_MODEL_PATH}")
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ORIGINAL_MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16
)

print('text_encoder = ', text_encoder)

print(f"Loading Tokenizer from: {ORIGINAL_MODEL_PATH}")
tokenizer = Qwen2Tokenizer.from_pretrained(
    ORIGINAL_MODEL_PATH, subfolder="tokenizer"
)

print('tokenizer = ', tokenizer)

print(f"Loading SMALL Transformer from: {SMALL_DIT_PATH}")
# 【关键】从您的新目录加载 Transformer
transformer = QwenImageTransformer2DModel.from_pretrained(
    SMALL_DIT_PATH, torch_dtype=torch.bfloat16
)

print('transformer = ', transformer)

# --- 3. 组装 Pipeline ---
print("Assembling pipeline...")
# 我们借用 QwenImagePipeline 的“空壳”，并手动填入我们的组件
pipe = QwenImagePipeline.from_pretrained(
    ORIGINAL_MODEL_PATH,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

pipe = pipe.to("cuda")

# --- 4. 运行推理测试 ---
prompt = "A majestic lion in the savanna"
negative_prompt = " "

print(f"Running inference for prompt: '{prompt}'")
generator = torch.Generator(device="cuda").manual_seed(42)

try:
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=30,
        true_cfg_scale=5,
        generator=generator
    ).images[0]

    # --- 5. 保存结果 ---
    output_path = "small_dit_inference_test.png"
    image.save(output_path)
    print("="*50)
    print(f"✅ 推理测试成功！")
    print(f"✅ 已保存测试图像到: {output_path}")
    print("✅ (图像是噪声或乱码是【正常】的!)")
    print("="*50)

except Exception as e:
    print("="*50)
    print(f"❌ 推理测试失败: {e}")
    print("="*50)