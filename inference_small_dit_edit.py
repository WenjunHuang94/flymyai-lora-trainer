# inference_small_dit_edit.py
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
import os

# --- 1. 定义您的模型路径 ---
# 原始 Edit 模型的路径
ORIGINAL_MODEL_PATH = "Qwen/Qwen-Image-Edit"


# 您新创建的小型 DiT 的路径
# SMALL_DIT_PATH = "./my_small_dit_1_7B"
# 【注意】如果您训练了第3步，可以把它改成 checkpoint 路径来测试训练结果
SMALL_DIT_PATH = "./output_small_dit_training/final_model/transformer"

# --- 2. 分别加载所有组件 ---
print(f"Loading VAE from: {ORIGINAL_MODEL_PATH}")
vae = AutoencoderKLQwenImage.from_pretrained(
    ORIGINAL_MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16
)

print(f"Loading Text Encoder, Tokenizer, and Processor from: {ORIGINAL_MODEL_PATH}")
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ORIGINAL_MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16
)
tokenizer = Qwen2Tokenizer.from_pretrained(
    ORIGINAL_MODEL_PATH, subfolder="tokenizer"
)
processor = Qwen2VLProcessor.from_pretrained(
    ORIGINAL_MODEL_PATH, subfolder="processor"
)

print(f"Loading SMALL Transformer from: {SMALL_DIT_PATH}")
# 【关键】从您的新目录加载 Transformer
transformer = QwenImageTransformer2DModel.from_pretrained(
    SMALL_DIT_PATH, torch_dtype=torch.bfloat16
)

# --- 3. 组装 Pipeline ---
print("Assembling QwenImageEditPipeline...")
pipe = QwenImageEditPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(ORIGINAL_MODEL_PATH, subfolder="scheduler")
)

pipe = pipe.to("cuda")

# --- 4. 准备输入 ---
try:
    # 【请修改】改成您自己的控制图路径
    CONTROL_IMAGE_PATH = "dataset/control_images/001.jpg"
    control_image = Image.open(CONTROL_IMAGE_PATH).convert("RGB")
    print(f"Loaded control image from: {CONTROL_IMAGE_PATH}")
except FileNotFoundError:
    print(f"错误: 找不到控制图 {CONTROL_IMAGE_PATH}")
    print("请创建一个 512x512 的随机图像作为测试...")
    control_image = Image.new("RGB", (512, 512), "gray")

# 【请修改】测试指令
TEST_PROMPT = "turn him into a superman"

print(f"Running inference for prompt: '{TEST_PROMPT}'")
generator = torch.manual_seed(12345)

# --- 5. 运行推理测试 ---
try:
    inputs = {
        "image": control_image,
        "prompt": TEST_PROMPT,
        "generator": generator,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 30,
    }

    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]

    # --- 6. 保存结果 ---
    output_path = "small_dit_edit_inference_test.png"
    output_image.save(output_path)
    print("=" * 50)
    print(f"✅ (Edit) 推理测试成功！")
    print(f"✅ 已保存测试图像到: {output_path}")
    print("✅ (如果DIT未训练，图像是噪声或乱码是【正常】的!)")
    print("=" * 50)

except Exception as e:
    print("=" * 50)
    print(f"❌ (Edit) 推理测试失败: {e}")
    print("=" * 50)