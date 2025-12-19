import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
import os

print("=" * 50)
print("正在加载 Qwen-Image-Edit 基础模型...")

# --- 1. 加载基础模型 ---
# 我们必须使用 LoRA 训练时所用的同一个基础模型
base_model_path = "Qwen/Qwen-Image-Edit"
pipe = QwenImageEditPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("基础模型加载完毕。")
print("=" * 50)

# --- 2. 加载您训练好的 LoRA 权重 ---

# 【【【 请修改这里 (1/3) 】】】
# 把它改成您想测试的 checkpoint 路径
lora_checkpoint_path = "/home/disk2/hwj/flymyai-lora-trainer/test_lora_saves_edit/checkpoint-20/pytorch_lora_weights.safetensors"

try:
    print(f"正在加载您的 LoRA: {lora_checkpoint_path}")
    pipe.load_lora_weights(lora_checkpoint_path, adapter_name="my_edit_lora")
    print("LoRA 加载成功！")
except Exception as e:
    print(f"加载 LoRA 失败: {e}")
    print("请确保 lora_checkpoint_path 路径正确。")
    exit()
print("=" * 50)

# --- 3. 准备输入 ---

# 【【【 请修改这里 (2/3) 】】】
# 必须是您的“控制图”（原始头像）
CONTROL_IMAGE_PATH = "dataset/control_images/001.jpg"

try:
    control_image = Image.open(CONTROL_IMAGE_PATH).convert("RGB")
    print(f"已加载控制图: {CONTROL_IMAGE_PATH}")
except FileNotFoundError:
    print(f"错误: 找不到控制图 {CONTROL_IMAGE_PATH}")
    exit()

# 【【【 请修改这里 (3/3) - 这是最重要的测试！ 】】】
# 您的指令，必须包含您的触发词 "ohwhwj man"
#
# **建议您使用一个“全新的”、“训练时没用过”的指令**
# 比如："ohwhwj man, turn his hair to bright green."
# 或者："ohwhwj man, make him wear a golden necklace."
TEST_PROMPT = "ohwhwj man, turn his hair to bright yellow. in the cinema."

print(f"使用指令: {TEST_PROMPT}")
print("=" * 50)

# --- 4. 运行“图生图”推理 ---
print("正在执行图像编辑... 请稍候...")

# 为 LoRA 设置参数 (和您训练时用的 seed 不同，以便测试泛化性)
inputs = {
    "image": control_image,       # <--- 输入1: 控制图
    "prompt": TEST_PROMPT,        # <--- 输入2: 指令
    "generator": torch.manual_seed(12345), # 换个新 seed
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipe(**inputs)
    output_image = output.images[0]

print("图像编辑完成！")

# --- 5. 保存结果 ---
output_filename = "TEST_EDIT_LORA_RESULT3.png"
output_image.save(output_filename)
print(f"测试结果已保存为: {os.path.abspath(output_filename)}")
print("=" * 50)