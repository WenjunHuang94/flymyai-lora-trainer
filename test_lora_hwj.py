import time
from diffusers import DiffusionPipeline
import torch

# 记录总开始时间
total_start_time = time.time()

# --- 1. 加载基础 Qwen-Image 模型 ---
print("=" * 50)
print("正在加载基础模型 Qwen/Qwen-Image ...")
model_load_start = time.time()

model_name = "Qwen/Qwen-Image"
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)  # TODO: 也可以直接调用QwenImagePipeline
pipe = pipe.to(device)

# =======================================================
# 🔽 在这里加入打印代码 🔽
# =======================================================

print("\n\n" + "=" * 80 + "\n")
print("=" * 25 + " 1. VAE 结构 (pipe.vae) " + "=" * 25)
# pipe.vae 就是脚本1中的 AutoencoderKLQwenImage 实例
print(pipe.vae)

print("\n\n" + "=" * 80 + "\n")
print("=" * 20 + " 2. DiT 结构 (pipe.transformer) " + "=" * 20)
# pipe.transformer 就是脚本1中的 QwenImageTransformer2DModel 实例
print(pipe.transformer)

print("\n\n" + "=" * 80 + "\n")
print("=" * 20 + " 3. 文本编码器 结构 (pipe.text_encoder) " + "=" * 20)
# pipe.text_encoder 就是脚本1中的 Qwen2_5_VLForConditionalGeneration 实例
print(pipe.text_encoder)
print("\n\n" + "=" * 80 + "\n")

# =======================================================
# 🔼 打印代码结束 🔼
# =======================================================

model_load_time = time.time() - model_load_start
print(f"基础模型加载完毕。耗时: {model_load_time:.2f} 秒")
print("=" * 50)

# --- 2. 加载您训练好的 LoRA 权重 ---
lora_load_start = time.time()
# 【修改点 1】: 路径已更新为您自己的 checkpoint 路径
lora_file_path = "/home/disk2/hwj/flymyai-lora-trainer/output/checkpoint-250/pytorch_lora_weights.safetensors"
# lora_file_path = "/home/disk2/hwj/flymyai-lora-trainer/qwen-image-realism-lora/flymy_realism.safetensors"

try:
    print(f"正在加载您训练的 LoRA 文件: {lora_file_path} ...")
    # 【修改点 2】: 取消了注释，并更改了 adapter_name
    pipe.load_lora_weights(lora_file_path, adapter_name="hwj")  # <--- 已取消注释
    print("设置 adapter_name 为 'hwj'")

    lora_load_time = time.time() - lora_load_start
    print(f"LoRA 加载成功！耗时: {lora_load_time:.2f} 秒")
except Exception as e:
    print(f"加载 LoRA 失败: {e}")
    print("请确保 lora_file_path 路径正确。")
    exit()

print("=" * 50)

# --- 3. 准备提示词 (Prompt) ---
# 【修改点 3】: 使用您的触发词和示例提示
prompt = '''ohwhwj man, wearing a red helmet and orange life jacket, smiling with water droplets on his face.'''

positive_magic = ", Ultra HD, 4K, cinematic composition."
negative_prompt = " "

# --- 4. 生成图像 ---
print("正在生成图像 (使用 'ohwhwj man' LoRA)...")
generation_start = time.time()

image = pipe(
    prompt=prompt + positive_magic,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=50,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(42)  # 您可以换个 seed 看看新效果, 比如 12345
).images[0]

generation_time = time.time() - generation_start
print(f"图像生成完成！耗时: {generation_time:.2f} 秒")
print("=" * 50)

# --- 5. 保存图像 ---
save_start = time.time()
# 【修改点 4】: 更改了输出文件名
output_filename = "output_hwj_checkpoint1000-9.png"
image.save(output_filename)
save_time = time.time() - save_start

print(f"图像已成功保存为: {output_filename}")
print(f"保存耗时: {save_time:.2f} 秒")

# --- 6. 统计总耗时 ---
total_time = time.time() - total_start_time
print("=" * 50)
print("=== 时间统计汇总 ===")
print(f"基础模型加载: {model_load_time:.2f} 秒")
print(f"LoRA 权重加载: {lora_load_time:.2f} 秒")
print(f"图像生成: {generation_time:.2f} 秒")
print(f"图像保存: {save_time:.2f} 秒")
print("-" * 30)
print(f"总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
print("=" * 50)