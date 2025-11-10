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

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

model_load_time = time.time() - model_load_start
print(f"基础模型加载完毕。耗时: {model_load_time:.2f} 秒")
print("=" * 50)

# --- 2. 加载你下载的 LoRA 权重 ---
lora_load_start = time.time()
lora_file_path = "/home/disk2/hwj/flymyai-lora-trainer/qwen-image-realism-lora/flymy_realism.safetensors"

try:
    print(f"正在加载 LoRA 文件: {lora_file_path} ...")
    # pipe.load_lora_weights(lora_file_path, adapter_name="realism")
    lora_load_time = time.time() - lora_load_start
    print(f"LoRA 加载成功！耗时: {lora_load_time:.2f} 秒")
except Exception as e:
    print(f"加载 LoRA 失败: {e}")
    print("请确保 lora_file_path 路径正确。")
    exit()

print("=" * 50)

# --- 3. 准备提示词 (Prompt) ---
prompt = '''Super Realism portrait of a teenager woman of African descent, serene calmness, arms crossed, illuminated by dramatic studio lighting, sunlit park in the background, adorned with delicate jewelry, three-quarter view, sun-kissed skin with natural imperfections, loose shoulder-length curls, slightly squinting eyes, environmental street portrait with text "FLYMY AI" on t-shirt.'''

positive_magic = ", Ultra HD, 4K, cinematic composition."
negative_prompt = " "

# --- 4. 生成图像 ---
print("正在生成图像，请稍候...")
generation_start = time.time()

image = pipe(
    prompt=prompt + positive_magic,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(346346)
).images[0]

generation_time = time.time() - generation_start
print(f"图像生成完成！耗时: {generation_time:.2f} 秒")
print("=" * 50)

# --- 5. 保存图像 ---
save_start = time.time()
output_filename = "output_with_lora.png"
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
print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
print("=" * 50)