import torch
from pipeline.qwen_multi_GPU import GlanceQwenSlowPipeline
from utils.distribute_free import distribute, free_pipe
import os

# 配置 LoRA 权重路径
# 如果本地有文件，请设置为你本地 LoRA 权重文件所在的目录路径
# 如果设置为 None 或文件不存在，将使用 Hugging Face 仓库（会触发下载）
lora_weights_dir = "./"  # 例如: "/path/to/your/lora/weights" 或 "./" 或 None

# 定义 slow 和 fast 的 timesteps
slow_timesteps = torch.tensor([
    1000.0000, 979.1915, 957.5157, 934.9171, 911.3354
], dtype=torch.bfloat16)

fast_timesteps = torch.tensor([
    886.7053, 745.0728, 562.9505, 320.0802, 20.0000
], dtype=torch.bfloat16)

# 只加载一次 pipeline（使用 slow pipeline 作为基础）
print("加载 Qwen-Image 模型（仅加载一次）...")
if lora_weights_dir and os.path.exists(os.path.join(lora_weights_dir, "glance_qwen_slow.safetensors")):
    pipe = GlanceQwenSlowPipeline.from_pretrained(
        "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/", 
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(lora_weights_dir, weight_name="glance_qwen_slow.safetensors")
    print(f"✓ 加载 Slow LoRA 权重从本地路径: {lora_weights_dir}")
else:
    repo = "CSU-JPG/Glance"
    pipe = GlanceQwenSlowPipeline.from_pretrained(
        "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/", 
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(repo, weight_name="glance_qwen_slow.safetensors")
    print(f"⚠ 加载 Slow LoRA 权重从 Hugging Face (将下载): {repo}")

# 分配模型到多个 GPU
distribute(pipe)

# 第一次推理：使用 slow timesteps 和 slow LoRA
prompt = "Please create a photograph capturing a young woman showcasing a dynamic presence as she bicycles alongside a river during a hot summer day. Her long hair streams behind her as she pedals, dressed in snug tights and a vibrant yellow tank top, complemented by New Balance running shoes that highlight her lean, athletic build. She sports a small backpack and sunglasses resting confidently atop her head."

print("\n=== 第一次推理：Slow 阶段 ===")
latents = pipe(
    prompt=prompt,
    negative_prompt=" ",
    width=1024,
    height=1024,
    num_inference_steps=5,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(0), 
    output_type="latent",
    custom_timesteps=slow_timesteps  # 使用 slow timesteps
).images[0].unsqueeze(0).detach().cpu()

print("✓ Slow 阶段完成，latents 形状:", latents.shape)

# 切换 LoRA 权重：卸载 slow LoRA，加载 fast LoRA
print("\n=== 切换 LoRA 权重：Slow -> Fast ===")
try:
    # 尝试卸载 LoRA（如果支持）
    if hasattr(pipe, 'unload_lora_weights'):
        pipe.unload_lora_weights()
        print("✓ 已卸载 Slow LoRA 权重")
except:
    # 如果不支持卸载，直接加载新的 LoRA（会自动替换）
    print("⚠ 不支持显式卸载，直接加载 Fast LoRA（将自动替换）")

# 加载 fast LoRA 权重
if lora_weights_dir and os.path.exists(os.path.join(lora_weights_dir, "glance_qwen_fast.safetensors")):
    pipe.load_lora_weights(lora_weights_dir, weight_name="glance_qwen_fast.safetensors")
    print(f"✓ 已加载 Fast LoRA 权重从本地路径: {lora_weights_dir}")
else:
    repo = "CSU-JPG/Glance"
    pipe.load_lora_weights(repo, weight_name="glance_qwen_fast.safetensors")
    print(f"✓ 已加载 Fast LoRA 权重从 Hugging Face: {repo}")

# 第二次推理：使用 fast timesteps 和 fast LoRA
print("\n=== 第二次推理：Fast 阶段 ===")
image = pipe(
    prompt=prompt,
    negative_prompt=" ", 
    width=1024,
    height=1024,
    num_inference_steps=5, 
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(0), 
    latents=latents.to("cuda", dtype=torch.bfloat16),
    custom_timesteps=fast_timesteps  # 使用 fast timesteps
).images[0]

print("✓ Fast 阶段完成")
image.save("output.png")
print("✓ 图像已保存到 output.png")

# 清理资源
free_pipe(pipe)
print("\n✓ 推理完成，资源已释放")
