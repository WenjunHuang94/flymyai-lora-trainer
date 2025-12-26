import torch
from pipeline.qwen_multi_GPU import GlanceQwenSlowPipeline, GlanceQwenFastPipeline
from utils.distribute_free import distribute, free_pipe
from optimum.quanto import quantize, qfloat8, freeze
from tqdm import tqdm
import os

# 配置 LoRA 权重路径
# 如果本地有文件，请设置为你本地 LoRA 权重文件所在的目录路径
# 如果设置为 None 或文件不存在，将使用 Hugging Face 仓库（会触发下载）
lora_weights_dir = "./"  # 例如: "/path/to/your/lora/weights" 或 "./" 或 None

# 配置是否对 DIT 进行量化
quantize_dit = True  # 设置为 True 启用量化，False 禁用

# 如果指定了本地路径且文件存在，使用本地路径；否则使用 HF 仓库
if lora_weights_dir and os.path.exists(os.path.join(lora_weights_dir, "glance_qwen_slow.safetensors")):
    slow_pipe = GlanceQwenSlowPipeline.from_pretrained("/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/", torch_dtype=torch.bfloat16)
    slow_pipe.load_lora_weights(lora_weights_dir, weight_name="glance_qwen_slow.safetensors")
    print(f"✓ Loading LoRA weights from local path: {lora_weights_dir}")
else:
    repo = "CSU-JPG/Glance"
    slow_pipe = GlanceQwenSlowPipeline.from_pretrained("/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/", torch_dtype=torch.bfloat16)
    slow_pipe.load_lora_weights(repo, weight_name="glance_qwen_slow.safetensors")
    print(f"⚠ Loading LoRA weights from Hugging Face (will download): {repo}")

# 对 DIT 进行量化
if quantize_dit:
    print("开始对 slow_pipe 的 DIT 进行量化...")
    torch_dtype = torch.bfloat16
    # 使用 GPU 1 进行量化，避免占用 GPU 0（GPU 0 预留给 text_encoder 和 vae）
    device = "cuda:1"
    qwen_transformer = slow_pipe.transformer
    all_blocks = list(qwen_transformer.transformer_blocks)
    
    for block in tqdm(all_blocks, desc="量化 transformer blocks"):
        block.to(device, dtype=torch_dtype)
        quantize(block, weights=qfloat8)
        freeze(block)
        block.to('cpu')
        # 清理显存缓存，避免累积
        torch.cuda.empty_cache()
    
    qwen_transformer.to(device, dtype=torch_dtype)
    quantize(qwen_transformer, weights=qfloat8)
    freeze(qwen_transformer)
    # 量化完成后将 transformer 移回 CPU，让 distribute() 可以正常分配
    qwen_transformer.to('cpu')
    torch.cuda.empty_cache()
    print("✓ slow_pipe DIT 量化完成")

distribute(slow_pipe)

prompt = "Please create a photograph capturing a young woman showcasing a dynamic presence as she bicycles alongside a river during a hot summer day. Her long hair streams behind her as she pedals, dressed in snug tights and a vibrant yellow tank top, complemented by New Balance running shoes that highlight her lean, athletic build. She sports a small backpack and sunglasses resting confidently atop her head."
latents = slow_pipe(
    prompt=prompt,
    negative_prompt=" ",
    width=1024,
    height=1024,
    num_inference_steps=5,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(0), 
    output_type="latent"
).images[0].unsqueeze(0).detach().cpu()
free_pipe(slow_pipe)

if lora_weights_dir and os.path.exists(os.path.join(lora_weights_dir, "glance_qwen_fast.safetensors")):
    fast_pipe = GlanceQwenFastPipeline.from_pretrained("/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/", torch_dtype=torch.bfloat16)
    fast_pipe.load_lora_weights(lora_weights_dir, weight_name="glance_qwen_fast.safetensors")
    print(f"✓ Loading LoRA weights from local path: {lora_weights_dir}")
else:
    repo = "CSU-JPG/Glance"
    fast_pipe = GlanceQwenFastPipeline.from_pretrained("/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/", torch_dtype=torch.bfloat16)
    fast_pipe.load_lora_weights(repo, weight_name="glance_qwen_fast.safetensors")
    print(f"⚠ Loading LoRA weights from Hugging Face (will download): {repo}")

# 对 DIT 进行量化
if quantize_dit:
    print("开始对 fast_pipe 的 DIT 进行量化...")
    torch_dtype = torch.bfloat16
    # 使用 GPU 1 进行量化，避免占用 GPU 0（GPU 0 预留给 text_encoder 和 vae）
    device = "cuda:1"
    qwen_transformer = fast_pipe.transformer
    all_blocks = list(qwen_transformer.transformer_blocks)
    
    for block in tqdm(all_blocks, desc="量化 transformer blocks"):
        block.to(device, dtype=torch_dtype)
        quantize(block, weights=qfloat8)
        freeze(block)
        block.to('cpu')
        # 清理显存缓存，避免累积
        torch.cuda.empty_cache()
    
    qwen_transformer.to(device, dtype=torch_dtype)
    quantize(qwen_transformer, weights=qfloat8)
    freeze(qwen_transformer)
    # 量化完成后将 transformer 移回 CPU，让 distribute() 可以正常分配
    qwen_transformer.to('cpu')
    torch.cuda.empty_cache()
    print("✓ fast_pipe DIT 量化完成")

distribute(fast_pipe)

image = fast_pipe(
    prompt=prompt,
    negative_prompt=" ", 
    width=1024,
    height=1024,
    num_inference_steps=5, 
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(0), 
    latents=latents.to("cuda", dtype=torch.bfloat16)
).images[0]
image.save("output.png")

