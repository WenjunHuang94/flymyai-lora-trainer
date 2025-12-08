# inference_overfit.py
import torch
from diffusers import QwenImagePipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
import os

# --- 配置 ---
ORIGINAL_MODEL = "Qwen/Qwen-Image"

# 指向刚才跑出来的 final_model
# TRAINED_DIT_PATH = "./output_overfit_final_fp32/checkpoint-1400/transformer"
TRAINED_DIT_PATH = "./output_overfit_high_precision_resume2/checkpoint-best/transformer"


# 从路径中提取文件夹名和checkpoint名称用于保存图片
path_parts = TRAINED_DIT_PATH.strip("./").split("/")
folder_name = path_parts[0]  # 提取 "output_overfit_final_fp32"
checkpoint_name = path_parts[1]  # 提取 "checkpoint-3000"
image_name = f"{folder_name}_{checkpoint_name}.png"  # 组合成 "output_overfit_final_fp32_checkpoint-3000.png"

# 【重要】这里必须填入你 my_training_data2 里面那个 txt 文件的完整内容！
PROMPT = "a man, wearing a pink t-shirt, looking directly at the camera, indoors with fluorescent lighting."

print("1. Loading Base Components...")

# 【修改点 1】加载 VAE 时加上 local_files_only=True
vae = AutoencoderKLQwenImage.from_pretrained(
    ORIGINAL_MODEL,
    subfolder="vae",
    torch_dtype=torch.bfloat16,
    local_files_only=True  # <--- 强制只读本地缓存
)

# 【修改点 2】加载 Text Encoder 时加上 local_files_only=True
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ORIGINAL_MODEL,
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
    local_files_only=True  # <--- 强制只读本地缓存
)

# 【修改点 3】加载 Tokenizer 时加上 local_files_only=True
tokenizer = Qwen2Tokenizer.from_pretrained(
    ORIGINAL_MODEL,
    subfolder="tokenizer",
    local_files_only=True  # <--- 强制只读本地缓存
)

print(f"2. Loading Overfitted Small DiT from {TRAINED_DIT_PATH}...")

# 【修改点 4】加载你的 Small DiT 时也加上，防止路径错误时去联网瞎找
transformer = QwenImageTransformer2DModel.from_pretrained(
    TRAINED_DIT_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True  # <--- 加上这个，如果路径不对它会直接报"找不到文件"而不是"网络错误"
)

# 组装 Pipeline (这里其实不用加，因为组件都齐了，但加了更保险)
pipe = QwenImagePipeline.from_pretrained(
    ORIGINAL_MODEL,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    local_files_only=True # <--- 为了加载 pipeline config，建议也加上
).to("cuda")

print(f"3. Generating image for prompt: '{PROMPT}'")
# 设置一个固定种子，看看能不能还原训练图
# 理论上如果 Loss 很低，无论 Seed 是多少，生成的主体都应该非常像原图
image = pipe(
    prompt=PROMPT,
    width=1024,
    height=1024,
    num_inference_steps=30, # 步数不需要太多
    true_cfg_scale=1.0,     # 【关键】过拟合测试建议 CFG=1.0 (不使用 guidance)，因为我们没有训练 negative prompt
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save(image_name)
print(f"✅ Saved as {image_name}")