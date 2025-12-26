# Hugging Face 下载位置汇总

本文档列出了代码中所有可能从 Hugging Face 自动下载权重的位置。

## 1. 训练脚本 (train_Glance_qwen.py)

**配置文件**: `train_configs/Glance_qwen.yaml`
- **第1行**: `pretrained_model_name_or_path: Qwen/Qwen-Image`
  - ⚠️ **会从 HF 下载**: 如果这个值是模型 ID（如 "Qwen/Qwen-Image"）而不是本地路径

**代码中的使用位置**:
1. **第140-142行**: `QwenImagePipeline.from_pretrained(args.pretrained_model_name_or_path, ...)`
   - 下载: text_encoder 相关组件
   
2. **第192-194行**: `AutoencoderKLQwenImage.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")`
   - 下载: VAE 模型权重
   
3. **第231-233行**: `QwenImageTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")`
   - 下载: Transformer 模型权重
   
4. **第255-257行**: `FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")`
   - 下载: Scheduler 配置

**解决方案**: 将配置文件中的 `pretrained_model_name_or_path` 改为本地路径，例如：
```yaml
pretrained_model_name_or_path: /storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/
```

---

## 2. 推理脚本 (infer_Glance_qwen_multi_GPU.py)

**第18行**: `slow_pipe.load_lora_weights(repo, weight_name="glance_qwen_slow.safetensors")`
- ⚠️ **会从 HF 下载**: `repo = "CSU-JPG/Glance"` 是 Hugging Face 仓库 ID
- 下载文件: `glance_qwen_slow.safetensors`

**第35行**: `fast_pipe.load_lora_weights(repo, weight_name="glance_qwen_fast.safetensors")`
- ⚠️ **会从 HF 下载**: 同样使用 `repo = "CSU-JPG/Glance"`
- 下载文件: `glance_qwen_fast.safetensors`

**解决方案**: 需要手动下载这两个 LoRA 权重文件，然后修改代码使用本地路径：
1. 下载 `https://huggingface.co/CSU-JPG/Glance/resolve/main/glance_qwen_slow.safetensors`
2. 下载 `https://huggingface.co/CSU-JPG/Glance/resolve/main/glance_qwen_fast.safetensors`
3. 修改代码使用本地路径加载

---

## 3. 推理脚本 (infer_Glance_qwen.py)

**第7行**: `slow_pipe.load_lora_weights(repo, weight_name="glance_qwen_slow.safetensors")`
- ⚠️ **会从 HF 下载**: `repo = "CSU-JPG/Glance"`

**第24行**: `fast_pipe.load_lora_weights(repo, weight_name="glance_qwen_fast.safetensors")`
- ⚠️ **会从 HF 下载**: 同上

---

## 4. Flux 训练脚本 (train_Glance_flux.py)

**配置文件**: `train_configs/Glance_flux.yaml`
- **第1行**: `pretrained_model_name_or_path: black-forest-labs/FLUX.1-dev`
  - ⚠️ **会从 HF 下载**: 如果使用模型 ID

**代码中的使用位置**:
1. **第104-105行**: `FluxPipeline.from_pretrained(args.pretrained_model_name_or_path, ...)`
2. **第107-109行**: `AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")`
3. **第111-112行**: `FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")`
4. **第122-124行**: `FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")`

---

## 需要手动下载的文件清单

### Qwen-Image 模型 (用于训练)
从 `https://huggingface.co/Qwen/Qwen-Image` 下载：
- 整个模型仓库（包括所有子文件夹：vae, transformer, scheduler, text_encoder 等）
- 或者使用 `huggingface-cli download Qwen/Qwen-Image --local-dir /your/local/path`

### Glance LoRA 权重 (用于推理)
从 `https://huggingface.co/CSU-JPG/Glance` 下载：
- `glance_qwen_slow.safetensors`
- `glance_qwen_fast.safetensors`

### FLUX 模型 (如果使用 Flux 训练)
从 `https://huggingface.co/black-forest-labs/FLUX.1-dev` 下载：
- 整个模型仓库

---

## 修改建议

### 1. 修改训练配置文件
编辑 `train_configs/Glance_qwen.yaml`:
```yaml
pretrained_model_name_or_path: /storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image/
```

### 2. 修改推理脚本中的 LoRA 加载
将 `infer_Glance_qwen_multi_GPU.py` 和 `infer_Glance_qwen.py` 中的：
```python
repo = "CSU-JPG/Glance"
slow_pipe.load_lora_weights(repo, weight_name="glance_qwen_slow.safetensors")
```
改为使用本地路径（需要查看 diffusers 的 load_lora_weights 是否支持本地路径，或使用其他加载方法）

