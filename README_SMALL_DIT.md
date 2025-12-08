# 如何缩小 Qwen-Image 的 DIT 模型

## 概述

如果您想要训练一个更小的 DIT（Diffusion Transformer）模型以减少显存占用和训练时间，有以下几种方法：

## 方法对比

### 方法 1: 减少层数（推荐）
- **优点**: 最简单直接，效果明显
- **缺点**: 可能影响模型容量
- **示例**: 从 24 层减到 12 层，参数量减少约 50%

### 方法 2: 减少隐藏层维度
- **优点**: 参数量减少最明显（平方关系）
- **缺点**: 需要同时调整 attention_heads
- **示例**: 从 2048 减到 1024，参数量减少约 75%

### 方法 3: 减少注意力头数
- **优点**: 减少计算量
- **缺点**: 可能影响模型表达能力
- **注意**: 必须能被 hidden_size 整除

### 方法 4: 组合使用
- 同时减少层数和维度，可以获得最大的参数量减少

## 使用方法

### 方法 A: 通过配置文件（推荐）

在您的训练配置 YAML 文件中添加 `dit_config` 部分：

```yaml
pretrained_model_name_or_path: Qwen/Qwen-Image

# 缩小 DIT 模型配置
dit_config:
  num_layers: 12        # 减少层数
  hidden_size: 1024    # 减少隐藏层维度（可选）
  # num_attention_heads: 16  # 减少注意力头数（可选）
  # intermediate_size: 4096  # 减少 FFN 大小（可选）

# ... 其他训练配置
```

然后正常运行训练：
```bash
python train.py --config your_config.yaml
```

### 方法 B: 使用辅助脚本创建模型

首先使用 `create_smaller_dit.py` 创建并保存更小的模型：

```python
from create_smaller_dit import create_smaller_dit_model

# 创建并保存更小的模型
model, config = create_smaller_dit_model(
    pretrained_model_path="Qwen/Qwen-Image",
    output_path="./small_dit_model",
    num_layers=12,
    hidden_size=1024,
)
```

然后在训练配置中指向这个模型路径。

## 配置示例

### 示例 1: 只减少层数（减少 50% 参数量）
```yaml
dit_config:
  num_layers: 12
```

### 示例 2: 只减少维度（减少 75% 参数量）
```yaml
dit_config:
  hidden_size: 1024
```

### 示例 3: 同时减少层数和维度（减少 87.5% 参数量）
```yaml
dit_config:
  num_layers: 12
  hidden_size: 1024
```

## 注意事项

1. **权重初始化**: 新模型会尝试从原始模型复制兼容的权重，不兼容的部分会使用随机初始化
2. **性能影响**: 缩小模型可能会影响生成质量，建议先小规模测试
3. **兼容性**: 确保修改后的配置参数之间是兼容的（如 attention_heads 必须能被 hidden_size 整除）
4. **训练策略**: 对于大幅缩小的模型，可能需要：
   - 调整学习率
   - 增加训练步数
   - 使用知识蒸馏等技术

## 检查原始模型配置

在修改之前，您可以先检查原始模型的配置：

```python
from diffusers import QwenImageTransformer2DModel

model = QwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image",
    subfolder="transformer",
)

print("模型配置:")
print(f"  num_layers: {model.config.num_layers}")
print(f"  hidden_size: {model.config.hidden_size}")
print(f"  num_attention_heads: {model.config.num_attention_heads}")
print(f"  intermediate_size: {model.config.intermediate_size}")

# 计算参数量
params = sum(p.numel() for p in model.parameters())
print(f"  总参数量: {params / 1e9:.2f}B")
```

## 替代方案：使用更新的小型 DIT

如果您想使用最新的、专门设计的小型 DIT 架构，可以考虑：

1. **DiT-XL/2** 或更小的变体
2. **Latte** (Latent Diffusion Transformer) - 专门为扩散模型设计的轻量级 Transformer
3. **U-ViT** - 另一种高效的扩散 Transformer

但这些需要重新实现或适配到 Qwen-Image 的架构中。

## 推荐配置

对于大多数情况，推荐使用**方法 1（减少层数）**，因为：
- 实现简单
- 效果可预测
- 可以保留原始模型的权重初始化

建议的配置：
```yaml
dit_config:
  num_layers: 12  # 从 24 减到 12，参数量减半
```

如果需要更激进的缩小：
```yaml
dit_config:
  num_layers: 8
  hidden_size: 1536  # 从 2048 减到 1536
```

