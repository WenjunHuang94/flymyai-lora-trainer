# Visual Marker - 图像编辑与数据生成工具包

本项目包含一系列 Python 脚本，用于调用 Qwen-VL 和 Qwen-Image-Edit 模型，实现图像编辑和 AI 训练数据的自动化生成。

## 项目结构

### 核心依赖（工具脚本）

这些脚本被其他主脚本调用，提供了核心功能。

#### `prompt_utils.py` - API 核心

**功能：** 封装了对阿里云 Dashscope API 的调用。

- **`edit_api`**: 使用 Qwen-VL 进行视觉定位，根据文本描述（如 "the cat"）返回其在图中的坐标。
- **`polish_edit_prompt`**: 使用 Qwen-Plus 润色和优化输入的自然语言指令。

**关系：** 被 `ultraedit_replace.py`、`gen_force.py` 和 `doodles_double.py` 依赖。

#### `multi3_infer_plus.py` - 多GPU推理引擎

**功能：** 加载 Qwen-Image-Edit-2509 模型，并将其分布到 4 个 GPU 上：
- `cuda:0`: 存放 VAE 和 TextEncoder
- `cuda:1,2,3`: 存放 Transformer（按块三等分）

**关系：** 被 `doodles_double.py` 脚本导入，作为其执行图像编辑的后端。

---

### 主要功能（任务脚本）

这些是您可以直接运行以执行特定任务的脚本。

#### `ultraedit_replace.py` - 替换指令 + 箭头标记

**功能：** 处理"替换"类指令。它会：
1. 读取 JSON，从指令中提取要被替换的物体（如 "replace car with boat"）
2. 使用 `prompt_utils.py` 中的 Qwen-VL 定位物体（car）
3. 在图上绘制一个箭头指向该物体
4. 将指令文本修改为 "replace the object pointed to by the arrow..." 并绘制在图上

**输入数据：**
- `json_path`: 一个 JSON 文件（例如 `ultraedit_replace.json`），包含：
  - `input`: 原图路径
  - `output`: 目标图路径
  - `instruction`: 替换指令
  - `input_description`: 图片描述（可选，用于帮助 API 理解图片内容）
- `input_dir`: 存放 JSON 中引用的所有原图

**保存的数据：**
- `result_input_dir`: 保存处理后的输入图。图上包含：
  - 一个指向被替换物体的箭头
  - 修改后的指令文本（例如："replace the object pointed to by the arrow..."）
- `result_output_dir`: 保存 `output_dir` 中原始目标图的副本

**使用：**
```bash
python ultraedit_replace.py          # 处理全部数据
python ultraedit_replace.py --test  # 测试模式：只处理前3张图片
python ultraedit_replace.py --limit N  # 限制处理的图片数量
```

#### `doodles_double.py` - Doodle 二阶段生成

**功能：** "涂鸦(Doodle) → 真实(Realistic)" 二阶段数据生成。

- **阶段 1**: 在原图上添加一个"涂鸦"风格的物体（如 "a doodle hat"），保存为 input 图片
- **阶段 2**: 加载上一步的 input 图片，将图中的"涂鸦"物体转换为"写实"风格，保存为 output 图片

**输入数据：**
- `original_images_dir`: 存放所有源图片（例如动物照片），默认路径为 `imgs_test/originals`
- `json_config_path`: 一个 JSON 文件（例如 `edit_doodles.json`），包含一个物体列表（例如 `["a party hat", "a tie"]`）

**保存的数据：**
- `input_dir`（例如 `imgs_test/input`）: 保存阶段一的图片，即 "原图 + 涂鸦风格的物体"
- `output_dir`（例如 `imgs_test/output`）: 保存阶段二的图片，即 "原图 + 写实风格的物体"

**使用：**
```bash
python doodles_double.py
```

**注意：** 脚本会随机为每张图片选择 1-2 个物体进行添加。

#### `filter_ultraedit.py` - 仅添加文本

**功能：** 一个简单的数据处理脚本。它读取 JSON，然后直接将指令（instruction）文本绘制到 input 图片的底部。

**注意：** 此脚本不使用 Qwen-VL，也不会绘制箭头。

**输入数据：**
- `json_path`: 一个 JSON 文件（例如 `ultraedit_add.json`），包含 `input`、`output` 和 `instruction`
- `input_dir`: 存放 JSON 中引用的所有原图

**保存的数据：**
- `result_input_dir`: 保存处理后的输入图。图上仅在底部添加了原始的指令文本
- `result_output_dir`: 保存 `output_dir` 中原始目标图的副本

**使用：**
```bash
python filter_ultraedit.py
```

#### `gen_force.py` - 力场箭头（演示脚本）

**功能：** 一个演示脚本。使用 Qwen-VL 定位图中的物体（如 "ball"），然后从物体中心向外随机绘制一个箭头。

**输入数据：**
- `input_image`: 一个硬编码在脚本中的单张图片路径（例如 `imgs/balls.png`）

**保存的数据：**
- `output_image`: 一个硬编码的单张输出图片路径（例如 `imgs/balls_with_force_arrow.png`），图上增加了从物体中心向外的箭头

**使用：**
```bash
python gen_force.py
```

---

## 依赖要求

### Python 包
- `torch`
- `PIL` (Pillow)
- `dashscope` (阿里云 Dashscope SDK)
- `diffusers` (Hugging Face)
- `tqdm`
- `numpy`

### API 密钥
需要在 `prompt_utils.py` 中配置阿里云 Dashscope API 密钥。

### 硬件要求
- 多 GPU 环境（推荐 4 个 GPU，用于 `doodles_double.py`）
- 足够的显存以加载 Qwen-Image-Edit-2509 模型

---

## 使用示例

### 示例 1: 处理替换指令数据集

```bash
# 修改 ultraedit_replace.py 中的路径配置
# 然后运行：
python ultraedit_replace.py --test  # 先测试
python ultraedit_replace.py         # 处理全部数据
```

### 示例 2: 生成 Doodle 二阶段数据

```bash
# 1. 准备源图片到 imgs_test/originals/
# 2. 配置 edit_doodles.json 中的物体列表
# 3. 运行：
python doodles_double.py
```

---

## 注意事项

1. **API 密钥配置**：使用前请确保在 `prompt_utils.py` 中正确配置了 Dashscope API 密钥
2. **路径配置**：各脚本中的路径可能需要根据实际环境进行调整
3. **GPU 配置**：`doodles_double.py` 需要多 GPU 环境，请确保 CUDA 设备配置正确
4. **字体文件**：部分脚本需要 Times New Roman 字体文件，如果缺失会使用默认字体

---

## 脚本关系图

```
prompt_utils.py (API核心)
    ├── ultraedit_replace.py
    ├── gen_force.py
    └── doodles_double.py
            └── multi3_infer_plus.py (多GPU推理引擎)

filter_ultraedit.py (独立脚本，不依赖API)
```

---

## 许可证

请参考项目根目录的 LICENSE 文件。

