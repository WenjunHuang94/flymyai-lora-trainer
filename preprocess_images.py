from PIL import Image
import os


def preprocess_image(image_path, output_dir, target_size=(1024, 1024), pad_color=(0, 0, 0)):
    """
    处理单张图片：缩放并添加留白，使其成为目标尺寸的正方形。
    Args:
        image_path (str): 输入图片路径。
        output_dir (str): 输出图片保存目录。
        target_size (tuple): 目标尺寸 (宽度, 高度)，默认为 (1024, 1024)。
        pad_color (tuple): 留白填充颜色 (R, G, B)，默认为黑色。
    """
    img = Image.open(image_path).convert("RGB")  # 确保是 RGB 模式

    original_width, original_height = img.size
    target_width, target_height = target_size

    # 计算缩放比例，使图片最长边适应目标尺寸，并保持原图比例
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 缩放图片
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建一个新的目标尺寸的空白图片，并填充背景颜色
    new_img = Image.new("RGB", target_size, pad_color)

    # 计算粘贴位置（居中）
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 将缩放后的图片粘贴到新图片的中心
    new_img.paste(img, (x_offset, y_offset))

    # 构建输出路径，保留原始文件名，保存为 JPG
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, output_filename)

    # 保存为 JPG 格式
    new_img.save(output_path, "JPEG", quality=95)  # quality=95 是一个不错的平衡点

    print(f"处理完成: {image_path} -> {output_path}")


# --- 配置 ---
input_images_dir = "./my_training_data"  # 替换成您存放原始照片的文件夹路径
output_processed_dir = "./my_processed_dataset"  # 替换成您希望保存处理后照片的文件夹路径

# 确保输出目录存在
os.makedirs(output_processed_dir, exist_ok=True)

# 遍历输入目录中的所有 JPG 图片并处理
for filename in os.listdir(input_images_dir):
    if filename.lower().endswith((".jpg", ".jpeg")):
        image_path = os.path.join(input_images_dir, filename)
        preprocess_image(image_path, output_processed_dir)

print("\n所有图片处理完毕。")