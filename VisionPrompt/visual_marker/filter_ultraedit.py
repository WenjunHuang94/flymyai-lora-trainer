'''ultraedit;该文件直接从json文件中提取prompt添加到原来的input image中'''
import json
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
from tqdm import tqdm

def load_json_data(json_path):
    """加载JSON文件并返回数据列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def wrap_text(text, font, max_width):
    """
    将文本按照指定宽度进行换行
    """
    lines = []
    words = text.split(' ')
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # 单个词太长，强制添加
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_text_color_from_background(img, box_x, box_y, box_width, box_height):
    """
    分析图片指定区域的平均颜色，返回合适的文字颜色
    参考filter_omniedit.py的实现
    """
    # 裁剪文本框区域
    region = img.crop((box_x, box_y, box_x + box_width, box_y + box_height))
    
    # 缩小图片以加快计算速度
    region = region.resize((50, 50))
    
    # 转换为RGB模式
    if region.mode != 'RGB':
        region = region.convert('RGB')
    
    # 获取所有像素
    pixels = list(region.getdata())
    
    # 计算平均RGB值
    avg_r = sum(p[0] for p in pixels) / len(pixels)
    avg_g = sum(p[1] for p in pixels) / len(pixels)
    avg_b = sum(p[2] for p in pixels) / len(pixels)
    
    # 计算感知亮度 (使用标准公式)
    brightness = (0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b)
    
    # 根据亮度选择文字颜色和描边颜色
    if brightness > 127:  # 浅色背景
        text_color = (0, 0, 0)  # 黑色文字
        outline_color = (255, 255, 255)  # 白色描边
    else:  # 深色背景
        text_color = (255, 255, 255)  # 白色文字
        outline_color = (0, 0, 0)  # 黑色描边
    
    return text_color, outline_color

def draw_text_with_outline(draw, position, text, font, text_color, outline_color, outline_width=2):
    """
    绘制带描边的文字
    """
    x, y = position
    
    # 绘制描边（在8个方向上绘制）
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    
    # 绘制主文字
    draw.text((x, y), text, font=font, fill=text_color)

def get_average_color(image, x: int, y: int, width: int, height: int):
    """获取指定区域的平均颜色"""
    import numpy as np
    
    img_array = np.array(image) if isinstance(image, Image.Image) else image
    img_height, img_width = img_array.shape[:2]
    
    # 简化的边界检查和采样
    x = max(10, min(x, img_width - width - 10))
    y = max(10, min(y, img_height - height - 10))
    x_end = min(x + width, img_width - 10)
    y_end = min(y + height, img_height - 10)
    
    # 直接采样中心区域
    center_x = (x + x_end) // 2
    center_y = (y + y_end) // 2
    sample_size = min(20, width // 3, height // 3)
    
    region = img_array[center_y:center_y+sample_size, center_x:center_x+sample_size]
    
    if len(region.shape) == 3:
        avg_color = np.mean(region, axis=(0, 1))
        return tuple(int(c) for c in avg_color)
    else:
        avg_color = int(np.mean(region))
        return (avg_color, avg_color, avg_color)

def get_contrasting_color(background_color):
    """根据背景色选择对比度最高的字体颜色 - 优化清晰度"""
    # 优化的候选颜色 - 移除容易模糊的颜色组合，增加高对比度颜色
    colors = [
        (255, 255, 255),  # 白色 - 优先级最高
        (255, 255, 0),    # 黄色 - 高可见性
        (0, 255, 255),    # 青色 - 高对比度
        (255, 100, 0),    # 橙色 - 温暖高对比度
        (0, 255, 0),      # 绿色 - 高可见性
        (255, 0, 255),    # 品红色 - 高对比度
        (255, 0, 0),      # 红色 - 警示色
        (0, 0, 0),        # 黑色 - 最后选择
    ]
    
    # 计算背景亮度
    bg_luminance = 0.299 * background_color[0] + 0.587 * background_color[1] + 0.114 * background_color[2]
    
    best_color = (255, 255, 255)  # 默认白色
    max_contrast = 0
    
    for color in colors:
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        contrast = abs(bg_luminance - luminance)
        
        # 额外的可读性检查
        if contrast > max_contrast:
            # 避免选择与背景过于相似的颜色
            color_distance = sum(abs(background_color[i] - color[i]) for i in range(3))
            if color_distance > 100:  # 确保足够的颜色差异
                max_contrast = contrast
                best_color = color
    
    # 如果背景很暗，优先使用亮色；如果背景很亮，优先使用深色
    if bg_luminance < 64:  # 很暗的背景
        return (255, 255, 255)  # 白色
    elif bg_luminance > 192:  # 很亮的背景
        return (0, 0, 0)  # 黑色
    
    return best_color

def load_font(size: int):
    """加载字体的统一方法"""
    # 尝试加载Times New Roman字体
    font_paths = [
        "/storage/v-jinpewang/lab_folder/junchao/data/Times_New_Roman.ttf",
        "Times New Roman.ttf"
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # 如果都失败，使用默认字体
    return ImageFont.load_default()

def adjust_font_size_with_measurement(text: str, img_width: int, img_height: int, 
                                     initial_size: int = 48, min_size: int = 16):
    """使用实际字体测量动态调整字体大小"""
    # 创建临时图像用于测量
    temp_img = Image.new('RGB', (100, 100))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # 计算可用空间，预留安全边距 - 根据图片大小动态调整
    if img_width < 200 or img_height < 100:  # 很小的图片
        safety_margin = 10
    elif img_width < 400 or img_height < 200:  # 小图片
        safety_margin = 20
    else:  # 正常大小图片
        safety_margin = 40
    
    max_available_width = img_width - safety_margin
    max_available_height = img_height - safety_margin
    
    # 确保最小可用空间
    max_available_width = max(50, max_available_width)
    max_available_height = max(30, max_available_height)
    
    # 根据文本长度动态调整初始字体大小 - 整体增大字体
    text_length = len(text)
    
    if text_length <= 8:  # 短文本 - 超大字体
        dynamic_initial_size = min(72, initial_size + 24)
        dynamic_min_size = max(24, min_size + 8)
    elif text_length <= 15:  # 中等文本 - 大字体
        dynamic_initial_size = min(60, initial_size + 12)
        dynamic_min_size = max(20, min_size + 4)
    elif text_length <= 25:  # 较长文本 - 标准字体
        dynamic_initial_size = initial_size
        dynamic_min_size = min_size
    else:  # 很长文本 - 稍小字体
        dynamic_initial_size = max(32, initial_size - 16)
        dynamic_min_size = max(12, min_size - 4)
    
    # 进一步根据图片大小调整字体范围 - 保持更大的字体
    img_area = img_width * img_height
    if img_area < 200000:  # 小图片 - 减少缩放幅度
        dynamic_initial_size = int(dynamic_initial_size * 0.9)
        dynamic_min_size = max(12, int(dynamic_min_size * 0.9))
    elif img_area > 1000000:  # 大图片 - 增加字体大小
        dynamic_initial_size = int(dynamic_initial_size * 1.3)
        dynamic_min_size = int(dynamic_min_size * 1.2)
    
    # 确保字体大小在合理范围内 - 允许更大的字体
    dynamic_initial_size = max(dynamic_min_size, min(96, dynamic_initial_size))
    
    for font_size in range(dynamic_initial_size, dynamic_min_size - 1, -1):
        # 加载字体
        font = load_font(font_size)
        
        # 实际测量文本尺寸
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        actual_text_width = bbox[2] - bbox[0]
        actual_text_height = bbox[3] - bbox[1]
        
        # 动态调整padding，根据字体大小和文本长度 - 增加更多padding确保文本不超出边界
        if font_size <= 12:
            padding = max(10, font_size // 2)
        elif font_size <= 20:
            padding = max(12, font_size // 2)
        elif font_size <= 40:
            padding = max(15, font_size // 3)
        else:
            padding = max(20, font_size // 3)
        
        # 长文本也保持足够的padding，避免超出边界
        if text_length > 20:
            padding = max(8, padding - 3)  # 减少的幅度更小
        
        box_width = actual_text_width + 2 * padding
        box_height = actual_text_height + 2 * padding
        
        # 严格检查文本框是否能完全放入图片
        if box_width <= max_available_width and box_height <= max_available_height:
            return font_size, box_width, box_height, actual_text_width, actual_text_height, font, padding
    
    # 如果所有字体都太大，使用最小字体并强制适应
    font = load_font(dynamic_min_size)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    actual_text_width = bbox[2] - bbox[0]
    actual_text_height = bbox[3] - bbox[1]
    
    # 最小padding - 增加以确保文本不超出边界
    min_padding = max(8, dynamic_min_size // 3)
    forced_box_width = min(actual_text_width + 2 * min_padding, max_available_width)
    forced_box_height = min(actual_text_height + 2 * min_padding, max_available_height)
    
    return dynamic_min_size, forced_box_width, forced_box_height, actual_text_width, actual_text_height, font, min_padding

def validate_textbox_boundaries(box_x, box_y, box_width, box_height, img_width, img_height, filename=""):
    """验证文本框边界是否在图片内"""
    errors = []
    
    if box_x < 0:
        errors.append(f"左边界超出: box_x={box_x}")
    if box_y < 0:
        errors.append(f"上边界超出: box_y={box_y}")
    if box_x + box_width > img_width:
        errors.append(f"右边界超出: box_x+width={box_x + box_width} > img_width={img_width}")
    if box_y + box_height > img_height:
        errors.append(f"下边界超出: box_y+height={box_y + box_height} > img_height={img_height}")
    
    if errors:
        tqdm.write(f"⚠️  边界错误 {filename}: {'; '.join(errors)}")
        return False
    return True

def add_text_box_to_image(image_path, instruction_text, output_path):
    """在图片上添加带背景的文本框 - 使用改进的算法"""
    try:
        import numpy as np
        
        # 打开并验证图片
        try:
            img = Image.open(image_path)
            # 验证图片完整性
            img.verify()
            # 重新打开图片（verify后需要重新打开）
            img = Image.open(image_path)
            
            # 检查图片基本属性
            if img.size[0] <= 0 or img.size[1] <= 0:
                raise ValueError(f"无效的图片尺寸: {img.size}")
                
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
        except Exception as img_error:
            print(f"打开或验证图片失败 {image_path}: {str(img_error)}")
            return False
        
        width, height = img.size
        filename = os.path.basename(image_path)
        
        # 验证指令文本
        if not instruction_text or not instruction_text.strip():
            print(f"无效的指令文本: '{instruction_text}' for {image_path}")
            return False
            
        instruction_text = instruction_text.strip()
        
        # 计算安全边距
        if width < 200 or height < 100:  # 很小的图片
            safety_margin = 10
        elif width < 400 or height < 200:  # 小图片
            safety_margin = 20
        else:  # 正常大小图片
            safety_margin = 40
        
        # 计算可用空间
        max_available_width = width - 2 * safety_margin
        max_available_height = height - 2 * safety_margin
        
        # 使用改进的字体大小调整算法 - 支持文本换行
        font_size, box_width, box_height, text_width, text_height, font, padding = adjust_font_size_with_measurement(
            instruction_text, width, height, initial_size=56, min_size=20)
        
        # 尝试对文本进行换行处理，如果文本太长
        max_text_width = int(max_available_width * 0.9)  # 文字最大宽度为可用宽度的90%
        lines = wrap_text(instruction_text, font, max_text_width)
        
        # 如果换行后有多行，重新计算文本框高度
        if len(lines) > 1:
            # 计算行高
            line_height = font.getbbox('Ay')[3] - font.getbbox('Ay')[1] + 5
            total_text_height = line_height * len(lines)
            
            # 计算最大文本宽度
            max_line_width = 0
            for line in lines:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                max_line_width = max(max_line_width, line_width)
            
            # 更新文本框尺寸
            text_width = max_line_width
            text_height = total_text_height
            box_width = text_width + 2 * padding
            box_height = text_height + 2 * padding
        
        # 确保文本框尺寸不超过图片尺寸
        max_box_width = max_available_width
        max_box_height = max_available_height
        
        # 如果文本框太大，调整尺寸
        if box_width > max_box_width:
            box_width = max_box_width
        if box_height > max_box_height:
            box_height = max_box_height
        
        # 计算文字区域（用于分析背景色）
        # 文字位置在底部居中
        padding = 15
        text_area_y = height - box_height - padding * 2
        text_area_x = (width - box_width) // 2
        
        # 确保文字区域在图片范围内
        text_area_y = max(0, text_area_y)
        text_area_x = max(0, text_area_x)
        text_area_width = min(box_width, width - text_area_x)
        text_area_height = min(box_height, height - text_area_y)
        
        # 使用新的背景色分析方法自动选择文字颜色和描边颜色
        text_color, outline_color = get_text_color_from_background(img, text_area_x, text_area_y, text_area_width, text_area_height)
        
        # 创建绘图对象（直接在原图上绘制，参考filter_omniedit.py）
        draw = ImageDraw.Draw(img)
        
        # 绘制多行文本（自适应颜色 + 描边效果，参考filter_omniedit.py的样式）
        # 计算行高
        line_height = font.getbbox('Ay')[3] - font.getbbox('Ay')[1] + 5
        total_text_height = line_height * len(lines)
        
        # 文字位置（底部居中，参考filter_omniedit.py）
        padding = 15
        start_y = height - total_text_height - padding * 2
        
        # 逐行绘制
        current_y = start_y
        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            
            # 水平居中
            text_x = (width - line_width) // 2
            
            # 使用新的绘制函数绘制带描边的文字
            draw_text_with_outline(
                draw, 
                (text_x, current_y), 
                line, 
                font, 
                text_color, 
                outline_color,
                outline_width=2
            )
            
            current_y += line_height
        
        # 保存图片，使用最基本的保存方式避免格式兼容性问题
        try:
            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 使用最简单的保存方式，不带任何额外参数
            img.save(output_path)
            
            # 验证保存的图片是否完整
            try:
                # 尝试重新打开图片验证完整性
                with Image.open(output_path) as verify_img:
                    verify_img.verify()  # 验证图片完整性
                    
                # 再次打开并检查基本属性
                with Image.open(output_path) as check_img:
                    if check_img.size != img.size:
                        raise ValueError(f"保存后图片尺寸不匹配: 期望{img.size}, 实际{check_img.size}")
                        
            except Exception as verify_error:
                print(f"保存的图片验证失败 {output_path}: {str(verify_error)}")
                # 删除损坏的文件
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False
                
            return True
            
        except Exception as save_error:
            print(f"保存图片失败 {output_path}: {str(save_error)}")
            # 确保不留下损坏的文件
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return False
        
    except Exception as e:
        print(f"处理图片时出错 {image_path}: {str(e)}")
        return False

def process_dataset(json_path, input_dir, output_dir, result_input_dir, result_output_dir):
    """处理整个数据集"""
    # 创建输出目录
    os.makedirs(result_input_dir, exist_ok=True)
    os.makedirs(result_output_dir, exist_ok=True)
    
    # 加载JSON数据
    data = load_json_data(json_path)
    print(f"加载了 {len(data)} 条数据")
    
    success_count = 0
    fail_count = 0
    
    # 使用tqdm显示处理进度
    with tqdm(total=len(data), desc="处理图片", unit="张", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for idx, item in enumerate(data):
            try:
                # 获取文件路径和指令
                input_filename = item['input'].split('/')[-1]  # 提取文件名
                output_filename = item['output'].split('/')[-1]
                instruction = item['instruction']
                
                input_img_path = os.path.join(input_dir, input_filename)
                output_img_path = os.path.join(output_dir, output_filename)
                
                # 更新进度条描述
                pbar.set_description(f"处理图片 {input_filename[:20]}...")
                
                # 检查输入文件是否存在
                if not os.path.exists(input_img_path):
                    tqdm.write(f"❌ 输入图片不存在: {input_filename}")
                    fail_count += 1
                    pbar.update(1)
                    continue
                
                # 检查输出图片是否存在（仅警告，不跳过处理）
                output_exists = os.path.exists(output_img_path)
                if not output_exists:
                    tqdm.write(f"⚠️  输出图片不存在，将只处理输入图片: {output_filename}")
                
                # 生成新的文件名
                result_input_path = os.path.join(result_input_dir, input_filename)
                result_output_path = os.path.join(result_output_dir, output_filename)
                
                # 在输入图片上添加文本框
                text_processing_success = False
                try:
                    text_processing_success = add_text_box_to_image(input_img_path, instruction, result_input_path)
                except Exception as text_error:
                    tqdm.write(f"❌ 文本框添加异常: {input_filename}, 错误: {str(text_error)}")
                    text_processing_success = False

                # 直接复制output图片（如果存在）
                output_copy_success = True
                if output_exists:
                    try:
                        shutil.copy2(output_img_path, result_output_path)
                    except Exception as copy_error:
                        tqdm.write(f"⚠️  复制输出图片失败: {output_filename}, 错误: {str(copy_error)}")
                        output_copy_success = False

                if text_processing_success:
                    # 验证生成的文件是否存在且有效
                    if not os.path.exists(result_input_path):
                        tqdm.write(f"❌ 处理后文件未生成: {input_filename}")
                        fail_count += 1
                    else:
                        if output_exists and output_copy_success:
                            tqdm.write(f"✅ 成功处理: {input_filename} (含输出图片)")
                        elif output_exists and not output_copy_success:
                            tqdm.write(f"⚠️  输入图片处理成功，但输出图片复制失败: {input_filename}")
                        else:
                            tqdm.write(f"✅ 成功处理: {input_filename} (仅输入图片)")
                        
                        success_count += 1
                        
                        # 每100张显示一次详细信息
                        if (idx + 1) % 100 == 0:
                            tqdm.write(f"✅ 已成功处理 {success_count} 张，失败 {fail_count} 张")
                else:
                    fail_count += 1
                    tqdm.write(f"❌ 处理失败: {input_filename}")
                    
                    # 清理可能存在的损坏文件
                    if os.path.exists(result_input_path):
                        try:
                            os.remove(result_input_path)
                            tqdm.write(f"🗑️  已清理损坏文件: {result_input_path}")
                        except Exception as cleanup_error:
                            tqdm.write(f"⚠️  清理损坏文件失败: {result_input_path}, 错误: {str(cleanup_error)}")
                    
                    # 如果输入图片处理失败，也要清理可能复制的输出图片
                    if os.path.exists(result_output_path):
                        try:
                            os.remove(result_output_path)
                            tqdm.write(f"🗑️  已清理输出图片: {result_output_path}")
                        except Exception as cleanup_error:
                            tqdm.write(f"⚠️  清理输出图片失败: {result_output_path}, 错误: {str(cleanup_error)}")
                    
            except Exception as e:
                fail_count += 1
                tqdm.write(f"❌ 处理第 {idx+1} 条数据时出错: {str(e)}")
            
            # 更新进度条
            pbar.update(1)
            
            # 更新进度条后缀信息
            pbar.set_postfix({
                '成功': success_count,
                '失败': fail_count,
                '成功率': f"{success_count/(success_count+fail_count)*100:.1f}%" if (success_count+fail_count) > 0 else "0%"
            })
    
    print(f"\n处理完成!")
    print(f"成功处理: {success_count} 张图片")
    print(f"处理失败: {fail_count} 张图片")

def main():
    # 设置路径
    base_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/GPT-Image-Edit/ultraedit/gpt-edit/ultraedit/ultraedit/add"
    json_path = os.path.join(base_dir, "metadata/ultraedit_add.json")
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    
    # 结果保存路径
    result_input_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/input"
    result_output_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/output"
    # result_input_dir = "/storage/v-jinpewang/lab_folder/junchao/data/test_ultraedit/input"
    # result_output_dir = "/storage/v-jinpewang/lab_folder/junchao/data/test_ultraedit/output"
    
    print("开始处理数据集...")
    print(f"JSON文件: {json_path}")
    print(f"输入图片目录: {input_dir}")
    print(f"输出图片目录: {output_dir}")
    print(f"处理后输入图片保存到: {result_input_dir}")
    print(f"处理后输出图片保存到: {result_output_dir}")
    
    process_dataset(json_path, input_dir, output_dir, result_input_dir, result_output_dir)

if __name__ == "__main__":
    main()

