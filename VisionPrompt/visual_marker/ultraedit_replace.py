'''ultraedit replace;从instruction中提取要替换的物体,使用Qwen VL标记物体位置,用箭头指示并添加instruction文本'''
import json
import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
from tqdm import tqdm
import sys

# 导入Qwen相关模块
try:
    from prompt_utils import edit_api, encode_image
except ImportError:
    print("警告: 无法导入prompt_utils,将使用简化版本")
    edit_api = None

def load_json_data(json_path):
    """加载JSON文件并返回数据列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_object_from_instruction(instruction):
    """从instruction中提取要替换的物体
    
    例如：
    - "Replace the high-rise building with a mountain" -> "high-rise building"
    - "replace the kite with a hot air balloon" -> "kite"
    - "remove the grass field and replace it with a snowy landscape" -> "grass field"
    """
    # 使用正则表达式提取物体名称,按优先级尝试不同模式
    patterns = [
        # 匹配 "remove ... and replace it with ..." 格式
        r'[Rr]emove\s+the\s+(.+?)\s+and\s+replace',
        r'[Rr]emove\s+(.+?)\s+and\s+replace',
        # 匹配 "replace ... with ..." 格式
        r'[Rr]eplace\s+the\s+(.+?)\s+with',
        r'[Rr]eplace\s+(.+?)\s+with',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            object_name = match.group(1).strip()
            return object_name
    
    # 如果没有匹配到,返回None
    return None

def is_plural(word):
    """判断单词是否为复数形式
    
    Args:
        word: 要判断的单词
    
    Returns:
        bool: True表示复数，False表示单数
    """
    if not word:
        return False
    
    word = word.strip().lower()
    
    # 常见复数规则
    # 1. 以s结尾的大多数情况
    if word.endswith('s'):
        # 排除一些以s结尾的单数词
        singular_s_endings = ['grass', 'glass', 'class', 'pass', 'mass', 'bass', 'brass']
        if word in singular_s_endings:
            return False
        
        # 以ss, us, is等结尾的通常是单数
        if word.endswith(('ss', 'us', 'is')):
            return False
            
        return True
    
    # 2. 不规则复数
    irregular_plurals = [
        'children', 'people', 'men', 'women', 'teeth', 'feet', 
        'mice', 'geese', 'oxen', 'sheep', 'deer', 'fish'
    ]
    if word in irregular_plurals:
        return True
    
    return False

def modify_instruction_text(instruction, object_name):
    """修改instruction文本,将被替换的物体改为'arrow'
    
    例如：
    - "replace the kite with a hot air balloon" 
      -> "replace the object pointed to by the arrow with a hot air balloon"
    - "replace the tulips with sunflowers" (复数)
      -> "replace all the objects of this type pointed to by the arrow with sunflowers"
    - "Replace the high-rise building with a mountain"
      -> "Replace the object pointed to by the arrow with a mountain"
    - "remove the grass field and replace it with a snowy landscape"
      -> "remove the object pointed to by the arrow and replace it with a snowy landscape"
    """
    if not object_name:
        return instruction
    
    # 检查首字母大小写
    is_capitalized = instruction[0].isupper()
    
    # 判断物体名称是否为复数
    plural = is_plural(object_name)
    
    # 根据单复数选择替换文本
    if plural:
        # 复数：使用 "all the objects of this type"
        remove_replace_text = 'Remove all the objects of this type pointed to by the arrow and replace' if is_capitalized else 'remove all the objects of this type pointed to by the arrow and replace'
        replace_text = 'Replace all the objects of this type pointed to by the arrow with' if is_capitalized else 'replace all the objects of this type pointed to by the arrow with'
    else:
        # 单数：使用 "the object"
        remove_replace_text = 'Remove the object pointed to by the arrow and replace' if is_capitalized else 'remove the object pointed to by the arrow and replace'
        replace_text = 'Replace the object pointed to by the arrow with' if is_capitalized else 'replace the object pointed to by the arrow with'
    
    # 替换模式,按优先级尝试
    patterns = [
        # 匹配 "remove the X and replace" 格式
        (r'[Rr]emove\s+the\s+' + re.escape(object_name) + r'\s+and\s+replace', remove_replace_text),
        # 匹配 "remove X and replace" 格式（无"the"）
        (r'[Rr]emove\s+' + re.escape(object_name) + r'\s+and\s+replace', remove_replace_text),
        # 匹配 "replace the X with" 格式
        (r'[Rr]eplace\s+the\s+' + re.escape(object_name) + r'\s+with', replace_text),
        # 匹配 "replace X with" 格式（无"the"）
        (r'[Rr]eplace\s+' + re.escape(object_name) + r'\s+with', replace_text),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, instruction):
            modified = re.sub(pattern, replacement, instruction)
            return modified
    
    return instruction

def get_object_position_from_qwen(image, object_name, img_width, img_height, input_description=None):
    """使用Qwen VL API获取物体在图片中的位置
    
    Args:
        image: PIL图片对象
        object_name: 要定位的物体名称
        img_width, img_height: 图片尺寸
        input_description: 图片描述（可选），帮助模型更好理解图片内容
    
    返回: (center_x, center_y, bbox) 或 None
    bbox格式: (x1, y1, x2, y2) 像素坐标
    """
    if edit_api is None:
        tqdm.write("  ⚠️  Qwen API不可用")
        return None
    
    try:
        # 参考api_process_plus.py的实现,直接要求返回坐标
        # 明确要求只返回最主要的一个物体
        # 包含input_description帮助模型理解图片
        description_text = f"\nImage description: {input_description}\n" if input_description else ""
        
        prompt = f"""The size of this image is {img_width}*{img_height}.{description_text}
                Please locate the bounding box of the {object_name} in the image.

                IMPORTANT: If there are multiple {object_name} in the image, please identify and return ONLY THE MOST PROMINENT ONE based on the following criteria (in priority order):
                1. The largest one in size
                2. The most centered or visually dominant one
                3. The one in the foreground (if applicable)

                Output the bounding box coordinates in the format: top-left corner (x1,y1) and bottom-right corner (x2,y2).
                Note: (0,0) is the upper left corner, X-axis extends right, Y-axis extends down.
                You only need to output the coordinates for the single most prominent {object_name}."""
        
        # 调用Qwen VL API
        response = edit_api(prompt, [image], model="qwen-vl-max-latest")
        tqdm.write(f"  [DEBUG] API返回: {response[:150]}")
        
        # 方法1: 尝试解析 (x1,y1) (x2,y2) 格式
        coord_pattern = r'\((\d+),\s*(\d+)\)'
        matches = re.findall(coord_pattern, response)
        
        if len(matches) >= 2:
            # 找到两个坐标点,假设是左上和右下
            x1, y1 = int(matches[0][0]), int(matches[0][1])
            x2, y2 = int(matches[1][0]), int(matches[1][1])
            
            # 确保坐标在范围内
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            # 确保 x1 < x2, y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            tqdm.write(f"  ✓ 提取到边界框: ({x1},{y1}) - ({x2},{y2})")
            return center_x, center_y, (x1, y1, x2, y2)
        
        elif len(matches) == 1:
            # 只找到一个坐标,假设是中心点
            center_x, center_y = int(matches[0][0]), int(matches[0][1])
            center_x = max(0, min(center_x, img_width - 1))
            center_y = max(0, min(center_y, img_height - 1))
            
            # 估算一个边界框（假设物体占图片的1/6）
            box_size = min(img_width, img_height) // 6
            x1 = max(0, center_x - box_size // 2)
            y1 = max(0, center_y - box_size // 2)
            x2 = min(img_width - 1, center_x + box_size // 2)
            y2 = min(img_height - 1, center_y + box_size // 2)
            
            tqdm.write(f"  ✓ 提取到中心点: ({center_x},{center_y}),估算边界框")
            return center_x, center_y, (x1, y1, x2, y2)
        
        # 方法2: 尝试解析 x1=, y1=, x2=, y2= 格式
        x1_match = re.search(r'x1?\s*[=:]\s*(\d+)', response, re.IGNORECASE)
        y1_match = re.search(r'y1?\s*[=:]\s*(\d+)', response, re.IGNORECASE)
        x2_match = re.search(r'x2\s*[=:]\s*(\d+)', response, re.IGNORECASE)
        y2_match = re.search(r'y2\s*[=:]\s*(\d+)', response, re.IGNORECASE)
        
        if x1_match and y1_match and x2_match and y2_match:
            x1 = max(0, min(int(x1_match.group(1)), img_width - 1))
            y1 = max(0, min(int(y1_match.group(1)), img_height - 1))
            x2 = max(0, min(int(x2_match.group(1)), img_width - 1))
            y2 = max(0, min(int(y2_match.group(1)), img_height - 1))
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            tqdm.write(f"  ✓ 提取到边界框(x=y=格式): ({x1},{y1}) - ({x2},{y2})")
            return center_x, center_y, (x1, y1, x2, y2)
        
        # 如果都没匹配到,返回None
        tqdm.write(f"  ⚠️  无法从API响应中提取坐标")
        return None
        
    except Exception as e:
        tqdm.write(f"  ⚠️  Qwen VL API调用失败: {str(e)}")
        return None

def calculate_arrow_start_point(center_x, center_y, bbox, img_width, img_height):
    """计算箭头起点,确保不在物体bounding box内
    
    Args:
        center_x, center_y: 物体中心点
        bbox: 物体边界框 (x1, y1, x2, y2)
        img_width, img_height: 图片尺寸
    
    Returns:
        (start_x, start_y): 箭头起点坐标
    """
    if bbox is None:
        # 如果没有bbox,使用默认位置
        return img_width // 2, int(img_height * 0.85)
    
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    
    # 箭头起点与box的安全距离（进一步缩短箭头）
    safe_distance = max(20, min(box_width, box_height) // 4)
    
    # 候选位置：上、下、左、右
    candidates = []
    
    # 下方：物体下方,水平居中
    if y2 + safe_distance < img_height - 50:
        candidates.append((center_x, y2 + safe_distance, 'bottom'))
    
    # 上方：物体上方,水平居中
    if y1 - safe_distance > 50:
        candidates.append((center_x, y1 - safe_distance, 'top'))
    
    # 右侧：物体右侧,垂直居中
    if x2 + safe_distance < img_width - 50:
        candidates.append((x2 + safe_distance, center_y, 'right'))
    
    # 左侧：物体左侧,垂直居中
    if x1 - safe_distance > 50:
        candidates.append((x1 - safe_distance, center_y, 'left'))
    
    # 如果没有合适的候选位置,使用图片底部
    if not candidates:
        return img_width // 2, int(img_height * 0.9)
    
    # 优先选择下方,然后是上方、右侧、左侧
    priority = {'bottom': 0, 'top': 1, 'right': 2, 'left': 3}
    candidates.sort(key=lambda x: priority.get(x[2], 4))
    
    return candidates[0][0], candidates[0][1]

def draw_arrow_to_object(draw, start_x, start_y, end_x, end_y, arrow_color=(255, 0, 0), arrow_width=3):
    """绘制箭头指向物体
    
    Args:
        draw: ImageDraw对象
        start_x, start_y: 箭头起点（通常是文本框边缘）
        end_x, end_y: 箭头终点（物体中心）
        arrow_color: 箭头颜色
        arrow_width: 箭头宽度
    """
    import math
    
    # 绘制箭头线
    draw.line([(start_x, start_y), (end_x, end_y)], fill=arrow_color, width=arrow_width)
    
    # 绘制白色边框使箭头更醒目（边框细一些）
    outline_color = (255, 255, 255)
    draw.line([(start_x, start_y), (end_x, end_y)], fill=outline_color, width=arrow_width + 1)
    draw.line([(start_x, start_y), (end_x, end_y)], fill=arrow_color, width=arrow_width)
    
    # 计算箭头头部
    angle = math.atan2(end_y - start_y, end_x - start_x)
    arrow_length = 15  # 箭头三角形长度
    arrow_angle = math.pi / 7  # 约26度,更窄的箭头
    
    # 箭头两个边
    left_x = end_x - arrow_length * math.cos(angle - arrow_angle)
    left_y = end_y - arrow_length * math.sin(angle - arrow_angle)
    right_x = end_x - arrow_length * math.cos(angle + arrow_angle)
    right_y = end_y - arrow_length * math.sin(angle + arrow_angle)
    
    # 绘制箭头头部（三角形）- 先画白色边框
    arrow_head = [(end_x, end_y), (left_x, left_y), (right_x, right_y)]
    # 白色描边（更细的描边）
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 or j != 0:
                offset_head = [(x + i, y + j) for x, y in arrow_head]
                draw.polygon(offset_head, fill=outline_color)
    # 红色箭头
    draw.polygon(arrow_head, fill=arrow_color)

def load_font(size: int):
    """加载字体的统一方法"""
    font_paths = [
        "/storage/v-jinpewang/lab_folder/junchao/data/Times_New_Roman.ttf",
        "Times New Roman.ttf"
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    return ImageFont.load_default()

def wrap_text(text, font, max_width):
    """将文本按照指定宽度进行换行"""
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
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_text_color_from_background(img, box_x, box_y, box_width, box_height):
    """分析图片指定区域的平均颜色,返回合适的文字颜色"""
    region = img.crop((box_x, box_y, box_x + box_width, box_y + box_height))
    region = region.resize((50, 50))
    
    if region.mode != 'RGB':
        region = region.convert('RGB')
    
    pixels = list(region.getdata())
    avg_r = sum(p[0] for p in pixels) / len(pixels)
    avg_g = sum(p[1] for p in pixels) / len(pixels)
    avg_b = sum(p[2] for p in pixels) / len(pixels)
    
    brightness = (0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b)
    
    if brightness > 127:
        text_color = (0, 0, 0)
        outline_color = (255, 255, 255)
    else:
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)
    
    return text_color, outline_color

def draw_text_with_outline(draw, position, text, font, text_color, outline_color, outline_width=2):
    """绘制带描边的文字"""
    x, y = position
    
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    
    draw.text((x, y), text, font=font, fill=text_color)

def adjust_font_size_with_measurement(text: str, img_width: int, img_height: int, 
                                     initial_size: int = 48, min_size: int = 16):
    """使用实际字体测量动态调整字体大小"""
    temp_img = Image.new('RGB', (100, 100))
    temp_draw = ImageDraw.Draw(temp_img)
    
    if img_width < 200 or img_height < 100:
        safety_margin = 10
    elif img_width < 400 or img_height < 200:
        safety_margin = 20
    else:
        safety_margin = 40
    
    max_available_width = img_width - safety_margin
    max_available_height = img_height - safety_margin
    max_available_width = max(50, max_available_width)
    max_available_height = max(30, max_available_height)
    
    text_length = len(text)
    
    if text_length <= 8:
        dynamic_initial_size = min(100, initial_size + 24)
        dynamic_min_size = max(36, min_size + 8)
    elif text_length <= 15:
        dynamic_initial_size = min(90, initial_size + 12)
        dynamic_min_size = max(32, min_size + 4)
    elif text_length <= 25:
        dynamic_initial_size = initial_size
        dynamic_min_size = min_size
    else:
        dynamic_initial_size = max(48, initial_size - 16)
        dynamic_min_size = max(24, min_size - 4)
    
    img_area = img_width * img_height
    if img_area < 200000:
        dynamic_initial_size = int(dynamic_initial_size * 0.9)
        dynamic_min_size = max(20, int(dynamic_min_size * 0.9))
    elif img_area > 1000000:
        dynamic_initial_size = int(dynamic_initial_size * 1.4)
        dynamic_min_size = int(dynamic_min_size * 1.3)
    
    dynamic_initial_size = max(dynamic_min_size, min(120, dynamic_initial_size))
    
    for font_size in range(dynamic_initial_size, dynamic_min_size - 1, -1):
        font = load_font(font_size)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        actual_text_width = bbox[2] - bbox[0]
        actual_text_height = bbox[3] - bbox[1]
        
        if font_size <= 12:
            padding = max(10, font_size // 2)
        elif font_size <= 20:
            padding = max(12, font_size // 2)
        elif font_size <= 40:
            padding = max(15, font_size // 3)
        else:
            padding = max(20, font_size // 3)
        
        if text_length > 20:
            padding = max(8, padding - 3)
        
        box_width = actual_text_width + 2 * padding
        box_height = actual_text_height + 2 * padding
        
        if box_width <= max_available_width and box_height <= max_available_height:
            return font_size, box_width, box_height, actual_text_width, actual_text_height, font, padding
    
    font = load_font(dynamic_min_size)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    actual_text_width = bbox[2] - bbox[0]
    actual_text_height = bbox[3] - bbox[1]
    
    min_padding = max(8, dynamic_min_size // 3)
    forced_box_width = min(actual_text_width + 2 * min_padding, max_available_width)
    forced_box_height = min(actual_text_height + 2 * min_padding, max_available_height)
    
    return dynamic_min_size, forced_box_width, forced_box_height, actual_text_width, actual_text_height, font, min_padding

def add_marker_and_text_to_image(image_path, instruction_text, output_path, input_description=None):
    """在图片上添加物体标记箭头和instruction文本
    
    Args:
        image_path: 输入图片路径
        instruction_text: instruction文本
        output_path: 输出图片路径
        input_description: 图片描述（可选），帮助API更好理解图片内容
    
    Returns:
        bool: 处理是否成功
    """
    try:
        # 打开图片
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
        
        if img.size[0] <= 0 or img.size[1] <= 0:
            raise ValueError(f"无效的图片尺寸: {img.size}")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        
        if not instruction_text or not instruction_text.strip():
            print(f"无效的指令文本: '{instruction_text}'")
            return False
        
        instruction_text = instruction_text.strip()
        original_instruction = instruction_text  # 保存原始instruction
        
        # 1. 提取要替换的物体
        object_name = extract_object_from_instruction(instruction_text)
        if object_name:
            tqdm.write(f"  提取物体: {object_name}")
            # 修改instruction文本
            instruction_text = modify_instruction_text(instruction_text, object_name)
            tqdm.write(f"  修改后文本: {instruction_text}")
        else:
            tqdm.write(f"  ⚠️  无法从instruction中提取物体,将只添加文本")
        
        # 2. 获取物体位置（使用Qwen VL）
        object_position = None
        center_x, center_y = None, None
        bbox = None
        if object_name and edit_api is not None:
            try:
                object_position = get_object_position_from_qwen(img, object_name, width, height, input_description)
                if object_position:
                    center_x, center_y, bbox = object_position
                    tqdm.write(f"  ✓ 检测到物体位置: ({center_x}, {center_y})")
                    tqdm.write(f"  ✓ 物体边界框: {bbox}")
            except Exception as e:
                tqdm.write(f"  ⚠️  物体检测失败: {str(e)}")
        
        # 创建绘图对象
        draw = ImageDraw.Draw(img)
        
        # 3. 先确定箭头位置
        arrow_start_x, arrow_start_y = None, None
        arrow_end_x, arrow_end_y = None, None
        
        if object_name:
            # 如果没有检测到物体位置,使用图片中心上方作为默认位置
            if not object_position:
                center_x = width // 2
                center_y = height // 3  # 图片上方1/3处
                tqdm.write(f"  ℹ️  使用默认位置: ({center_x}, {center_y})")
                # 估算一个默认的bbox（物体大小假设为图片的1/6）
                box_size = min(width, height) // 6
                bbox = (
                    center_x - box_size // 2,
                    center_y - box_size // 2,
                    center_x + box_size // 2,
                    center_y + box_size // 2
                )
            
            # 箭头终点：物体中心
            arrow_end_x = center_x
            arrow_end_y = center_y
            
            # 箭头起点：使用智能计算,确保不在bbox内
            arrow_start_x, arrow_start_y = calculate_arrow_start_point(  # TODO: 这里是计算箭头终点
                center_x, center_y, bbox, width, height
            )
            tqdm.write(f"  ✓ 箭头起点: ({arrow_start_x}, {arrow_start_y})")
        
        # 4. 计算文本框位置和大小（基于箭头尾部）
        safety_margin = 40 if width >= 400 and height >= 200 else (20 if width >= 200 and height >= 100 else 10)
        max_available_width = width - 2 * safety_margin
        max_available_height = height - 2 * safety_margin
        
        font_size, box_width, box_height, text_width, text_height, font, padding = adjust_font_size_with_measurement(
            instruction_text, width, height, initial_size=80, min_size=32)
        
        # 文本换行（允许更宽的文本区域以容纳更大字体）
        max_text_width = int(max_available_width * 0.95)
        lines = wrap_text(instruction_text, font, max_text_width)
        
        if len(lines) > 1:
            line_height = font.getbbox('Ay')[3] - font.getbbox('Ay')[1] + 5
            total_text_height = line_height * len(lines)
            max_line_width = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)
            text_width = max_line_width
            text_height = total_text_height
            box_width = text_width + 2 * padding
            box_height = text_height + 2 * padding
        
        # 确保文本框不超过图片边界
        box_width = min(box_width, max_available_width)
        box_height = min(box_height, max_available_height)
        
        # 文本框位置（确保完全避开箭头）
        text_gap = 35  # 文本与箭头起点的最小间隔
        if arrow_start_x is not None and arrow_start_y is not None and bbox is not None:
            # 根据箭头起点位置和物体位置,智能选择文本位置
            x1, y1, x2, y2 = bbox
            
            # 计算箭头方向（从起点到终点）
            arrow_dx = arrow_end_x - arrow_start_x
            arrow_dy = arrow_end_y - arrow_start_y
            
            # 判断箭头起点在物体的哪个方向
            if arrow_start_y > y2:  # 箭头起点在物体下方
                # 文本放在箭头起点的侧面（而非延长线上）,避免遮挡箭头
                # 检查箭头是否垂直,如果是则放在侧面
                if abs(arrow_dx) < 30:  # 箭头接近垂直
                    # 优先放在左侧（避免超出右边界）
                    if arrow_start_x - box_width - text_gap > safety_margin:
                        text_area_x = arrow_start_x - box_width - text_gap
                        text_area_y = arrow_start_y - box_height // 2
                    else:
                        # 左侧空间不够,放右侧
                        text_area_x = arrow_start_x + text_gap
                        text_area_y = arrow_start_y - box_height // 2
                else:
                    # 箭头倾斜,放在下方但留出更多空间
                    text_area_x = arrow_start_x - box_width // 2
                    text_area_y = arrow_start_y + text_gap + 10
                    
            elif arrow_start_y < y1:  # 箭头起点在物体上方
                # 文本放在箭头起点的侧面
                if abs(arrow_dx) < 30:  # 箭头接近垂直
                    # 优先放在左侧
                    if arrow_start_x - box_width - text_gap > safety_margin:
                        text_area_x = arrow_start_x - box_width - text_gap
                        text_area_y = arrow_start_y - box_height // 2
                    else:
                        text_area_x = arrow_start_x + text_gap
                        text_area_y = arrow_start_y - box_height // 2
                else:
                    text_area_x = arrow_start_x - box_width // 2
                    text_area_y = arrow_start_y - box_height - text_gap - 10
                    
            elif arrow_start_x > x2:  # 箭头起点在物体右侧
                # 文本放在箭头起点的上方或下方,避开箭头线
                if abs(arrow_dy) < 30:  # 箭头接近水平
                    # 优先放在下方
                    if arrow_start_y + text_gap + box_height < height - safety_margin:
                        text_area_x = arrow_start_x - box_width // 2
                        text_area_y = arrow_start_y + text_gap
                    else:
                        text_area_x = arrow_start_x - box_width // 2
                        text_area_y = arrow_start_y - box_height - text_gap
                else:
                    text_area_x = arrow_start_x + text_gap + 10
                    text_area_y = arrow_start_y - box_height // 2
                    
            else:  # 箭头起点在物体左侧
                # 文本放在箭头起点的上方或下方
                if abs(arrow_dy) < 30:  # 箭头接近水平
                    # 优先放在下方
                    if arrow_start_y + text_gap + box_height < height - safety_margin:
                        text_area_x = arrow_start_x - box_width // 2
                        text_area_y = arrow_start_y + text_gap
                    else:
                        text_area_x = arrow_start_x - box_width // 2
                        text_area_y = arrow_start_y - box_height - text_gap
                else:
                    text_area_x = arrow_start_x - box_width - text_gap - 10
                    text_area_y = arrow_start_y - box_height // 2
            
            # 确保文本不超出边界
            text_area_x = max(safety_margin, min(text_area_x, width - box_width - safety_margin))
            text_area_y = max(safety_margin, min(text_area_y, height - box_height - safety_margin))
        elif arrow_start_x is not None and arrow_start_y is not None:
            # 如果没有bbox,默认放在箭头起点附近
            text_area_x = arrow_start_x - box_width // 2
            text_area_y = arrow_start_y + text_gap
            text_area_x = max(safety_margin, min(text_area_x, width - box_width - safety_margin))
            text_area_y = max(safety_margin, min(text_area_y, height - box_height - safety_margin))
        else:
            # 如果没有箭头,使用底部居中
            text_area_x = (width - box_width) // 2
            text_area_y = height - box_height - safety_margin
            text_area_x = max(0, text_area_x)
            text_area_y = max(0, text_area_y)
        
        # 5. 绘制箭头标记物体
        if object_name and arrow_start_x is not None:
            # 只有当箭头起点和终点不太接近时才绘制箭头
            distance = ((arrow_end_x - arrow_start_x)**2 + (arrow_end_y - arrow_start_y)**2)**0.5
            if distance > 50:  # 至少50像素的距离
                draw_arrow_to_object(draw, arrow_start_x, arrow_start_y, arrow_end_x, arrow_end_y,  # TODO： 画箭头
                                   arrow_color=(255, 0, 0), arrow_width=3)
                tqdm.write(f"  ✓ 箭头已绘制")
        
        # 6. 绘制instruction文本
        text_color, outline_color = get_text_color_from_background(img, text_area_x, text_area_y, box_width, box_height)
        
        line_height = font.getbbox('Ay')[3] - font.getbbox('Ay')[1] + 5
        total_text_height = line_height * len(lines)
        
        # 使用text_area_y作为起始位置
        padding_text = 10
        current_y = text_area_y + padding_text
        
        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            
            # 文本居中对齐于text_area_x + box_width/2
            text_x = text_area_x + (box_width - line_width) // 2
            
            draw_text_with_outline(draw, (text_x, current_y), line, font, 
                                 text_color, outline_color, outline_width=2)  #   # TODO：绘制instruction文本
            current_y += line_height
        
        # 7. 保存图片
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        img.save(output_path)
        
        # 验证保存的图片
        with Image.open(output_path) as verify_img:
            verify_img.verify()
        
        with Image.open(output_path) as check_img:
            if check_img.size != img.size:
                raise ValueError(f"保存后图片尺寸不匹配")
        
        return True
        
    except Exception as e:
        print(f"处理图片时出错 {image_path}: {str(e)}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False

def process_dataset(json_path, input_dir, output_dir, result_input_dir, result_output_dir):
    """处理整个数据集"""
    os.makedirs(result_input_dir, exist_ok=True)
    os.makedirs(result_output_dir, exist_ok=True)
    
    data = load_json_data(json_path)
    print(f"加载了 {len(data)} 条数据")
    
    success_count = 0
    fail_count = 0
    
    with tqdm(total=len(data), desc="处理图片", unit="张", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for idx, item in enumerate(data):
            try:
                input_filename = item['input'].split('/')[-1]
                output_filename = item['output'].split('/')[-1]
                instruction = item['instruction']
                input_description = item.get('input_description', None)  # 获取图片描述（可选）
                
                input_img_path = os.path.join(input_dir, input_filename)
                output_img_path = os.path.join(output_dir, output_filename)
                
                pbar.set_description(f"处理图片 {input_filename[:20]}...")
                
                if not os.path.exists(input_img_path):
                    tqdm.write(f"❌ 输入图片不存在: {input_filename}")
                    fail_count += 1
                    pbar.update(1)
                    continue
                
                output_exists = os.path.exists(output_img_path)
                if not output_exists:
                    tqdm.write(f"⚠️  输出图片不存在: {output_filename}")
                
                result_input_path = os.path.join(result_input_dir, input_filename)
                result_output_path = os.path.join(result_output_dir, output_filename)
                
                # 添加标记和文本到输入图片
                text_processing_success = False
                try:
                    text_processing_success = add_marker_and_text_to_image(input_img_path, instruction, result_input_path, input_description)
                except Exception as text_error:
                    tqdm.write(f"❌ 处理异常: {input_filename}, 错误: {str(text_error)}")
                    text_processing_success = False
                
                # 复制output图片
                import shutil
                output_copy_success = True
                if output_exists:
                    try:
                        shutil.copy2(output_img_path, result_output_path)
                    except Exception as copy_error:
                        tqdm.write(f"⚠️  复制输出图片失败: {output_filename}")
                        output_copy_success = False
                
                if text_processing_success:
                    if not os.path.exists(result_input_path):
                        tqdm.write(f"❌ 处理后文件未生成: {input_filename}")
                        fail_count += 1
                    else:
                        if output_exists and output_copy_success:
                            tqdm.write(f"✅ 成功: {input_filename}")
                        else:
                            tqdm.write(f"✅ 成功(仅输入): {input_filename}")
                        success_count += 1
                        
                        if (idx + 1) % 100 == 0:
                            tqdm.write(f"📊 进度: 成功 {success_count}, 失败 {fail_count}")
                else:
                    fail_count += 1
                    tqdm.write(f"❌ 处理失败: {input_filename}")
                    
                    if os.path.exists(result_input_path):
                        try:
                            os.remove(result_input_path)
                        except:
                            pass
                    if os.path.exists(result_output_path):
                        try:
                            os.remove(result_output_path)
                        except:
                            pass
                
            except Exception as e:
                fail_count += 1
                tqdm.write(f"❌ 处理第 {idx+1} 条数据时出错: {str(e)}")
            
            pbar.update(1)
            pbar.set_postfix({
                '成功': success_count,
                '失败': fail_count,
                '成功率': f"{success_count/(success_count+fail_count)*100:.1f}%" if (success_count+fail_count) > 0 else "0%"
            })
    
    print(f"\n处理完成!")
    print(f"成功处理: {success_count} 张图片")
    print(f"处理失败: {fail_count} 张图片")

def test_single_image():
    """测试单张图片处理功能"""
    # 测试路径
    base_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/GPT-Image-Edit/ultraedit/gpt-edit/ultraedit/ultraedit/replace"
    json_path = os.path.join(base_dir, "metadata/ultraedit_replace.json")
    input_dir = os.path.join(base_dir, "input")
    
    # 测试输出
    test_output_dir = "/storage/v-jinpewang/lab_folder/junchao/Data_scripts/visual_marker/test/replace_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 加载JSON获取第一条数据
    data = load_json_data(json_path)
    if not data:
        print("❌ 没有找到测试数据")
        return
    
    # 测试前10张图片
    print("=" * 60)
    print("测试模式 - 处理前10张图片")
    print("=" * 60)
    
    for i, item in enumerate(data[:10]):
        print(f"\n测试图片 {i+1}/10:")
        input_filename = item['input'].split('/')[-1]
        instruction = item['instruction']
        input_description = item.get('input_description', None)  # 获取图片描述（可选）
        input_img_path = os.path.join(input_dir, input_filename)
        output_path = os.path.join(test_output_dir, f"test_{i+1}_{input_filename}")
        
        print(f"  文件: {input_filename}")
        print(f"  指令: {instruction}")
        if input_description:
            print(f"  描述: {input_description}")
        
        if not os.path.exists(input_img_path):
            print(f"  ❌ 图片不存在")
            continue
        
        success = add_marker_and_text_to_image(input_img_path, instruction, output_path, input_description)
        if success:
            print(f"  ✅ 成功保存到: {output_path}")
        else:
            print(f"  ❌ 处理失败")
    
    print(f"\n测试完成! 结果保存在: {test_output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='UltraEdit Replace数据集处理 - 添加物体标记和文本')
    parser.add_argument('--test', action='store_true', help='测试模式：只处理前3张图片')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的图片数量')
    args = parser.parse_args()
    
    if args.test:
        test_single_image()
        return
    
    # 设置路径 - replace数据集
    base_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/GPT-Image-Edit/ultraedit/gpt-edit/ultraedit/ultraedit/replace"
    json_path = os.path.join(base_dir, "metadata/ultraedit_replace.json")
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    
    # 结果保存路径
    result_input_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_with_marker/replace/ultraedit/input"
    result_output_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_with_marker/replace/ultraedit/output"
    
    print("=" * 60)
    print("UltraEdit Replace 数据集处理")
    print("=" * 60)
    print(f"JSON文件: {json_path}")
    print(f"输入图片目录: {input_dir}")
    print(f"输出图片目录: {output_dir}")
    print(f"处理后输入图片保存到: {result_input_dir}")
    print(f"处理后输出图片保存到: {result_output_dir}")
    if args.limit:
        print(f"限制处理数量: {args.limit} 张")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 错误: JSON文件不存在: {json_path}")
        return
    
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        return
    
    # 如果设置了limit,修改数据处理函数
    if args.limit:
        # 先加载数据
        data = load_json_data(json_path)
        # 只处理前N条
        limited_data = data[:args.limit]
        # 保存临时JSON
        temp_json = "/tmp/temp_ultraedit_replace.json"
        with open(temp_json, 'w', encoding='utf-8') as f:
            json.dump(limited_data, f, ensure_ascii=False, indent=2)
        process_dataset(temp_json, input_dir, output_dir, result_input_dir, result_output_dir)
    else:
        process_dataset(json_path, input_dir, output_dir, result_input_dir, result_output_dir)

if __name__ == "__main__":
    main()
