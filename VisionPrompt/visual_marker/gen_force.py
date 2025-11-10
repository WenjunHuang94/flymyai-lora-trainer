"""从物体中心向外绘制箭头，箭头三角形在远端"""
import os
import re
import math
from PIL import Image, ImageDraw
import sys

# 导入Qwen相关模块
try:
    from prompt_utils import edit_api
except ImportError:
    print("警告: 无法导入prompt_utils，请确保该模块在路径中")
    edit_api = None


def get_object_position_from_qwen(image, object_name, img_width, img_height):
    """使用Qwen VL API获取物体在图片中的位置
    
    Args:
        image: PIL图片对象
        object_name: 要定位的物体名称
        img_width, img_height: 图片尺寸
    
    返回: (center_x, center_y, bbox) 或 None
    bbox格式: (x1, y1, x2, y2) 像素坐标
    """
    if edit_api is None:
        print("  ⚠️  Qwen API不可用")
        return None
    
    try:
        prompt = f"""The size of this image is {img_width}*{img_height}.
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
        print(f"  [DEBUG] API返回: {response[:150]}")
        
        # 尝试解析 (x1,y1) (x2,y2) 格式
        coord_pattern = r'\((\d+),\s*(\d+)\)'
        matches = re.findall(coord_pattern, response)
        
        if len(matches) >= 2:
            # 找到两个坐标点，假设是左上和右下
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
            
            print(f"  ✓ 提取到边界框: ({x1},{y1}) - ({x2},{y2})")
            return center_x, center_y, (x1, y1, x2, y2)
        
        elif len(matches) == 1:
            # 只找到一个坐标，假设是中心点
            center_x, center_y = int(matches[0][0]), int(matches[0][1])
            center_x = max(0, min(center_x, img_width - 1))
            center_y = max(0, min(center_y, img_height - 1))
            
            # 估算一个边界框（假设物体占图片的1/6）
            box_size = min(img_width, img_height) // 6
            x1 = max(0, center_x - box_size // 2)
            y1 = max(0, center_y - box_size // 2)
            x2 = min(img_width - 1, center_x + box_size // 2)
            y2 = min(img_height - 1, center_y + box_size // 2)
            
            print(f"  ✓ 提取到中心点: ({center_x},{center_y})，估算边界框")
            return center_x, center_y, (x1, y1, x2, y2)
        
        # 尝试解析 x1=, y1=, x2=, y2= 格式
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
            
            print(f"  ✓ 提取到边界框(x=y=格式): ({x1},{y1}) - ({x2},{y2})")
            return center_x, center_y, (x1, y1, x2, y2)
        
        # 如果都没匹配到，返回None
        print(f"  ⚠️  无法从API响应中提取坐标")
        return None
        
    except Exception as e:
        print(f"  ⚠️  Qwen VL API调用失败: {str(e)}")
        return None


def draw_outward_arrow(draw, start_x, start_y, end_x, end_y, img_width, img_height, arrow_color=(255, 0, 0)):
    """从起点向终点绘制箭头，箭头三角形在终点（远端）
    
    这与ultraedit_replace.py中的箭头相反：
    - 原版：箭头指向物体，三角形在物体端
    - 现在：箭头从物体中心向外，三角形在远端
    
    Args:
        draw: ImageDraw对象
        start_x, start_y: 箭头起点（物体中心）
        end_x, end_y: 箭头终点（远端，三角形位置）
        img_width, img_height: 图片尺寸（用于动态调整箭头大小）
        arrow_color: 箭头颜色
    """
    # 根据图片尺寸动态计算箭头参数
    min_dimension = min(img_width, img_height)
    arrow_width = max(2, int(min_dimension * 0.004))  # 箭头线宽度，约为最小边的0.4%
    arrow_length = max(10, int(min_dimension * 0.02))  # 箭头三角形长度，约为最小边的2%
    outline_width = max(1, int(arrow_width * 0.5))  # 描边宽度
    
    # 绘制箭头线（带白色边框使其更醒目）
    outline_color = (255, 255, 255)
    draw.line([(start_x, start_y), (end_x, end_y)], fill=outline_color, width=arrow_width + outline_width * 2)
    draw.line([(start_x, start_y), (end_x, end_y)], fill=arrow_color, width=arrow_width)
    
    # 计算箭头方向角度
    angle = math.atan2(end_y - start_y, end_x - start_x)
    arrow_angle = math.pi / 6  # 30度，箭头张角
    
    # 计算箭头三角形的两个边点
    # 注意：三角形在终点（end_x, end_y），向后延伸
    left_x = end_x - arrow_length * math.cos(angle - arrow_angle)
    left_y = end_y - arrow_length * math.sin(angle - arrow_angle)
    right_x = end_x - arrow_length * math.cos(angle + arrow_angle)
    right_y = end_y - arrow_length * math.sin(angle + arrow_angle)
    
    # 绘制箭头头部（三角形）
    arrow_head = [(end_x, end_y), (left_x, left_y), (right_x, right_y)]
    
    # 白色描边（根据图片尺寸动态调整描边范围）
    outline_range = max(1, int(outline_width))
    for i in range(-outline_range, outline_range + 1):
        for j in range(-outline_range, outline_range + 1):
            if i != 0 or j != 0:
                offset_head = [(x + i, y + j) for x, y in arrow_head]
                draw.polygon(offset_head, fill=outline_color)
    
    # 红色箭头三角形
    draw.polygon(arrow_head, fill=arrow_color)


def calculate_arrow_endpoint(center_x, center_y, bbox, img_width, img_height):
    """计算箭头终点，根据图片尺寸动态调整箭头长度，确保不超出图片边界
    
    Args:
        center_x, center_y: 物体中心点（箭头起点）
        bbox: 物体边界框 (x1, y1, x2, y2)
        img_width, img_height: 图片尺寸
    
    Returns:
        (end_x, end_y): 箭头终点坐标
    """
    import random
    
    # 安全边距
    margin = 30
    
    # 动态计算箭头长度：使用图片较小边的35-45%
    base_length = min(img_width, img_height) * 0.4
    # 添加一些随机性，让箭头长度在基准长度的85%-100%之间
    arrow_length = base_length * random.uniform(0.85, 1.0)
    
    print(f"  动态计算箭头长度: {arrow_length:.1f} 像素 (基于图片尺寸 {img_width}x{img_height})")
    
    # 生成随机角度（0-360度）
    angle = random.uniform(0, 2 * math.pi)
    
    # 计算该角度方向上从中心到边界的最大可用距离
    # 考虑4个边界：左、右、上、下
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    # 计算在当前角度下，从中心到边界的最大距离
    max_distances = []
    
    # 到右边界的距离
    if cos_angle > 0:
        max_distances.append((img_width - margin - center_x) / cos_angle)
    # 到左边界的距离
    elif cos_angle < 0:
        max_distances.append((margin - center_x) / cos_angle)
    
    # 到下边界的距离
    if sin_angle > 0:
        max_distances.append((img_height - margin - center_y) / sin_angle)
    # 到上边界的距离
    elif sin_angle < 0:
        max_distances.append((margin - center_y) / sin_angle)
    
    # 获取最小的正距离（即在该方向上能走的最远距离）
    valid_distances = [d for d in max_distances if d > 0]
    if valid_distances:
        max_distance = min(valid_distances)
        # 使用期望长度和最大可用距离中的较小值
        actual_length = min(arrow_length, max_distance * 0.9)  # 留10%余量
    else:
        # 如果计算失败，使用较短的默认值
        actual_length = min(arrow_length, min(img_width, img_height) * 0.3)
    
    # 计算终点位置
    end_x = center_x + actual_length * cos_angle
    end_y = center_y + actual_length * sin_angle
    
    # 再次确保终点在边界内（双重保险）
    end_x = max(margin, min(end_x, img_width - margin))
    end_y = max(margin, min(end_y, img_height - margin))
    
    # 计算实际箭头长度
    final_length = math.sqrt((end_x - center_x)**2 + (end_y - center_y)**2)
    print(f"  实际箭头长度: {final_length:.1f} 像素")
    
    return int(end_x), int(end_y)


def add_force_arrow_to_image(image_path, object_name, output_path):
    """在图片上添加从物体中心向外的箭头（箭头长度根据图片尺寸动态调整）
    
    Args:
        image_path: 输入图片路径
        object_name: 要定位的物体名称（如"ball"）
        output_path: 输出图片路径
    
    Returns:
        bool: 处理是否成功
    """
    try:
        # 打开图片
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        print(f"图片尺寸: {width} x {height}")
        
        # 1. 调用API获取物体位置
        print(f"正在定位 '{object_name}'...")
        object_position = get_object_position_from_qwen(img, object_name, width, height)
        
        if not object_position:
            print("❌ 无法定位物体")
            return False
        
        center_x, center_y, bbox = object_position
        print(f"✓ 物体中心: ({center_x}, {center_y})")
        print(f"✓ 物体边界框: {bbox}")
        
        # 2. 计算箭头终点（从中心向外，长度自动根据图片尺寸调整）
        end_x, end_y = calculate_arrow_endpoint(center_x, center_y, bbox, width, height)
        print(f"✓ 箭头终点: ({end_x}, {end_y})")
        
        # 3. 绘制箭头
        draw = ImageDraw.Draw(img)
        draw_outward_arrow(draw, center_x, center_y, end_x, end_y, 
                          width, height, arrow_color=(255, 0, 0))
        print(f"✓ 箭头已绘制")
        
        # 4. 保存图片
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        img.save(output_path)
        print(f"✅ 图片已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理图片时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = os.path.join(script_dir, "imgs/balls.png")
    output_image = os.path.join(script_dir, "imgs/balls_with_force_arrow.png")
    
    print("=" * 60)
    print("生成力场箭头 - 从物体中心向外")
    print("=" * 60)
    print(f"输入图片: {input_image}")
    print(f"输出图片: {output_image}")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_image):
        print(f"❌ 输入图片不存在: {input_image}")
        return
    
    # 处理图片（箭头长度会根据图片尺寸自动调整）
    success = add_force_arrow_to_image(
        image_path=input_image,
        object_name="ball",  # 定位"球"
        output_path=output_image
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 处理完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 处理失败")
        print("=" * 60)


if __name__ == "__main__":
    main()

