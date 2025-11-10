'''20step/9min
This file is used to add objects to images and save both with textbox (bounding box + label) and without textbox (text only)
Creates two folder structures:
- with_textbox: input (original with bbox), output (edited)
- wo_textbox: input (original with text only), output (edited)
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import random
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from prompt_utils import polish_edit_prompt, edit_api
import glob
from tqdm import tqdm
import re

from multi3_infer_plus import MyQwenImageEditPipeline, MultiGPUTransformer


class ObjectPromptGenerator:
    """物体生成prompt的类,参考add_text_boxes_to_images.py的generate_unique_text函数"""
    
    def __init__(self):
        self.used_texts = set()  # 记录已使用的文本,确保唯一性
        
        # 从JSON文件加载物体词库
        self.objects = self._load_objects_from_json()
        
        # 颜色列表 - 简化为短词汇,避免文本过长
        self.colors = [
            # 基础颜色
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
            "black", "white", "gray", "silver", "gold",
            
            # 扩展颜色 - 只保留短词汇
            "cyan", "lime", "navy", "olive", "teal", "aqua", "coral", "violet", 
            "tan", "cream", "ivory", "bronze", "copper",
                        
            # 特殊色调
            "metallic", "glossy", "matte", "neon", "fluorescent", "pastel", "vintage",
            "rainbow", "multicolored", "transparent", "crystal", "frosted"
        ]
        
        # 位置描述词 - 简化为短词汇,避免文本过长
        self.positions = [
            # 基础位置
            "here", "there",
                        
            # 简化的空间位置
            "nearby",
            
            # 相对方向
            "close by", "nearby", "somewhere",
            
            # 简单位置
            "", "in position", "in place"
        ]
        
        # 动作类型 - 只保留添加物体
        self.action_types = ["add"]
    
    def _load_objects_from_json(self):
        """从JSON文件加载物体词库"""
        json_file = "/storage/v-jinpewang/lab_folder/junchao/data/objects.json"
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    objects = json.load(f)
                
                if isinstance(objects, list) and objects:
                    # 过滤掉过长的物体名称,保持文本简洁
                    filtered_objects = [obj for obj in objects if len(obj) <= 15]
                    print(f"✅ 成功加载物体词库: {json_file} ({len(filtered_objects)} 个物体,已过滤长名称)")
                    return filtered_objects
                else:
                    print(f"❌ JSON文件格式错误: {json_file}")
                    
            except json.JSONDecodeError:
                print(f"❌ JSON文件解析错误: {json_file}")
            except Exception as e:
                print(f"❌ 加载文件失败: {json_file} - {e}")
        else:
            print(f"❌ 未找到物体词库文件: {json_file}")
    
    def generate_unique_object_prompt(self, max_length: int = 25):
        """生成唯一的物体添加prompt,返回(prompt, object_name)"""
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            obj1 = random.choice(self.objects)
            color = random.choice(self.colors)
            
            # 基础动词（高频）
            base_verbs = ["add", "put", "place", "draw", "create", "insert"]
            
            # 构建不同复杂度的模板组
            simple_templates = []  # 简单模板（高频）
            medium_templates = []  # 中等模板（中频）
            
            # 简单模板：动词 + 物体（70%概率）
            for verb in base_verbs:
                simple_templates.append(f"{verb} a {obj1}")
            
            # 中等模板：动词 + 颜色 + 物体（30%概率）
            for verb in base_verbs:
                medium_templates.append(f"{verb} a {color} {obj1}")
            
            # 按权重随机选择模板类型
            template_choice = random.random()
            if template_choice < 0.7:  # 70%选择简单模板
                templates = simple_templates
            else:  # 30%选择中等模板
                templates = medium_templates
            
            # 从选定的模板组中随机选择一个模板
            if templates:
                template = random.choice(templates)
                # 清理多余空格
                text = ' '.join(template.split())
                
                # 检查长度限制和唯一性
                if len(text) <= max_length and text not in self.used_texts:
                    self.used_texts.add(text)
                    return text, obj1  # 返回prompt和物体名称
            
            attempts += 1
        
        # 如果生成了太多重复,使用最短的模板
        obj1 = random.choice(self.objects)
        fallback_text = f"add {obj1}"
        if len(fallback_text) > max_length:
            # 如果物体名称太长,截断
            fallback_text = f"add {obj1[:max_length-4]}"
        
        unique_text = f"{fallback_text}#{random.randint(10, 99)}"
        self.used_texts.add(unique_text)
        return unique_text, obj1  # 返回prompt和物体名称


class TextBoxDrawer:
    """文本框绘制类,参考add_text_boxes_to_images.py"""
    
    def __init__(self):
        pass
    
    def get_average_color(self, image, x: int, y: int, width: int, height: int):
        """获取指定区域的平均颜色"""
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
    
    def get_contrasting_color(self, background_color):
        """根据背景色选择对比度最高的字体颜色"""
        # 简化的候选颜色
        colors = [(0, 0, 0), (255, 255, 255), (255, 255, 0), (255, 0, 255), 
                 (0, 255, 255), (255, 100, 0), (0, 255, 0), (255, 0, 0)]
        
        # 简化的亮度计算和对比度选择
        bg_luminance = 0.299 * background_color[0] + 0.587 * background_color[1] + 0.114 * background_color[2]
        
        best_color = (0, 0, 0)
        max_contrast = 0
        
        for color in colors:
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            contrast = abs(bg_luminance - luminance)
            if contrast > max_contrast:
                max_contrast = contrast
                best_color = color
        
        return best_color
    
    def _load_font(self, size: int):
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
        
        # 如果都失败,使用默认字体
        return ImageFont.load_default()
    
    def adjust_font_size_with_actual_measurement(self, text: str, img_width: int, img_height: int, 
                                               initial_size: int = 32, min_size: int = 8):
        """使用实际字体测量动态调整字体大小"""
        # 创建临时图像用于测量
        temp_img = Image.new('RGB', (100, 100))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # 计算可用空间,预留安全边距
        safety_margin = 40
        max_available_width = img_width - safety_margin
        max_available_height = img_height - safety_margin
        
        # 如果图片太小,进一步减少安全边距
        if max_available_width < 100 or max_available_height < 50:
            safety_margin = 20
            max_available_width = img_width - safety_margin
            max_available_height = img_height - safety_margin
        
        # 根据文本长度动态调整初始字体大小
        text_length = len(text)
        
        if text_length <= 8:  # 短文本 - 大字体
            dynamic_initial_size = min(48, initial_size + 16)
            dynamic_min_size = max(16, min_size + 8)
        elif text_length <= 15:  # 中等文本 - 标准字体
            dynamic_initial_size = initial_size
            dynamic_min_size = min_size + 4
        elif text_length <= 25:  # 较长文本 - 小字体
            dynamic_initial_size = max(24, initial_size - 8)
            dynamic_min_size = min_size + 2
        else:  # 很长文本 - 最小字体
            dynamic_initial_size = max(16, initial_size - 16)
            dynamic_min_size = min_size
        
        # 进一步根据图片大小调整字体范围
        img_area = img_width * img_height
        if img_area < 200000:  # 小图片
            dynamic_initial_size = int(dynamic_initial_size * 0.8)
            dynamic_min_size = max(6, int(dynamic_min_size * 0.8))
        elif img_area > 1000000:  # 大图片
            dynamic_initial_size = int(dynamic_initial_size * 1.2)
            dynamic_min_size = int(dynamic_min_size * 1.1)
        
        # 确保字体大小在合理范围内
        dynamic_initial_size = max(dynamic_min_size, min(64, dynamic_initial_size))
        
        for font_size in range(dynamic_initial_size, dynamic_min_size - 1, -1):
            # 加载字体
            font = self._load_font(font_size)
            
            # 实际测量文本尺寸，使用精确的边界框
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            actual_text_width = bbox[2] - bbox[0]
            actual_text_height = bbox[3] - bbox[1]
            
            # 动态调整padding,根据字体大小和文本长度
            if font_size <= 12:
                padding = max(8, font_size // 2)  # 增加最小padding确保有足够空间
            elif font_size <= 20:
                padding = max(10, font_size // 2)
            else:
                padding = max(12, font_size // 3)
            
            # 长文本使用更少的padding以节省空间，但保持最小值
            if text_length > 20:
                padding = max(6, padding - 2)
            
            # 为了确保文本完全居中，给高度额外增加一些空间
            # 这是因为某些字体的ascent/descent可能不对称
            extra_height_margin = max(2, font_size // 8)
            
            box_width = actual_text_width + 2 * padding
            box_height = actual_text_height + 2 * padding + extra_height_margin
            
            # 严格检查文本框是否能完全放入图片
            if box_width <= max_available_width and box_height <= max_available_height:
                return font_size, box_width, box_height, actual_text_width, actual_text_height, font, padding
        
        # 如果所有字体都太大,使用最小字体并强制适应
        font = self._load_font(dynamic_min_size)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        actual_text_width = bbox[2] - bbox[0]
        actual_text_height = bbox[3] - bbox[1]
        
        # 最小padding，但要确保有足够空间用于居中
        min_padding = max(4, dynamic_min_size // 4)
        extra_height_margin = max(2, dynamic_min_size // 8)
        
        forced_box_width = min(actual_text_width + 2 * min_padding, max_available_width)
        forced_box_height = min(actual_text_height + 2 * min_padding + extra_height_margin, max_available_height)
        
        return dynamic_min_size, forced_box_width, forced_box_height, actual_text_width, actual_text_height, font, min_padding
    

    def draw_bounding_box_with_text(self, image: Image.Image, x1: int, y1: int, x2: int, y2: int, text: str):
        """绘制边界框并添加文字标签"""
        # 在框的周边动态识别位置来写文字!!!

        img_copy = image.copy()
        
        # 确保坐标在图片范围内
        x1 = max(0, min(x1, image.width - 1))
        y1 = max(0, min(y1, image.height - 1))
        x2 = max(x1 + 1, min(x2, image.width))
        y2 = max(y1 + 1, min(y2, image.height))
        
        # 创建带透明度的覆盖层
        overlay = Image.new('RGBA', img_copy.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 绘制物体边框（红色边框，直接使用边界框坐标）
        border_color = (255, 0, 0, 255)
        overlay_draw.rectangle([x1, y1, x2, y2], fill=None, outline=border_color, width=3)
        
        # 添加文字标签
        if text:
            # 计算合适的字体大小（基于图片尺寸，让文字更大）
            min_font_size = 28  # 提高最小字体大小
            base_font_size = max(min_font_size, min(image.width, image.height) // 20)  # 调整比例让字体更大
            
            try:
                # 使用指定的字体文件
                from PIL import ImageFont
                try:
                    # 使用指定的Times New Roman字体
                    font = ImageFont.truetype("/storage/v-jinpewang/lab_folder/junchao/data/Times_New_Roman.ttf", base_font_size)
                    print(f"成功加载指定字体: Times_New_Roman.ttf，大小: {base_font_size}")
                except (OSError, IOError) as e:
                    print(f"❌ 无法加载指定字体文件: {e}")
                    try:
                        # 备用方案：尝试加载系统字体
                        font = ImageFont.truetype("arial.ttf", base_font_size)
                        print("使用备用字体: arial.ttf")
                    except (OSError, IOError):
                        try:
                            font = ImageFont.truetype("DejaVuSans.ttf", base_font_size)
                            print("使用备用字体: DejaVuSans.ttf")
                        except (OSError, IOError):
                            # 使用PIL默认字体
                            font = ImageFont.load_default()
                            print("使用PIL默认字体")
            except ImportError:
                # 如果ImageFont不可用，使用默认字体
                font = None
                print("ImageFont不可用，使用默认字体")
            
            # 获取文字尺寸
            if font:
                bbox = overlay_draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # 使用默认字体时的估算
                text_width = len(text) * (base_font_size // 2)
                text_height = base_font_size
            
            # 确定文字位置：根据边界框位置智能选择
            margin = 5  # 文字与边界框的间距
            
            # 检查边界框是否靠近图片顶部或底部
            near_top = y1 < text_height + margin + 20  # 如果边界框顶部距离图片顶部太近
            near_bottom = y2 > image.height - text_height - margin - 20  # 如果边界框底部距离图片底部太近
            
            # 计算框的中心和宽度
            box_center_x = (x1 + x2) // 2
            box_width = x2 - x1
            
            # 先确定垂直位置（上方、下方或内部）
            if near_top and not near_bottom:
                # 边界框靠近顶部，文字放在边界框下方
                text_y = y2 + margin
                position_desc = "边界框下方"
            elif near_bottom and not near_top:
                # 边界框靠近底部，文字放在边界框上方
                text_y = y1 - text_height - margin
                position_desc = "边界框上方"
            elif near_top and near_bottom:
                # 边界框占据了大部分垂直空间，文字放在边界框内部顶部
                text_y = y1 + margin
                position_desc = "边界框内部顶部"
            else:
                # 默认情况：文字放在边界框上方
                text_y = y1 - text_height - margin
                position_desc = "边界框上方（默认）"
            
            # 计算水平位置：尝试让文本居中对齐框，或者根据空间智能调整
            # 首选：文本中心对齐框中心
            text_x = box_center_x - text_width // 2
            
            # 检查是否超出左边界
            if text_x < 0:
                text_x = 0  # 贴近左边界
                adjustment = "左边界对齐"
            # 检查是否超出右边界
            elif text_x + text_width > image.width:
                text_x = image.width - text_width  # 贴近右边界
                adjustment = "右边界对齐"
            else:
                adjustment = "居中对齐框"
            
            # 进一步优化：如果文本距离框太远，适当调整位置使其靠近框
            # 定义"太远"的阈值：文本与框之间的距离超过框宽的一半
            max_distance = box_width * 0.5
            
            # 如果文本完全在框的左侧且距离太远，右移靠近框
            if text_x + text_width < x1 and (x1 - (text_x + text_width)) > max_distance:
                text_x = max(0, x1 - text_width - 10)  # 靠近框左边，留10像素间距
                adjustment = "左移靠近框"
            # 如果文本完全在框的右侧且距离太远，左移靠近框
            elif text_x > x2 and (text_x - x2) > max_distance:
                text_x = min(image.width - text_width, x2 + 10)  # 靠近框右边，留10像素间距
                adjustment = "右移靠近框"
            
            # 最终边界检查，确保不超出图片
            text_x = max(0, min(text_x, image.width - text_width))
            text_y = max(0, min(text_y, image.height - text_height))
            
            print(f"文字位置：{position_desc}，水平调整：{adjustment} ({text_x}, {text_y})")
            
            # 计算文字背景区域
            bg_padding = 3
            bg_x1 = max(0, text_x - bg_padding)
            bg_y1 = max(0, text_y - bg_padding)
            bg_x2 = min(image.width, text_x + text_width + bg_padding)
            bg_y2 = min(image.height, text_y + text_height + bg_padding)
            bg_width = bg_x2 - bg_x1
            bg_height = bg_y2 - bg_y1
            
            # 检测背景色并选择对比色
            background_color = self.get_average_color(image, bg_x1, bg_y1, bg_width, bg_height)
            text_color_rgb = self.get_contrasting_color(background_color)
            text_color_rgba = (*text_color_rgb, 255)
            
            print(f"背景颜色: {background_color}, 选择的文字颜色: {text_color_rgb}")
            
            # 根据文字颜色选择描边颜色
            outline_color = (0, 0, 0, 255) if text_color_rgb != (0, 0, 0) else (255, 255, 255, 255)
            
            # 根据字体大小调整描边效果
            if base_font_size < 16:
                outline_width = 1
                # 只绘制4个方向的描边,减少粘连
                outline_positions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            else:
                outline_width = 2
                # 大字体使用8方向描边
                outline_positions = [(-outline_width, -outline_width), (-outline_width, 0), (-outline_width, outline_width),
                                   (0, -outline_width), (0, outline_width),
                                   (outline_width, -outline_width), (outline_width, 0), (outline_width, outline_width)]
            
            # 绘制描边
            for dx, dy in outline_positions:
                overlay_draw.text((text_x + dx, text_y + dy), text, 
                                fill=outline_color, font=font)
            
            # 绘制主文字（使用动态选择的颜色）
            overlay_draw.text((text_x, text_y), text, fill=text_color_rgba, font=font)
            
            print(f"绘制文字: '{text}' 在位置 ({text_x}, {text_y}), 字体大小: {base_font_size}")
        
        # 合并图像
        img_copy = Image.alpha_composite(img_copy.convert('RGBA'), overlay).convert('RGB')
        
        print(f"绘制边界框: ({x1}, {y1}, {x2}, {y2})")
        
        return img_copy, {
            'bounding_box': (x1, y1, x2, y2),
            'text': text,
            'text_position': (text_x, text_y) if text else None,
            'background_color': background_color if text else None,
            'text_color': text_color_rgb if text else None,
            'font_size': base_font_size if text else None
        }
    
    def draw_text_only_at_center(self, image: Image.Image, text: str, center_x: int, center_y: int):
        """在指定中心位置只绘制文本（无边框）- 使用与draw_text_box_at_center完全相同的字体参数"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # 获取字体参数 - 与draw_text_box_at_center使用完全相同的逻辑
        font_size, box_width, box_height, text_width, text_height, font, padding = self.adjust_font_size_with_actual_measurement(
            text, image.width, image.height)
        
        # 计算文本框的左上角位置（以center_x, center_y为中心）
        x = center_x - box_width // 2
        y = center_y - box_height // 2
        
        # 确保文本框完全在图片内
        x = max(0, min(x, image.width - box_width))
        y = max(0, min(y, image.height - box_height))
        
        # 检测背景色 - 与原方法完全相同
        background_color = self.get_average_color(image, x, y, box_width, box_height)
        text_color = self.get_contrasting_color(background_color)
        text_color_rgba = (*text_color, 255)
        
        # 创建带透明度的覆盖层
        overlay = Image.new('RGBA', img_copy.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 计算文本在文本框中的精确居中位置 - 与原方法完全相同
        temp_bbox = overlay_draw.textbbox((0, 0), text, font=font)
        text_actual_width = temp_bbox[2] - temp_bbox[0]
        text_actual_height = temp_bbox[3] - temp_bbox[1]
        text_offset_y = temp_bbox[1]  # 文本顶部到基线的偏移
        
        # 水平居中：文本框中心 - 文本实际宽度的一半
        text_x = x + (box_width - text_actual_width) // 2
        
        # 垂直居中：考虑字体的ascent和descent，确保文本视觉上居中
        text_y = y + (box_height - text_actual_height) // 2 - text_offset_y
        
        # 根据字体大小调整描边效果 - 与原方法完全相同
        outline_color = (0, 0, 0, 255) if text_color != (0, 0, 0) else (255, 255, 255, 255)
        
        # 小字体使用更细的描边
        if font_size < 16:
            outline_width = 1
            # 只绘制4个方向的描边,减少粘连
            outline_positions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            outline_width = 2
            # 大字体使用8方向描边
            outline_positions = [(-outline_width, -outline_width), (-outline_width, 0), (-outline_width, outline_width),
                               (0, -outline_width), (0, outline_width),
                               (outline_width, -outline_width), (outline_width, 0), (outline_width, outline_width)]
        
        # 绘制描边
        for dx, dy in outline_positions:
            overlay_draw.text((text_x + dx, text_y + dy), text, 
                            fill=outline_color, font=font)
        
        # 绘制主文本
        overlay_draw.text((text_x, text_y), text, fill=text_color_rgba, font=font)
        
        # 合并图像
        img_copy = Image.alpha_composite(img_copy.convert('RGBA'), overlay).convert('RGB')
        
        return img_copy, {
            'text': text,
            'center_position': (center_x, center_y),
            'text_position': (text_x, text_y),
            'background_color': background_color,
            'text_color': text_color,
            'font_size': font_size
        }

# 初始化组件
prompt_generator = ObjectPromptGenerator()
text_drawer = TextBoxDrawer()


def get_object_position_and_size(edited_image, object_name, input_width, input_height):
    """调用API获取物体在编辑后图片中的位置坐标和大小信息"""
    position_prompt = f"The size of this image is {input_width}*{input_height}. Please locate the bounding box of the object {object_name} in the image."\
    "The bounding box should be a rectangle that tightly encloses the object.Output only the bounding box coordinates in the format: (x1, y1, x2, y2)."\
    "x1 and y1 represent the coordinates of the upper left corner of the rectangle.x2 and y2 represent the coordinates of the lower right corner of the rectangle."\
    "All coordinates are in pixels, and the origin (0, 0) is the upper left corner of the image.You only need to output the final answer."
    
    try:
        result = edit_api(position_prompt, [edited_image])
        print(f"API返回结果: {result}")
        
        # 首先尝试匹配新的 (x1, y1, x2, y2) 格式
        bbox_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        bbox_match = re.search(bbox_pattern, result)
        
        if bbox_match:
            x1, y1, x2, y2 = int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))
            
            # 确保坐标顺序正确 (x1 < x2, y1 < y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, input_width - 1))
            y1 = max(0, min(y1, input_height - 1))
            x2 = max(x1 + 1, min(x2, input_width))
            y2 = max(y1 + 1, min(y2, input_height))
            
            print(f"提取到的边界框: ({x1}, {y1}, {x2}, {y2})")
            return x1, y1, x2, y2
        
        else:
            # 回退到旧的解析方式作为兼容性处理
            # 匹配 Position: (x,y), Size: (width,height) 格式
            position_size_pattern = r'Position:\s*\((\d+),\s*(\d+)\),\s*Size:\s*\((\d+),\s*(\d+)\)'
            match = re.search(position_size_pattern, result, re.IGNORECASE)
            
            if match:
                x, y, width, height = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
                # 确保坐标在图片范围内
                x = max(0, min(x, input_width - 1))
                y = max(0, min(y, input_height - 1))
                # 确保大小合理
                width = max(10, min(width, input_width))
                height = max(10, min(height, input_height))
                # 转换为边界框坐标
                x1, y1 = x, y
                x2, y2 = x + width, y + height
                print(f"使用旧格式提取到的位置和大小: 位置({x}, {y}), 大小({width}, {height})")
                print(f"转换为边界框: ({x1}, {y1}, {x2}, {y2})")
                return x1, y1, x2, y2
            else:
                # 如果没有匹配到完整格式，尝试分别匹配位置和大小
                # 匹配位置 (数字,数字) 格式
                coord_pattern = r'\((\d+),\s*(\d+)\)'
                coord_matches = re.findall(coord_pattern, result)
                
                # 匹配大小相关的数字
                size_pattern = r'(?:width|w):\s*(\d+).*?(?:height|h):\s*(\d+)'
                size_match = re.search(size_pattern, result, re.IGNORECASE)
                
                if coord_matches and size_match:
                    x, y = int(coord_matches[0][0]), int(coord_matches[0][1])
                    width, height = int(size_match.group(1)), int(size_match.group(2))
                    # 确保数值在合理范围内
                    x = max(0, min(x, input_width - 1))
                    y = max(0, min(y, input_height - 1))
                    width = max(10, min(width, input_width))
                    height = max(10, min(height, input_height))
                    # 转换为边界框坐标
                    x1, y1 = x, y
                    x2, y2 = x + width, y + height
                    print(f"分别提取到的位置和大小: 位置({x}, {y}), 大小({width}, {height})")
                    print(f"转换为边界框: ({x1}, {y1}, {x2}, {y2})")
                    return x1, y1, x2, y2
                elif coord_matches:
                    # 只有位置信息，使用默认大小
                    x, y = int(coord_matches[0][0]), int(coord_matches[0][1])
                    x = max(0, min(x, input_width - 1))
                    y = max(0, min(y, input_height - 1))
                    # 使用默认大小（图片的10%）
                    default_width = max(50, input_width // 10)
                    default_height = max(50, input_height // 10)
                    # 转换为边界框坐标
                    x1, y1 = x, y
                    x2, y2 = x + default_width, y + default_height
                    print(f"只提取到位置: ({x}, {y})，使用默认大小({default_width}, {default_height})")
                    print(f"转换为边界框: ({x1}, {y1}, {x2}, {y2})")
                    return x1, y1, x2, y2
                else:
                    print("❌ 无法从API结果中提取位置和大小信息,使用默认值")
                    # 使用图片中心和默认大小
                    center_x, center_y = input_width // 2, input_height // 2
                    default_width = max(50, input_width // 10)
                    default_height = max(50, input_height // 10)
                    x1 = center_x - default_width // 2
                    y1 = center_y - default_height // 2
                    x2 = x1 + default_width
                    y2 = y1 + default_height
                    return x1, y1, x2, y2
                
    except Exception as e:
        print(f"❌ 调用API获取位置和大小时出错: {e},使用默认值")
        # 使用图片中心和默认大小
        center_x, center_y = input_width // 2, input_height // 2
        default_width = max(50, input_width // 10)
        default_height = max(50, input_height // 10)
        x1 = center_x - default_width // 2
        y1 = center_y - default_height // 2
        x2 = x1 + default_width
        y2 = y1 + default_height
        return x1, y1, x2, y2

def initialize_pipeline():
    """初始化图像编辑管道"""
    # 本地模型路径 - 请根据您的实际路径修改
    # local_model_path = "/storage/v-jinpewang/lab_folder/junchao/pretrained/Qwen-Image-Edit"
    pipeline = MyQwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16, cache_dir="/tmp")
    
    pipeline.transformer.to(torch.float32)
    pipeline.vae.to("cuda:0")
    pipeline.text_encoder.to("cuda:0")
    total_blocks = len(pipeline.transformer.transformer_blocks)
    gpu_split_points = [total_blocks//3, 2*total_blocks//3]  # 三等分
    pipeline.transformer = MultiGPUTransformer(pipeline.transformer, gpu_split_points)

    pipeline.set_progress_bar_config(disable=None)
    print("pipeline loaded")
    return pipeline


def process_single_image(pipeline, image_path, with_textbox_input_dir, with_textbox_output_dir, 
                         wo_textbox_input_dir, wo_textbox_output_dir, prompt_generator, text_drawer):
    """处理单张图片，生成两种版本（带边界框和只有文本）"""
    # 1. 读取原始图片
    original_image = Image.open(image_path).convert("RGB")
    input_width, input_height = original_image.size
    print(f"处理图片: {image_path}, 尺寸: {input_width} x {input_height}")
    
    # 2. 生成随机物体添加的prompt（不带位置信息）
    object_prompt, object_name = prompt_generator.generate_unique_object_prompt()
    print(f"生成的物体prompt: {object_prompt}")
    print(f"物体名称: {object_name}")
    
    # 3. 直接使用修改的prompt进行推理
    constraints = ".It is crucial to adhere to the following constraints: " \
            "1. Keep all the original elements and areas in the image completely unchanged. You can only add objects on this basis. " \
            "2. Maintain the original camera angle, perspective, and zoom level without any changes. Ensure the edit integrates naturally with the existing scene. "
    prompt = f"{object_prompt} {constraints}"
    polished_prompt = polish_edit_prompt(prompt, original_image)
    print(f"使用的推理prompt: {polished_prompt}")
    
    inputs = {
        "image": original_image,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 25,
        "guidance_scale": 1.0,
    }

    # 4. 执行推理生成编辑后的图片
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
    
    # 5. 将输出图片resize到与输入图片相同的尺寸，用于API调用
    resized_output_image = output_image.resize((input_width, input_height), Image.LANCZOS)
    
    # 6. 调用API获取物体在编辑后图片中的边界框坐标
    print("\n📞 调用API：获取边界框坐标...")
    x1, y1, x2, y2 = get_object_position_and_size(
        resized_output_image, object_name, input_width, input_height)
    print(f"获取到的边界框坐标: ({x1}, {y1}, {x2}, {y2})")

    # 7. 从边界框坐标计算物体中心位置（用于绘制文本）
    textbox_center_x = (x1 + x2) // 2
    textbox_center_y = (y1 + y2) // 2
    print(f"计算得到的中心位置: ({textbox_center_x}, {textbox_center_y})")

    # 8. 在原始图片上绘制边界框并添加文字标签
    print("\n🎨 绘制带边界框的图片...")
    with_bbox_image, textbox_info = text_drawer.draw_bounding_box_with_text(
        original_image, x1, y1, x2, y2, object_prompt)  # TODO： 为什么这里不用 polished_prompt？
    
    # 9. 在原始图片上只绘制文本（无边框）
    print("🎨 绘制只有文本的图片...")
    text_only_image, text_only_info = text_drawer.draw_text_only_at_center(
        original_image, object_prompt, textbox_center_x, textbox_center_y)

    # 10. 保存四张图片
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    
    print("\n💾 保存图片...")
    
    # 保存到 with_textbox 文件夹
    # input: 带边界框的原始图片
    with_bbox_input_path = os.path.join(with_textbox_input_dir, f"{name_without_ext}_textbox{ext}")
    if with_bbox_image.size != (512, 512):
        resized_with_bbox = with_bbox_image.resize((512, 512), Image.LANCZOS)
        resized_with_bbox.save(with_bbox_input_path)
    else:
        with_bbox_image.save(with_bbox_input_path)
    
    # output: 编辑后的图片
    with_bbox_output_path = os.path.join(with_textbox_output_dir, f"{name_without_ext}_edited{ext}")
    resized_output_image.save(with_bbox_output_path)
    
    # 保存到 wo_textbox 文件夹
    # input: 只有文本的原始图片
    text_only_input_path = os.path.join(wo_textbox_input_dir, f"{name_without_ext}_text_only{ext}")
    if text_only_image.size != (512, 512):
        resized_text_only = text_only_image.resize((512, 512), Image.LANCZOS)
        resized_text_only.save(text_only_input_path)
    else:
        text_only_image.save(text_only_input_path)
    
    # output: 编辑后的图片（与with_textbox的output相同）
    text_only_output_path = os.path.join(wo_textbox_output_dir, f"{name_without_ext}_edited{ext}")
    resized_output_image.save(text_only_output_path)
    
    # 返回处理结果
    result = {
        'input_file': base_name,
        'object_prompt': object_prompt,
        'object_name': object_name,
        'polished_prompt': polished_prompt,
        'bounding_box': (x1, y1, x2, y2),
        'text_center': (textbox_center_x, textbox_center_y),
        'with_bbox_input_path': with_bbox_input_path,
        'with_bbox_output_path': with_bbox_output_path,
        'text_only_input_path': text_only_input_path,
        'text_only_output_path': text_only_output_path,
        'textbox_info': textbox_info,
        'text_only_info': text_only_info,
        'success': True
    }
    
    print(f"✅ 成功处理: {base_name}")
    print(f"   📁 with_textbox/input: {with_bbox_input_path}")
    print(f"   📁 with_textbox/output: {with_bbox_output_path}")
    print(f"   📁 wo_textbox/input: {text_only_input_path}")
    print(f"   📁 wo_textbox/output: {text_only_output_path}")
    print(f"   🏷️  物体prompt: {object_prompt}")
    print(f"   🎯 物体名称: {object_name}")
    print(f"   📦 边界框: ({x1}, {y1}, {x2}, {y2})")
    print(f"   📍 中心位置: ({textbox_center_x}, {textbox_center_y})")
    
    return result


def print_results_summary(results, base_dir, results_path, max_images=None):
    """打印处理结果摘要"""
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful

    print(f"\n" + "="*80)
    print(f"处理完成!")
    print(f"="*80)
    if max_images is not None:
        print(f"设置限制: 最多处理 {max_images} 张图片")
    print(f"实际处理: {len(results)} 张图片")
    print(f"成功处理: {successful} 张图片")
    print(f"处理失败: {failed} 张图片")

    if successful > 0:
        print(f"\n📊 处理统计:")
        # 统计使用的物体类型
        object_counts = {}
        for result in results:
            if result.get('success', False):
                object_name = result.get('object_name', '')
                if object_name:
                    object_counts[object_name] = object_counts.get(object_name, 0) + 1
        
        if object_counts:
            print("   最常用的物体:")
            for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     - {obj}: {count} 次")
    print(f"="*80)


def main(max_images=None):
    """主函数：执行图像编辑处理流程
    
    Args:
        max_images (int, optional): 最大处理图片数量限制,None表示处理所有图片
    """
    # 初始化管道
    pipeline = initialize_pipeline()
    
    # 设置输入输出目录
    input_dir = "/storage/v-jinpewang/lab_folder/junchao/data/cluster_unprocessed_ultraedit/1/0/"
    
    # 创建基础输出目录
    base_output_dir = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen"
    
    # 创建文件夹结构
    with_textbox_dir = os.path.join(base_output_dir, "with_textbox")
    with_textbox_input_dir = os.path.join(with_textbox_dir, "input")
    with_textbox_output_dir = os.path.join(with_textbox_dir, "output")
    
    wo_textbox_dir = os.path.join(base_output_dir, "wo_textbox")
    wo_textbox_input_dir = os.path.join(wo_textbox_dir, "input")
    wo_textbox_output_dir = os.path.join(wo_textbox_dir, "output")
    
    # 创建所有必要的目录
    os.makedirs(with_textbox_input_dir, exist_ok=True)
    os.makedirs(with_textbox_output_dir, exist_ok=True)
    os.makedirs(wo_textbox_input_dir, exist_ok=True)
    os.makedirs(wo_textbox_output_dir, exist_ok=True)

    # 获取所有图片文件
    all_paths = sorted(glob.glob(os.path.join(input_dir, "*")))
    image_files = [
        p for p in all_paths
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in {".png", ".jpg", ".jpeg"}
    ]
    
    # 应用图片数量限制
    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]
        print(f"\n📊 设置处理图片数量限制: {max_images} 张")
    
    print(f"📁 找到图片文件: {len(image_files)} 张\n")

    # 保存处理结果
    results = []

    # 处理每张图片
    for i, image_path in enumerate(tqdm(image_files, desc="处理图片")):
        try:
            result = process_single_image(
                pipeline, image_path, 
                with_textbox_input_dir, with_textbox_output_dir,
                wo_textbox_input_dir, wo_textbox_output_dir,
                prompt_generator, text_drawer
            )
            results.append(result)
            
            # 检查是否达到限制（额外的安全检查）
            if max_images is not None and len(results) >= max_images:
                print(f"\n✅ 已达到处理图片数量限制 ({max_images} 张)，停止处理")
                break
                
        except Exception as e:
            print(f"\n❌ 处理图片 {image_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'input_file': os.path.basename(image_path),
                'error': str(e),
                'success': False
            }
            results.append(result)
            
            # 检查是否达到限制（包含错误的情况）
            if max_images is not None and len(results) >= max_images:
                print(f"\n✅ 已达到处理图片数量限制 ({max_images} 张)，停止处理")
                break

    # 保存处理结果到JSON文件
    results_path = os.path.join(base_output_dir, "processing_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印结果摘要
    print_results_summary(results, base_output_dir, results_path, max_images)


if __name__ == "__main__":
    # 修改这个数字来限制处理的图片数量，设置为None表示处理所有图片
    MAX_IMAGES_LIMIT = None
    
    main(max_images=MAX_IMAGES_LIMIT) 