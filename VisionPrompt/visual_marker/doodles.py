# '''
# 使用qwen-image-edit为图片添加doodle效果，然后转换为真实图片
# 为图中的elephant添加一个涂鸦绘制的帽子，然后将doodle转换为真实的帽子
# '''
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# import torch
# from PIL import Image
# from multi3_infer_plus import MyQwenImageEditPipeline, MultiGPUTransformer
# from prompt_utils import polish_edit_prompt


# def initialize_pipeline():
#     """初始化图像编辑管道"""
#     print("正在加载模型...")
#     pipeline = MyQwenImageEditPipeline.from_pretrained(
#         "Qwen/Qwen-Image-Edit-2509", 
#         torch_dtype=torch.bfloat16, 
#         cache_dir="/tmp"
#     )
    
#     # 设置模型精度和设备
#     pipeline.transformer.to(torch.float32)
#     pipeline.vae.to("cuda:0")
#     pipeline.text_encoder.to("cuda:0")
    
#     # 配置多GPU分布
#     total_blocks = len(pipeline.transformer.transformer_blocks)
#     gpu_split_points = [total_blocks//3, 2*total_blocks//3]  # 三等分
#     pipeline.transformer = MultiGPUTransformer(pipeline.transformer, gpu_split_points)
    
#     pipeline.set_progress_bar_config(disable=None)
#     print("✅ 模型加载完成")
#     return pipeline


# def add_doodle_hat(image_path, output_path, pipeline):
#     """为图中的elephant添加涂鸦帽子"""
#     # 读取图片
#     print(f"📖 读取图片: {image_path}")
#     original_image = Image.open(image_path).convert("RGB")
#     input_width, input_height = original_image.size
#     print(f"   图片尺寸: {input_width} x {input_height}")
    
#     # 设置prompt
#     original_prompt = "Add a hat on the elephant in the image using only simple, rough outline strokes. The hat should be drawn with minimal lines - just basic contours and shapes, like a quick sketch. No shading, no details, no filling - only simple line outlines. Keep everything else exactly the same as the original image, maintaining complete consistency except for the added hat outline."
#     print(f"📝 原始编辑指令: {original_prompt}")
    
#     # 使用polish_edit_prompt润色prompt
#     polished_prompt = polish_edit_prompt(original_prompt, original_image)
#     print(f"✨ 润色后的指令: {polished_prompt}")
    
#     # 准备推理参数
#     inputs = {
#         "image": original_image,
#         "prompt": polished_prompt,
#         "generator": torch.Generator(device="cuda").manual_seed(0),
#         "true_cfg_scale": 4.0,
#         "negative_prompt": " ",
#         "num_inference_steps": 25,  # 参考add_with_textbox.py的设置
#         "guidance_scale": 1.0,
#     }
    
#     # 执行推理
#     print("🎨 开始生成...")
#     with torch.inference_mode():
#         output = pipeline(**inputs)
#         output_image = output.images[0]
    
#     # 将输出图片resize到与输入图片相同的尺寸
#     resized_output_image = output_image.resize((input_width, input_height), Image.LANCZOS)
    
#     # 保存结果
#     resized_output_image.save(output_path)
#     print(f"✅ 图片已保存: {output_path}")
    
#     return resized_output_image


# def convert_doodle_to_realistic(doodle_image_path, output_path, pipeline):
#     """将doodle图片转换为真实的图片"""
#     # 读取doodle图片
#     print(f"📖 读取doodle图片: {doodle_image_path}")
#     doodle_image = Image.open(doodle_image_path).convert("RGB")
#     input_width, input_height = doodle_image.size
#     print(f"   图片尺寸: {input_width} x {input_height}")
    
#     # 设置prompt - 将doodle转为真实
#     original_prompt = "Convert the doodle-style hat into a realistic, photorealistic hat. Keep everything else exactly the same as the original, maintaining complete consistency."
#     print(f"📝 原始编辑指令: {original_prompt}")
    
#     # 使用polish_edit_prompt润色prompt
#     polished_prompt = polish_edit_prompt(original_prompt, doodle_image)
#     print(f"✨ 润色后的指令: {polished_prompt}")
    
#     # 准备推理参数
#     inputs = {
#         "image": doodle_image,
#         "prompt": polished_prompt,
#         "generator": torch.Generator(device="cuda").manual_seed(0),
#         "true_cfg_scale": 4.0,
#         "negative_prompt": " ",
#         "num_inference_steps": 25,  # 参考add_with_textbox.py的设置
#         "guidance_scale": 1.0,
#     }
    
#     # 执行推理
#     print("🎨 开始生成真实图片...")
#     with torch.inference_mode():
#         output = pipeline(**inputs)
#         output_image = output.images[0]
    
#     # 将输出图片resize到与输入图片相同的尺寸
#     resized_output_image = output_image.resize((input_width, input_height), Image.LANCZOS)
    
#     # 保存结果
#     resized_output_image.save(output_path)
#     print(f"✅ 真实图片已保存: {output_path}")
    
#     return resized_output_image


# def main():
#     """主函数"""
#     # 设置路径
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     original_image_path = os.path.join(script_dir, "imgs/image.png")
    
#     # 创建input和output目录在imgs下
#     input_dir = os.path.join(script_dir, "imgs/input")
#     output_dir = os.path.join(script_dir, "imgs/output")
#     os.makedirs(input_dir, exist_ok=True)
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 设置输出路径
#     doodle_image_path = os.path.join(input_dir, "image_with_doodle_hat.png")
#     realistic_image_path = os.path.join(output_dir, "image_with_realistic_hat.png")
    
#     # 检查输入文件是否存在
#     if not os.path.exists(original_image_path):
#         print(f"❌ 错误: 找不到图片文件 {original_image_path}")
#         return
    
#     # 初始化模型
#     pipeline = initialize_pipeline()
    
#     # 第一步：生成doodle图片
#     print("\n" + "="*60)
#     print("步骤 1/2: 生成doodle帽子")
#     print("="*60)
#     add_doodle_hat(original_image_path, doodle_image_path, pipeline)
    
#     # # 第二步：将doodle转换为真实图片
#     # print("\n" + "="*60)
#     # print("步骤 2/2: 将doodle转换为真实帽子")
#     # print("="*60)
#     # convert_doodle_to_realistic(doodle_image_path, realistic_image_path, pipeline)
    
#     # print("\n" + "="*60)
#     # print("�� 全部处理完成！")
#     # print(f"📁 Doodle图片: {doodle_image_path}")
#     # print(f"📁 真实图片: {realistic_image_path}")
#     # print("="*60)


# if __name__ == "__main__":
#     main()

'''
使用qwen-image-edit为图片添加doodle效果，然后转换为真实图片
为图中的elephant添加一个涂鸦绘制的帽子，然后将doodle转换为真实的帽子
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import glob
import json
import random
import re
from PIL import Image
from multi3_infer_plus import MyQwenImageEditPipeline, MultiGPUTransformer
from prompt_utils import polish_edit_prompt


def slugify(text):
    """
    将字符串转换为一个安全的文件名（"slug"）。
    例如："a party hat" -> "a_party_hat"
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)  # 移除特殊字符
    text = re.sub(r'[\s-]+', '_', text).strip('_') # 替换空格和-为_
    return text

def initialize_pipeline():
    """初始化图像编辑管道"""
    print("正在加载模型...")
    pipeline = MyQwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", 
        torch_dtype=torch.bfloat16, 
        cache_dir="/tmp"
    )
    
    # 设置模型精度和设备
    pipeline.transformer.to(torch.float32)
    pipeline.vae.to("cuda:0")
    pipeline.text_encoder.to("cuda:0")
    
    # 配置多GPU分布
    total_blocks = len(pipeline.transformer.transformer_blocks)
    gpu_split_points = [total_blocks//3, 2*total_blocks//3]  # 三等分
    pipeline.transformer = MultiGPUTransformer(pipeline.transformer, gpu_split_points)
    
    pipeline.set_progress_bar_config(disable=None)
    print("✅ 模型加载完成")
    return pipeline


def add_doodle_hat(image_path, output_path, pipeline,object_name):
    # 读取图片
    print(f"📖 读取图片: {image_path}")
    original_image = Image.open(image_path).convert("RGB")
    input_width, input_height = original_image.size
    print(f"   图片尺寸: {input_width} x {input_height}")
    
    # 设置prompt
    print(f"🎨 目标物体: {object_name}")
    # original_prompt = f"Add {object_name} on the subject in the image using only simple, rough outline strokes. The {object_name} should be drawn with minimal lines - just basic contours and shapes, like a quick sketch. No shading, no details, no filling - only simple line outlines. Keep everything else exactly the same as the original image, maintaining complete consistency except for the added {object_name} outline."
#     original_prompt = (
#     f"Add {object_name} on the subject in the image. "
#     f"The {object_name} MUST be drawn using ONLY simple, rough outline strokes, like a quick sketch. "
#     f"NO shading, NO details, NO filling, ONLY simple line outlines for the {object_name}. "
#     f"CRITICALLY: The rest of the image, including the subject and background, "
#     f"MUST remain IDENTICAL to the original. "
#     f"Maintain the EXACT original photorealistic style, colors, and textures of EVERYTHING else. "
#     f"Absolutely NO changes to the subject or background's original appearance. "
#     f"ONLY add the {object_name} outline."
# )
    original_prompt = (
    f"Add {object_name} on the subject in the image."
    f"This {object_name} MUST be in a doodle-style, drawn using only simple, minimal lines for its basic contour. NO details, NO filling, NO shading, and NO color are allowed; only pure line outlines."
    f"CRITICAL REQUIREMENT: Aside from this added {object_name} line outline, ALL other parts of the image—including the subject and background—MUST remain 100% IDENTICAL to the original. You MUST strictly preserve the subject's original photorealistic style, all details, textures, and colors. Absolutely NO stylistic or content changes to the subject or background are permitted."
    f"CRITICAL REQUIREMENT: You MUST NOT add any lines, outlines, or style changes to the subject or the background. "
    )
    print(f"📝 原始编辑指令: {original_prompt}")

    
    # 使用polish_edit_prompt润色prompt
    polished_prompt = polish_edit_prompt(original_prompt, original_image)
    print(f"✨ 润色后的指令: {polished_prompt}")
    
    # 准备推理参数
    inputs = {
        "image": original_image,
        "prompt": polished_prompt,
        "generator": torch.Generator(device="cuda").manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 25,  # 参考add_with_textbox.py的设置
        "guidance_scale": 1.0,
    }
    
    # 执行推理
    print("🎨 开始生成...")
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
    
    # 将输出图片resize到与输入图片相同的尺寸
    resized_output_image = output_image.resize((input_width, input_height), Image.LANCZOS)
    
    # 保存结果
    resized_output_image.save(output_path)
    print(f"✅ 图片已保存: {output_path}")
    
    return resized_output_image


def convert_doodle_to_realistic(doodle_image_path, output_path, pipeline,object_name):
    """将doodle图片转换为真实的图片"""
    # 读取doodle图片
    print(f"📖 读取doodle图片: {doodle_image_path}")
    doodle_image = Image.open(doodle_image_path).convert("RGB")
    input_width, input_height = doodle_image.size
    print(f"   图片尺寸: {input_width} x {input_height}")
    
    # 设置prompt - 将doodle转为真实
    print(f"🎨 目标物体: {object_name}")
    # original_prompt = f"Convert the doodle-style {object_name} into a realistic, photorealistic {object_name}. Keep everything else exactly the same as the original, maintaining complete consistency."
    original_prompt = (
        f"Convert the doodle-style {object_name} on the subject into a realistic, photorealistic {object_name}."
        f"CRITICAL REQUIREMENT: The rest of the image—including the subject's appearance, its exact colors, textures, and the entire background—MUST remain 100% IDENTICAL to the original base image (before any doodle was added). Strictly preserve the original photorealistic style, brightness, contrast, and color vividness of EVERYTHING except the {object_name}. Absolutely NO changes to the subject or background's original pixel data, only render the {object_name} realistically."
    )
    print(f"📝 原始编辑指令: {original_prompt}")
    
    # 使用polish_edit_prompt润色prompt
    polished_prompt = polish_edit_prompt(original_prompt, doodle_image)
    print(f"✨ 润色后的指令: {polished_prompt}")
    
    # 准备推理参数
    inputs = {
        "image": doodle_image,
        "prompt": polished_prompt,
        "generator": torch.Generator(device="cuda").manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 25,  # 参考add_with_textbox.py的设置
        "guidance_scale": 1.0,
    }
    
    # 执行推理
    print("🎨 开始生成真实图片...")
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
    
    # 将输出图片resize到与输入图片相同的尺寸
    resized_output_image = output_image.resize((input_width, input_height), Image.LANCZOS)
    
    # 保存结果
    resized_output_image.save(output_path)
    print(f"✅ 真实图片已保存: {output_path}")
    
    return resized_output_image


def main():
    """主函数"""
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # original_image_path = os.path.join(script_dir, "imgs/image.png")
    original_images_dir = os.path.join(script_dir, "imgs_test/originals")
    json_config_path = os.path.join(script_dir, "edit_doodles.json")
    
    # 创建input和output目录在imgs下
    input_dir = os.path.join(script_dir, "imgs_test/input")
    output_dir = os.path.join(script_dir, "imgs_test/output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    

    if not os.path.exists(json_config_path):
        print(f"❌ 错误: 找不到配置文件 {json_config_path}")
        return
    
    # 检查输入文件是否存在
    if not os.path.exists(original_images_dir):
        print(f"❌ 错误: 找不到图片文件 {original_images_dir}")
        return
    print(f"📖 正在读取配置文件: {json_config_path}")
    with open(json_config_path, 'r') as f:
            config = json.load(f)
    objects_to_add = config.get("objects", [])
    if not objects_to_add:
        print("❌ 错误: JSON文件中未找到 'objects' 列表或列表为空。")
        return
    print(f"🔍 找到 {len(objects_to_add)} 个可用物体: {objects_to_add}")

    # 查找所有图片
    print(f"📂 正在扫描源图片目录: {original_images_dir}")
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    all_image_paths = []
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(original_images_dir, ext)))
        
    if not all_image_paths:
        print(f"❌ 错误: 在 {original_images_dir} 中未找到任何图片文件。")
        print(f"   (支持的格式: {', '.join(image_extensions)})")
        return
        
    print(f"🖼️ 找到 {len(all_image_paths)} 张待处理图片。")

    # 初始化模型
    pipeline = initialize_pipeline()
    for i, original_image_path in enumerate(all_image_paths):
        print("\n" + "="*80)
        print(f"🚀 开始处理图片 {i+1}/{len(all_image_paths)}: {os.path.basename(original_image_path)}")
        print("="*80)
        
        random_object_name = random.choice(objects_to_add)
        print(f"🎲 为此图片随机选择的物体是: {random_object_name}")
        
        # 使用slugify和原始文件名创建唯一的输出文件名
        object_slug = slugify(random_object_name)
        original_basename = os.path.splitext(os.path.basename(original_image_path))[0]
        # 设置动态输出路径
        doodle_image_path = os.path.join(input_dir, f"{original_basename}_doodle_{object_slug}.png")
        realistic_image_path = os.path.join(output_dir, f"{original_basename}_realistic_{object_slug}.png")
    
        try:
            # 第一步：生成doodle图片
            print("\n" + "-"*60)
            print(f"步骤 1/2: 生成doodle ({random_object_name})")
            print("-"*60)
            add_doodle_hat(original_image_path, doodle_image_path, pipeline, random_object_name)
            
            # 第二步：将doodle转换为真实图片
            print("\n" + "-"*60)
            print(f"步骤 2/2: 将doodle转换为真实 ({random_object_name})")
            print("-"*60)
            convert_doodle_to_realistic(doodle_image_path, realistic_image_path, pipeline, random_object_name)
            
            print("\n" + "✓"*60)
            print(f"🎉 图片 '{os.path.basename(original_image_path)}' 处理完成！")
            print(f"   📁 Doodle图片: {doodle_image_path}")
            print(f"   📁 真实图片: {realistic_image_path}")
            print("✓"*60)

        except Exception as e:
            print(f"❌❌❌ 处理图片 {original_image_path} 时发生严重错误: {e}")
            print("将跳过此图片，继续处理下一个...")

    print("\n" + "="*80)
    print("✨✨✨ 全部任务处理完成！ ✨✨✨")
    print("="*80)


if __name__ == "__main__":
    main()






