import inspect
from diffusers import QwenImagePipeline

# 获取 QwenImagePipeline 类的源文件路径
file_path = inspect.getfile(QwenImagePipeline)
print(f"Qwen-Image 的 Pipeline 源码位于: \n{file_path}")

# 你也可以顺便查看 diffusers 库的安装根目录
# import diffusers
# print(f"\nDiffusers 库的安装目录: {diffusers.__path__[0]}")