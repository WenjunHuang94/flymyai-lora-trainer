# create_smaller_dit.py (已修复)
import torch
import time
import argparse
import os
from diffusers import QwenImageTransformer2DModel
from transformers import logging as hf_logging
import logging
import math

# 压制日志
hf_logging.set_verbosity_error()
# 我们使用 Python 自己的日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _copy_compatible_weights(src_model, dst_model):
    """
    安全地从源模型复制“形状完全匹配”的权重到目标模型。
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    copied_layers = 0
    total_layers = len(dst_state.keys())

    print("  [权重复制] 开始复制兼容的权重...")
    for key in dst_state.keys():
        if key in src_state and src_state[key].shape == dst_state[key].shape:
            # 权重形状完全兼容
            dst_state[key].copy_(src_state[key])
            copied_layers += 1
        else:
            # 权重不存在或形状不匹配（例如 transformer_blocks.20 之后）
            # print(f"    SKIPPED (将使用随机初始化): {key}")
            pass

    print(f"  [权重复制] 成功复制 {copied_layers}/{total_layers} 个权重层。")
    return copied_layers, total_layers


def create_smaller_dit_model(
        pretrained_model_path="Qwen/Qwen-Image",
        output_path="./my_small_dit",
        # 【【【 新增的控制旋钮 】】】
        new_num_layers: int = None,
        new_num_attention_heads: int = None,
        new_attention_head_dim: int = None
):
    """
    从 Qwen-Image 加载 DiT，根据新参数缩小它，
    复制兼容的权重，然后保存到新目录。
    """
    print("=" * 60)
    print(f"开始创建“小型 DIT”...")
    print("=" * 60)

    # 1. 加载“原始”的 DiT（必须加载权重以便复制）
    print(f"\n1. 正在加载原始 DiT 模型: {pretrained_model_path}")
    print("   (这可能需要几分钟并占用大量内存/显存)...")
    load_start = time.time()
    try:
        original_transformer = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16  # 使用 bfloat16 节省内存
        ).eval()  # 设置为评估模式
        original_config = original_transformer.config
        orig_params = sum(p.numel() for p in original_transformer.parameters())
        print(f"  原始模型加载完毕 (耗时: {time.time() - load_start:.2f}s)")

        original_inner_dim = original_config.num_attention_heads * original_config.attention_head_dim

    except Exception as e:
        print(f"  加载原始 DiT 失败: {e}")
        return

    # 2. 创建“新”配置
    print(f"\n2. 正在修改配置...")
    new_config_dict = dict(original_config)

    if 'pooled_projection_dim' in new_config_dict:
        removed_key = new_config_dict.pop('pooled_projection_dim')
        print(f"  (Debug) 移除了不兼容的配置项: pooled_projection_dim: {removed_key}")

    # --- 应用我们的“缩小”修改 ---
    if new_num_layers is not None:
        new_config_dict['num_layers'] = int(new_num_layers)
        print(f"  配置已修改: num_layers 从 {original_config.num_layers} -> {new_num_layers}")

    if new_num_attention_heads is not None:
        new_config_dict['num_attention_heads'] = int(new_num_attention_heads)
        print(
            f"  配置已修改: num_attention_heads 从 {original_config.num_attention_heads} -> {new_num_attention_heads}")

    # --- 【【【 关键修复：RoPE 缩放逻辑 】】】 ---

    print('original_config = ', original_config)

    # 1. 确定最终的 head_dim
    original_head_dim = original_config.attention_head_dim
    if new_attention_head_dim is not None:
        final_head_dim = int(new_attention_head_dim)
        if final_head_dim != original_head_dim:
            print(f" 配置已修改: attention_head_dim 从 {original_head_dim} -> {final_head_dim}")
    else:
        final_head_dim = original_head_dim  # 保持不变

    new_config_dict['attention_head_dim'] = final_head_dim

    # 2. 检查 head_dim 是否被修改
    if final_head_dim != original_head_dim:
        print(f" [!!] 关键修复：attention_head_dim 已从 {original_head_dim} 更改为 {final_head_dim}。")
        print("      正在按比例缩放 'axes_dims_rope'...")

        original_rope_dims = original_config.axes_dims_rope  # (16, 56, 56)

        # 3. 按比例计算新的 RoPE 维度
        # 我们使用 (d/total) * new_total 的比例，并确保结果为偶数
        d0, d1, d2 = original_rope_dims
        total_orig_dim = d0 + d1 + d2  # 128

        # 按比例分配，并四舍五入到最近的偶数
        new_d0 = int(round((d0 / total_orig_dim) * final_head_dim / 2.0)) * 2
        new_d1 = int(round((d1 / total_orig_dim) * final_head_dim / 2.0)) * 2

        # 最后一个维度填充剩余部分，以确保总和 = final_head_dim
        new_d2 = final_head_dim - new_d0 - new_d1

        # 极端情况检查：如果 d2 变为奇数或负数，使用回退策略
        if new_d2 < 0 or new_d2 % 2 != 0:
            print("       (比例缩放失败，使用回退策略...)")
            # 回退：将第一个设为 16 (或更小)，剩下平分
            new_d0 = min(16, final_head_dim // 4)
            if new_d0 % 2 != 0: new_d0 -= 1

            new_d1 = (final_head_dim - new_d0) // 2
            if new_d1 % 2 != 0: new_d1 -= 1

            new_d2 = final_head_dim - new_d0 - new_d1

        new_rope_dims = (new_d0, new_d1, new_d2)

        print(f"      新的 axes_dims_rope: {new_rope_dims} (总和: {sum(new_rope_dims)})")

        # 4. 更新配置
        new_config_dict['axes_dims_rope'] = new_rope_dims

    # --- 【【【 修复结束 】】】 ---

    # # --- 强制 in/out channels 保持 VAE 兼容 (64, 16) ---
    # original_in_channels = 64
    # original_out_channels = 16
    #
    # if 'in_channels' not in new_config_dict or new_config_dict['in_channels'] != original_in_channels:
    #     print(f" [!!] 关键修复：正在强制 'in_channels' 恢复为 {original_in_channels}")
    #     new_config_dict['in_channels'] = original_in_channels
    #
    # if 'out_channels' not in new_config_dict or new_config_dict['out_channels'] != original_out_channels:
    #     print(f" [!!] 关键修复：正在强制 'out_channels' 恢复为 {original_out_channels}")
    #     new_config_dict['out_channels'] = original_out_channels

    # 3. 创建“新”的“空壳”模型
    print(f"\n3. 正在实例化新的 DiT 架构...")
    # 【【【 关键修复：使用 **new_config_dict 来调用构造函数 】】】
    new_model = QwenImageTransformer2DModel(**new_config_dict).to(torch.bfloat16)
    new_params = sum(p.numel() for p in new_model.parameters())
    print(" 新架构实例化完毕。")
    # print('new_model = ', new_model) # 调试时取消注释

    # 4. 复制权重
    print(f"\n4. 正在从原始模型向新模型复制权重...")
    copied, total = _copy_compatible_weights(original_transformer.to("cpu"), new_model)

    # 5. 清理内存
    print(f"\n5. 正在清理原始模型内存...")
    del original_transformer
    torch.cuda.empty_cache()

    # 6. 保存“新”的小型 DIT 模型
    print(f"\n6. 正在保存您的小型 DiT 模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)  # 确保目录存在
    new_model.save_pretrained(output_path)
    print(f" 模型已成功保存！")

    # 7. 总结
    reduction = (1 - new_params / orig_params) * 100
    print("\n" + "=" * 60)
    print("总结：")
    print(f" 原始参数量 ({original_config.num_layers} 层, {original_inner_dim} 维): {orig_params / 1e9:.2f}B")
    print(f" 新参数量 ({new_config_dict['num_layers']} 层, {new_model.inner_dim} 维): {new_params / 1e9:.2f}B")
    print(f" 参数量减少: {reduction:.1f}%")
    print(f" 成功复制了 {copied}/{total} 个权重层。")
    print("=" * 60)


if __name__ == "__main__":
    # --- 在这里配置您的“缩小”目标 ---

    # 方案 2: (更激进) 20 层, 16 个头, 96 维
    TARGET_LAYERS = 20
    TARGET_HEADS = 16
    TARGET_HEAD_DIM = 96
    OUTPUT_DIRECTORY = "./my_small_dit_1_7B"

    # ---------------------------------

    create_smaller_dit_model(
        pretrained_model_path="Qwen/Qwen-Image",
        output_path=OUTPUT_DIRECTORY,
        new_num_layers=TARGET_LAYERS,
        new_num_attention_heads=TARGET_HEADS,
        new_attention_head_dim=TARGET_HEAD_DIM
    )