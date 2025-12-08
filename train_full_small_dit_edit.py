# train_full_small_dit_edit.py
#
# 基于 train_qwen_edit_lora.py 修改而来，实现了分离式加载和全量训练
#
import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
import math
import gc

import torch
from tqdm.auto import tqdm
import numpy as np
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageEditPipeline,  # 使用 Edit Pipeline
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.control_dataset import loader, image_resize  # 使用 control_dataset
from omegaconf import OmegaConf
import transformers
from loguru import logger as loguru_logger  # 使用 Loguru
import bitsandbytes as bnb

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """仅解析 config 文件路径"""
    parser = argparse.ArgumentParser(description="Full training script for Small Qwen DiT Edit model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to training config file (e.g., train_full_small_dit_edit.yaml)",
    )
    args = parser.parse_args()
    return args.config


# --- 从 train_full_qwen_image.py 复制的辅助函数 ---
def setup_model_for_training(model):
    """Setup model parameters for full training"""
    loguru_logger.info("Setting up DIT for full training (enabling all gradients)...")
    model.train()
    model.requires_grad_(True)  # 默认DIT的所有参数都可训练

    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        loguru_logger.info("Gradient checkpointing enabled for Transformer")
    return model


def save_full_model(model, save_path, accelerator, args):
    """Save the full model state (只保存 Transformer)"""
    loguru_logger.info(f"Saving full DIT model to {save_path}")
    os.makedirs(save_path, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model._orig_mod if is_compiled_module(unwrapped_model) else unwrapped_model

    transformer_path = os.path.join(save_path, "transformer")
    unwrapped_model.save_pretrained(transformer_path, safe_serialization=True)

    config_path = os.path.join(save_path, "config.yaml")
    OmegaConf.save(args, config_path)

    loguru_logger.info(f"Transformer model saved successfully to {transformer_path}")


# --- 辅助函数结束 ---


# 复制自 train_qwen_edit_lora.py (无需修改)
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height, None


def main():
    args = OmegaConf.load(parse_args())

    # --- (日志和目录创建) ---
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    loguru_logger.add(
        os.path.join(logging_dir, "training.log"),
        rotation="100 MB", retention="10 days", level="INFO"
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # --- (设置DType) ---
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    loguru_logger.info(f"Using weight dtype: {weight_dtype}")

    # ------------------------------------------------------------------
    # 【【【 关键修改：分离式加载 】】】
    # ------------------------------------------------------------------
    loguru_logger.info("Loading models...")

    # 1. 从 original_model_path 加载 VAE, Text Encoder, 和 Scheduler
    loguru_logger.info(f"Loading VAE, TextEncoder, Scheduler from: {args.original_model_path}")

    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.original_model_path,
        transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.original_model_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.original_model_path,
        subfolder="scheduler",
    )

    # 2. 从 transformer_path 加载您的小型DIT
    loguru_logger.info(f"Loading SMALL Transformer from: {args.transformer_path}")
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.transformer_path,
        torch_dtype=weight_dtype
    )

    # ------------------------------------------------------------------
    # 【【【 关键修改：数据预缓存 (来自 train_qwen_edit_lora.py) 】】】
    # ------------------------------------------------------------------
    # (这部分逻辑被保留，因为它对 Edit 任务的数据处理至关重要)
    loguru_logger.info("Starting data pre-caching...")

    # 检查 YAML 中是否定义了缓存标志，如果未定义则使用默认值
    save_cache = getattr(args, 'save_cache_on_disk', False)
    precompute_text = getattr(args, 'precompute_text_embeddings', True)
    precompute_image = getattr(args, 'precompute_image_embeddings', True)

    cached_text_embeddings = None
    txt_cache_dir = None

    if precompute_text:
        loguru_logger.info("Pre-computing text embeddings (TextEncoder + Control Image)...")
        with torch.no_grad():
            if save_cache:
                cache_dir = os.path.join(args.output_dir, "cache")
                os.makedirs(cache_dir, exist_ok=True)
                txt_cache_dir = os.path.join(cache_dir, "text_embs")
                os.makedirs(txt_cache_dir, exist_ok=True)
            else:
                cached_text_embeddings = {}

            control_image_files = [i for i in os.listdir(args.data_config.control_dir) if
                                   i.endswith((".png", ".jpg", ".jpeg", ".webp"))]

            for img_name in tqdm(control_image_files, disable=not accelerator.is_main_process):
                img_path = os.path.join(args.data_config.control_dir, img_name)
                txt_path = os.path.join(args.data_config.img_dir, os.path.splitext(img_name)[0] + '.txt')

                if not os.path.exists(txt_path):
                    loguru_logger.warning(f"Skipping {img_name}: Corresponding text file not found at {txt_path}")
                    continue

                img = Image.open(img_path).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                prompt_image = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)

                prompt = open(txt_path, encoding='utf-8').read()

                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=prompt_image,
                    prompt=[prompt],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )

                txt_key = os.path.splitext(img_name)[0] + '.txt'
                if save_cache:
                    torch.save({'prompt_embeds': prompt_embeds[0].to('cpu'),
                                'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')},
                               os.path.join(txt_cache_dir, txt_key + '.pt'))
                else:
                    cached_text_embeddings[txt_key] = {'prompt_embeds': prompt_embeds[0].to('cpu'),
                                                       'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}

    # 预缓存图片 (来自 train_qwen_edit_lora.py)
    cached_image_embeddings = None
    cached_image_embeddings_control = None
    img_cache_dir = None

    if precompute_image:
        loguru_logger.info("Pre-computing image latents (VAE)...")
        if not save_cache:
            cached_image_embeddings = {}
            cached_image_embeddings_control = {}

        with torch.no_grad():
            # 1. 缓存目标图
            loguru_logger.info("Caching target images...")
            target_image_files = [i for i in os.listdir(args.data_config.img_dir) if
                                  i.endswith((".png", ".jpg", ".jpeg", ".webp"))]
            for img_name in tqdm(target_image_files, disable=not accelerator.is_main_process):
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                img = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)
                img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if not save_cache:
                    cached_image_embeddings[img_name] = pixel_latents

            # 2. 缓存控制图
            loguru_logger.info("Caching control images...")
            control_image_files = [i for i in os.listdir(args.data_config.control_dir) if
                                   i.endswith((".png", ".jpg", ".jpeg", ".webp"))]
            for img_name in tqdm(control_image_files, disable=not accelerator.is_main_process):
                img = Image.open(os.path.join(args.data_config.control_dir, img_name)).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                img = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)
                img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if not save_cache:
                    cached_image_embeddings_control[img_name] = pixel_latents

    # 清理显存
    loguru_logger.info("Clearing VAE and TextEncoder from VRAM...")
    vae.to('cpu')
    text_encoding_pipeline.to("cpu")
    torch.cuda.empty_cache()
    del text_encoding_pipeline
    gc.collect()

    # ------------------------------------------------------------------
    # 【【【 关键修改：设置全量训练 】】】
    # ------------------------------------------------------------------
    flux_transformer = setup_model_for_training(flux_transformer)

    # 准备 Sigmas (同 train_qwen_edit_lora.py)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ------------------------------------------------------------------
    # 【【【 关键修改：设置全量优化器 】】】
    # ------------------------------------------------------------------
    trainable_params = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    loguru_logger.info(f"Found {len(trainable_params)} trainable parameters for DIT.")

    if getattr(args, 'use_8bit_adam', False):
        loguru_logger.info("Using 8-bit Adam optimizer")
        optimizer = bnb.optim.AdamW8bit(
            trainable_params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
        )
    else:
        loguru_logger.info("Using standard AdamW optimizer")
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
        )

    # 准备数据加载器 (同 train_qwen_edit_lora.py)
    train_dataloader = loader(
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings,
        cached_image_embeddings_control=cached_image_embeddings_control,
        **args.data_config
    )

    # 准备 LR 调度器 (同 train_qwen_edit_lora.py)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # 【关键】用 Accelerator 准备
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # (恢复 Checkpoint 的逻辑来自 train_full_small_dit.py)
    global_step = 0
    initial_global_step = 0
    resume_checkpoint = getattr(args, 'resume_from_checkpoint', None)
    if resume_checkpoint:  # TODO: 这里有问题，需要解决
        # ... (省略了与 train_full_small_dit.py 相同的恢复逻辑) ...
        loguru_logger.warning(
            "Checkpoint resuming not fully implemented in this example script. Starting from scratch.")
        # (您可以从 train_full_small_dit.py 复制完整的恢复逻辑)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"config": OmegaConf.to_yaml(args)})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training (Qwen-Image-Edit Full DIT) *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # -------------------------------------------------
    # --- 训练循环 (来自 train_qwen_edit_lora.py) ---
    # -------------------------------------------------
    vae_scale_factor = 2 ** len(vae.config.temperal_downsample)
    num_train_epochs = getattr(args, 'num_train_epochs', 1)

    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):

                # (数据来自预缓存)
                img, prompt_embeds, prompt_embeds_mask, control_img = batch

                prompt_embeds = prompt_embeds.to(dtype=weight_dtype).to(accelerator.device)
                prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32).to(accelerator.device)
                control_img = control_img.to(dtype=weight_dtype).to(accelerator.device)
                pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)

                with torch.no_grad():
                    # (数据已在缓存中被 VAE 编码和归一化)
                    # ( permute 和 normalize )
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                    control_img = control_img.permute(0, 2, 1, 3, 4)

                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std
                    control_img = (control_img - latents_mean) * latents_std

                    # (加噪和时间步)
                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                    sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                    noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise  # 目标图加噪

                    # 【Edit 关键逻辑 1: 打包】
                    packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                        noisy_model_input, bsz,
                        noisy_model_input.shape[2], noisy_model_input.shape[3], noisy_model_input.shape[4],
                    )
                    packed_control_img = QwenImageEditPipeline._pack_latents(
                        control_img, bsz,
                        control_img.shape[2], control_img.shape[3], control_img.shape[4],
                    )

                    img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                                   (1, control_img.shape[3] // 2, control_img.shape[4] // 2)]] * bsz

                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                    # 【Edit 关键逻辑 2: 拼接】
                    packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)

                # DIT 前向传播
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,  # <--- 使用拼接后的输入
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                # 【Edit 关键逻辑 3: 切片】
                # 只取预测结果的前半部分（对应目标图的部分）
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # (计算 Loss)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # ------------------------------------------------------
                # 【【【 关键修改：保存全量模型 】】】
                # ------------------------------------------------------
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # (省略了检查点数量管理)
                        # ...

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                        # 【关键】使用我们修改后的保存函数
                        save_full_model(flux_transformer, save_path, accelerator, args)
                        logger.info(f"Saved full DIT checkpoint to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            loguru_logger.info("Reached max_train_steps. Exiting training loop.")
            break

    # 训练结束
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model")
        save_full_model(flux_transformer, final_save_path, accelerator, args)
        loguru_logger.info(f"Training completed! Final model saved to {final_save_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()