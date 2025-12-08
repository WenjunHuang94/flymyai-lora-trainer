# train_full_small_dit.py
import argparse
import copy
import logging
import os
import shutil
import math
import gc

import torch
from tqdm.auto import tqdm
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.dataset import loader
from omegaconf import OmegaConf
from loguru import logger as loguru_logger
import bitsandbytes as bnb

# 屏蔽 Transformers 的详细日志
transformers.logging.set_verbosity_error()
diffusers.utils.logging.set_verbosity_error()

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """仅解析 config 文件路径"""
    parser = argparse.ArgumentParser(description="Full training script for Small Qwen DiT model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to training config file (e.g., train_small_dit.yaml)",
    )
    args = parser.parse_args()
    return args.config


def calculate_model_size(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loguru_logger.info(f"Total parameters: {total_params / 1_000_000:.2f}M")
    loguru_logger.info(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")
    return total_params, trainable_params


def setup_model_for_training(model):
    """设置模型为训练模式"""
    model.train()
    model.requires_grad_(True)  # 开启梯度

    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        loguru_logger.info("Gradient checkpointing enabled for Transformer")

    return model


def save_inference_weights(model, save_path, accelerator, args):
    """
    仅保存用于推理的模型权重 (safetensors)，不包含优化器状态。
    方便 inference_small_dit.py 直接加载。
    """
    transformer_path = os.path.join(save_path, "transformer")
    os.makedirs(transformer_path, exist_ok=True)

    # 解包模型
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model._orig_mod if is_compiled_module(unwrapped_model) else unwrapped_model

    # 保存权重
    unwrapped_model.save_pretrained(transformer_path, safe_serialization=True)

    # 保存配置
    config_path = os.path.join(save_path, "config.yaml")
    OmegaConf.save(args, config_path)

    loguru_logger.info(f"Inference weights saved to {transformer_path}")


def main():
    config_path = parse_args()
    args = OmegaConf.load(config_path)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # --- 日志设置 ---
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        loguru_logger.add(
            os.path.join(logging_dir, "training.log"),
            rotation="100 MB",
            retention="10 days",
            level="INFO"
        )

    # --- 设置精度 ---
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    loguru_logger.info(f"Using weight dtype: {weight_dtype}")

    # ------------------------------------------------------------------
    # 1. 加载固定组件 (VAE, Text Encoder, Scheduler)
    # ------------------------------------------------------------------
    loguru_logger.info(f"Loading frozen components from: {args.original_model_path}")

    # 加载 Text Encoder Pipeline (仅用于编码 Prompt)
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.original_model_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype
    )

    # 加载 VAE
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.original_model_path,
        subfolder="vae",
    )

    # 加载 Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.original_model_path,
        subfolder="scheduler",
    )

    # 冻结并移动到 GPU
    text_encoding_pipeline.to(accelerator.device)
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()
    text_encoding_pipeline.text_encoder.requires_grad_(False)
    text_encoding_pipeline.text_encoder.eval()

    # ------------------------------------------------------------------
    # 2. 加载待训练的小型 DiT
    # ------------------------------------------------------------------
    loguru_logger.info(f"Loading SMALL Transformer from: {args.transformer_path}")
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.transformer_path,
        torch_dtype=weight_dtype
    )

    # 设置为训练模式
    flux_transformer = setup_model_for_training(flux_transformer)

    if accelerator.is_main_process:
        calculate_model_size(flux_transformer)

    # ------------------------------------------------------------------
    # 3. 准备优化器
    # ------------------------------------------------------------------
    trainable_params = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))

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

    # ------------------------------------------------------------------
    # 4. 准备数据加载器和调度器
    # ------------------------------------------------------------------
    loguru_logger.info("Setting up data loader...")
    train_dataloader = loader(**args.data_config)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ------------------------------------------------------------------
    # 5. Accelerator Prepare (核心步骤)
    # ------------------------------------------------------------------
    # 仅 prepare 需要训练和迭代的对象
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # ------------------------------------------------------------------
    # 6. 恢复训练状态 (Resume Logic - 修复版)
    # ------------------------------------------------------------------
    global_step = 0
    initial_global_step = 0
    resume_checkpoint = getattr(args, 'resume_from_checkpoint', None)

    if resume_checkpoint:  # TODO: 这里有问题，需要解决
        if resume_checkpoint == "latest":
            # 寻找输出目录中以 checkpoint- 开头的文件夹
            dirs = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if dirs:
                # 按步数排序
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                resume_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_checkpoint = None

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            loguru_logger.info(f"Resuming training state from: {resume_checkpoint}")
            try:
                # 【核心】恢复所有状态（模型权重、优化器、调度器、随机数）
                accelerator.load_state(resume_checkpoint)

                # 解析当前步数
                global_step = int(os.path.basename(resume_checkpoint).split("-")[1])
                initial_global_step = global_step
                loguru_logger.info(f"Successfully resumed from global_step {global_step}")
            except Exception as e:
                loguru_logger.error(f"Failed to load accelerator state: {e}")
                loguru_logger.warning("Starting training from scratch...")

    # ------------------------------------------------------------------
    # 7. 训练循环准备
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"config": OmegaConf.to_yaml(args)})

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae_scale_factor = 2 ** len(vae.temperal_downsample)

    # 定义 Sigmas 获取函数
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    loguru_logger.info("***** Running training *****")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ------------------------------------------------------------------
    # 8. 开始训练
    # ------------------------------------------------------------------
    num_train_epochs = getattr(args, 'num_train_epochs', 1)

    # 为了支持无限步数直到 max_train_steps，我们使用 while 循环或者大 epoch
    for epoch in range(num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                img, prompts = batch

                # 打印首个 Prompt 用于 Debug
                if global_step == initial_global_step and step == 0 and accelerator.is_main_process:
                    loguru_logger.info(f"Debug - First Prompt: {prompts[0]}")

                # --- 数据准备 (No Grad) ---
                with torch.no_grad():
                    # 1. VAE 编码
                    pixel_values = img.to(dtype=weight_dtype)
                    pixel_values = pixel_values.unsqueeze(2)
                    pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

                    # 归一化
                    latents_mean = torch.tensor(vae.config.latents_mean).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype)
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype)
                    pixel_latents = (pixel_latents - latents_mean) * latents_std

                    # 2. 加噪
                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)

                    # 时间步采样
                    u = compute_density_for_timestep_sampling(weighting_scheme="none", batch_size=bsz, logit_mean=0.0,
                                                              logit_std=1.0, mode_scale=1.29)
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                    sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                    noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                    # 3. 打包 Latents
                    packed_noisy_model_input = QwenImagePipeline._pack_latents(
                        noisy_model_input, bsz, noisy_model_input.shape[2], noisy_model_input.shape[3],
                        noisy_model_input.shape[4]
                    )
                    img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz

                    # 4. 编码 Prompt
                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                        prompt=prompts,
                        device=packed_noisy_model_input.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                # --- 模型前向传播 (With Grad) ---
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                # Unpack
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # 计算 Loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # 记录 Loss
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # --- 反向传播 ---
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --- 更新步骤 ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix({"loss": f"{train_loss:.5f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}"})
                train_loss = 0.0

                # --------------------------------------------------------------
                # 保存 Checkpoint (修正逻辑)
                # --------------------------------------------------------------
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                        # 1. 保存完整训练状态 (用于 Resume)
                        # 这会保存模型参数、优化器状态、LR调度器状态
                        accelerator.save_state(save_path)

                        # 2. 同时保存方便推理的纯权重 (safetensors)
                        # 保存在子目录 transformer 中，方便 inference 脚本直接读取
                        save_inference_weights(flux_transformer, save_path, accelerator, args)

                        loguru_logger.info(f"Saved state and weights to {save_path}")

                        # 3. 管理 Checkpoint 数量
                        if hasattr(args, 'checkpoints_total_limit') and args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                loguru_logger.info(f"Removing {len(removing_checkpoints)} old checkpoints")
                                for removing_checkpoint in removing_checkpoints:
                                    try:
                                        shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                                    except OSError as e:
                                        loguru_logger.warning(f"Error removing checkpoint: {e}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # --- 训练结束保存最终模型 ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model")
        save_inference_weights(flux_transformer, final_save_path, accelerator, args)
        loguru_logger.info(f"Training completed. Final weights saved to {final_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()