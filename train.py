import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
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
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.dataset import loader
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers

logger = get_logger(__name__, log_level="INFO")




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make one log on every process with the configuration for debugging.
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
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(  # 负责读取您的 .txt 文件并将其转换为“文本嵌入”。 text_encoder (文本编码器) 应该加载 Qwen-2.5-VL 模型
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    vae = AutoencoderKLQwenImage.from_pretrained(  # vae: 负责将您的 .jpg 图片压缩成模型能理解的“潜空间”数据 (Latent)。
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(  # flux_transformer: 这是 Qwen-Image 的“主大脑”，负责处理图像和文本。
        args.pretrained_model_name_or_path,
        subfolder="transformer",    )
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.add_adapter(lora_config)   # 它根据您 YAML 中的 rank: 16 创建了一个 LoRA 配置，然后调用 add_adapter 将这些微小的、可训练的 LoRA 层“注入”到 flux_transformer 的关键模块中
    text_encoding_pipeline.to(accelerator.device)
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
        
    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)


    flux_transformer.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True  # 冻结 flux_transformer 的所有原始参数（几十亿个参数），只保留 lora 层的参数为可训练状态（几百万个参数）
            print(n)
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())

    flux_transformer.enable_gradient_checkpointing()
    optimizer = optimizer_cls(  # 它创建了一个 AdamW 优化器，并明确告诉它：“你只需要更新那些可训练的 lora_layers”
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)  # 根据您 img_dir 的配置，初始化了数据加载器，准备开始读取您的 10 张图片和 .txt 文件

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    vae.to(accelerator.device, dtype=weight_dtype)
    flux_transformer, optimizer, _, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, deepcopy(train_dataloader), lr_scheduler
    )


    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    for epoch in range(1):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                img, prompts = batch
                with torch.no_grad():
                    pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                    pixel_values = pixel_values.unsqueeze(2)

                    pixel_latents = vae.encode(pixel_values).latent_dist.sample()  # 使用 vae 将图片转换为潜空间数据
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std  # VAE数据归一化
                    

                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)  # 它模拟一个“加噪”的过程，为模型去噪做准备
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)  # 在每一步（step）训练开始时，脚本都会随机从 1000 个总时间步中**“抽取”**一个 t 值。比如，第 1 步抽到了 t=77，第 2 步抽到了 t=845

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)  # 脚本拿着刚抽到的 t=845，去“调度总管”（scheduler）的“Sigma 速查表”（scheduler.sigmas）里查找对应的**“混合比例” $\sigma$**（比如 0.81）。
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise  # 脚本执行“混合”：（1 - 0.81) * 您的干净图片 + 0.81 * 纯噪声。这就得到了在 t=845 时对应的“噪声图片”（noisy_model_input）。
                # Concatenate across channels.
                # pack the latents.
                packed_noisy_model_input = QwenImagePipeline._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                # latent image ids for RoPE.
                img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz
                with torch.no_grad():
                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(  # 将您的描述（"ohwhwj man..."）转换为模型能理解的数学向量
                        prompt=prompts,
                        device=packed_noisy_model_input.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                model_pred = flux_transformer(  # 这是学习的关键） 模型同时接收“加噪的图像”和“文本向量”，然后预测出它认为“干净”的图像应该是什么样子
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,  # 脚本把同一个随机抽取的 t 值（t=845），除以 1000（标准化为 0.845），然后作为 timestep 参数传入 flux_transformer（DiT）
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                # flow-matching loss
                target = noise - pixel_latents  # 模型应该预测出的标准答案
                target = target.permute(0, 2, 1, 3, 4)
                loss = torch.mean(  # (model_pred.float() - target.float()) 比喻： 教练拿出您射出的“实际轨迹” (model_pred)，和“完美轨迹” (target) 进行对比，计算它们之间的偏差  # TODO: 注意这里预测的是偏差
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()  # 您的“轨迹”是由几百万个像素点组成的。教练会计算出所有像素点上“惩罚值”的平均数
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:  # 250次保存一次
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    #accelerator.save_state(save_path)
                    try:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                    except:
                        pass
                    unwrapped_flux_transformer = unwrap_model(flux_transformer)  # 从 accelerator 中“解包”出原始的 flux_transformer 模型
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )

                    QwenImagePipeline.save_lora_weights(  # 将这些 LoRA 权重保存为 .safetensors 文件
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,
                    )

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
