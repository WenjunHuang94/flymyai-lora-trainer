import argparse
import copy
import os
import shutil
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage, QwenImagePipeline, \
    QwenImageTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.dataset import loader
from omegaconf import OmegaConf
import transformers
from loguru import logger as loguru_logger
import logging

# --- 强制设置随机种子 ---
SEED = 42
set_seed(SEED)

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./train_overfit.yaml", required=True)
    args = parser.parse_args()
    return args.config


def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loguru_logger.info(f"Total: {total_params / 1e6:.2f}M, Trainable: {trainable_params / 1e6:.2f}M")


def setup_model_for_training(model):
    model.train()
    model.requires_grad_(True)
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    return model


def save_full_model(model, save_path, accelerator, args):
    # 如果目录存在，为了确保干净的替换，可以先清理（可选，但 diffusers save 通常会覆盖）
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)

    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model._orig_mod if is_compiled_module(unwrapped_model) else unwrapped_model
    transformer_path = os.path.join(save_path, "transformer")
    unwrapped_model.save_pretrained(transformer_path, safe_serialization=True)
    OmegaConf.save(args, os.path.join(save_path, "config.yaml"))
    loguru_logger.info(f"Saved model to {transformer_path}")


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # Logging setup
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%m/%d %H:%M", level=logging.INFO)
    loguru_logger.add(os.path.join(logging_dir, "train.log"), level="INFO")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

    # 1. Load Components
    loguru_logger.info("Loading models...")
    vae = AutoencoderKLQwenImage.from_pretrained(args.original_model_path, subfolder="vae").to(accelerator.device,
                                                                                               dtype=weight_dtype)
    text_pipeline = QwenImagePipeline.from_pretrained(args.original_model_path, transformer=None, vae=None,
                                                      torch_dtype=weight_dtype).to(accelerator.device)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.original_model_path, subfolder="scheduler")

    # Load Small DiT
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(args.transformer_path, torch_dtype=weight_dtype)
    flux_transformer = setup_model_for_training(flux_transformer)

    # 2. Optimizer
    optimizer = torch.optim.AdamW(flux_transformer.parameters(), lr=args.learning_rate,
                                  weight_decay=args.adam_weight_decay)

    # 3. Dataloader
    train_dataloader = loader(**args.data_config)

    # 4. Scheduler
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=args.max_train_steps)

    # 5. Prepare
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Freeze VAE/Text
    vae.eval()
    text_pipeline.text_encoder.eval()

    # Sigmas helper
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim: sigma = sigma.unsqueeze(-1)
        return sigma

    # Training Loop
    global_step = 0
    loguru_logger.info("STARTING OVERFIT EXPERIMENT")

    vae_scale_factor = 2 ** (len(vae.config.temperal_downsample))

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # --- 【新增】最优模型记录变量 ---
    best_loss = float('inf')  # 初始设为无穷大
    interval_loss_sum = 0.0  # 记录一个 checkpoint 周期内的总 Loss
    interval_steps = 0  # 记录一个 checkpoint 周期内的步数

    while global_step < args.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                img, prompts = batch

                if global_step == 0 and accelerator.is_main_process:
                    print(f"Overfitting on prompt: '{prompts[0]}'")

                with torch.no_grad():
                    pixel_values = img.to(dtype=weight_dtype).unsqueeze(2)
                    pixel_latents = vae.encode(pixel_values).latent_dist.sample().permute(0, 2, 1, 3, 4)
                    # Normalize
                    mean = torch.tensor(vae.config.latents_mean).view(1, 1, 16, 1, 1).to(pixel_latents)
                    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, 16, 1, 1).to(pixel_latents)
                    pixel_latents = (pixel_latents - mean) * std

                    # Noise
                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents)

                    # Timestep Sampling
                    u = compute_density_for_timestep_sampling(weighting_scheme="none", batch_size=bsz)
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(accelerator.device)

                    sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                    noisy_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                    # Pack
                    packed_noisy = QwenImagePipeline._pack_latents(noisy_input, bsz, noisy_input.shape[2],
                                                                   noisy_input.shape[3], noisy_input.shape[4])
                    img_shapes = [(1, noisy_input.shape[3] // 2, noisy_input.shape[4] // 2)] * bsz

                    # Text Embeds
                    prompt_embeds, prompt_mask = text_pipeline.encode_prompt(prompt=prompts, device=accelerator.device,
                                                                             max_sequence_length=1024)
                    txt_lens = prompt_mask.sum(dim=1).tolist()

                # Forward
                pred = flux_transformer(
                    hidden_states=packed_noisy,
                    timestep=timesteps / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_lens,
                    return_dict=False
                )[0]

                # Unpack & Loss
                pred = QwenImagePipeline._unpack_latents(pred, noisy_input.shape[3] * vae_scale_factor,
                                                         noisy_input.shape[4] * vae_scale_factor, vae_scale_factor)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = (noise - pixel_latents).permute(0, 2, 1, 3, 4)
                loss = torch.mean(
                    (weighting.float() * (pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                loss_val = loss.detach().item()

                # --- 【新增】累积 Loss ---
                interval_loss_sum += loss_val
                interval_steps += 1

                progress_bar.set_postfix({"loss": f"{loss_val:.6f}"})

                if global_step % args.checkpointing_steps == 0:
                    # 1. 保存常规 Checkpoint (按步数命名，保留历史)
                    save_full_model(flux_transformer, os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                                    accelerator, args)

                    # 2. 【新增】计算平均 Loss 并保存最佳模型
                    avg_loss = interval_loss_sum / interval_steps

                    loguru_logger.info(f"Step {global_step}: Avg Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")

                    if avg_loss < best_loss:
                        loguru_logger.info(
                            f"🔥 Found new best model! Loss dropped from {best_loss:.6f} to {avg_loss:.6f}")
                        best_loss = avg_loss

                        # 保存到固定目录 checkpoint-best，会自动覆盖旧文件
                        best_save_path = os.path.join(args.output_dir, "checkpoint-best")
                        save_full_model(flux_transformer, best_save_path, accelerator, args)

                    # 重置计数器
                    interval_loss_sum = 0.0
                    interval_steps = 0

            if global_step >= args.max_train_steps: break

    save_full_model(flux_transformer, os.path.join(args.output_dir, "final_model"), accelerator, args)


if __name__ == "__main__":
    main()