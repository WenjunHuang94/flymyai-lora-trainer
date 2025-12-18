import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1, cached_text_embeddings=None,
                 cached_image_embeddings=None, control_dir=None, cached_image_embeddings_control=None,
                 txt_cache_dir=None, img_cache_dir=None, control_img_cache_dir=None):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.control_dir = control_dir
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control
        self.txt_cache_dir = txt_cache_dir
        self.img_cache_dir = img_cache_dir
        self.control_img_cache_dir = control_img_cache_dir
        print('cached_text_embeddings', type(cached_text_embeddings))

    def __len__(self):
        return 999999

    def __getitem__(self, idx):
        try:
            idx = random.randint(0, len(self.images) - 1)
            basename = os.path.basename(self.images[idx])

            # target image (or its precomputed latents)
            if self.cached_image_embeddings is None and self.img_cache_dir is None:
                img = Image.open(self.images[idx]).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        img = crop_to_aspect_ratio(img, ratio)
                img = image_resize(img, self.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1)
            elif self.img_cache_dir is not None:
                img = torch.load(os.path.join(self.img_cache_dir, basename + '.pt'))
            else:
                img = self.cached_image_embeddings[basename]

            # control image (or its precomputed latents)
            if self.cached_control_image_embeddings is None and self.control_img_cache_dir is None:
                if self.control_dir is None:
                    raise ValueError("control_dir is required when control image latents are not cached.")
                control_path = os.path.join(self.control_dir, basename)
                control_img = Image.open(control_path).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        control_img = crop_to_aspect_ratio(control_img, ratio)
                control_img = image_resize(control_img, self.img_size)
                w, h = control_img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                control_img = control_img.resize((new_w, new_h))
                control_img = torch.from_numpy((np.array(control_img) / 127.5) - 1)
                control_img = control_img.permute(2, 0, 1)
            elif self.control_img_cache_dir is not None:
                control_img = torch.load(os.path.join(self.control_img_cache_dir, basename + '.pt'))
            else:
                control_img = self.cached_control_image_embeddings[basename]

            # rsplit to avoid breaking on paths containing '.' (e.g. './images')
            txt_path = self.images[idx].rsplit('.', 1)[0] + '.' + self.caption_type
            if self.cached_text_embeddings is None and self.txt_cache_dir is None:
                prompt = open(txt_path, encoding='utf-8').read()
                if throw_one(self.caption_dropout_rate):
                    return img, " ", control_img
                else:
                    return img, prompt, control_img
            elif self.txt_cache_dir is not None:
                # In edit mode, prompt embeddings are image-conditioned; we store per-sample empty embeddings.
                if throw_one(self.caption_dropout_rate):
                    txt_embs = torch.load(os.path.join(self.txt_cache_dir, os.path.basename(txt_path) + '.empty.pt'))
                    return img, txt_embs['prompt_embeds'], txt_embs['prompt_embeds_mask'], control_img
                else:
                    txt_embs = torch.load(os.path.join(self.txt_cache_dir, os.path.basename(txt_path) + '.pt'))
                    return img, txt_embs['prompt_embeds'], txt_embs['prompt_embeds_mask'], control_img
            else:
                txt = os.path.basename(txt_path)
                if throw_one(self.caption_dropout_rate):
                    return img, self.cached_text_embeddings[txt + 'empty_embedding']['prompt_embeds'], \
                    self.cached_text_embeddings[txt + 'empty_embedding']['prompt_embeds_mask'], control_img
                else:
                    return img, self.cached_text_embeddings[txt]['prompt_embeds'], self.cached_text_embeddings[txt][
                        'prompt_embeds_mask'], control_img

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))
        

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
