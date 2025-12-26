#! /bin/bash
python producer.py \
    --pretrained_model "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit" \
    --img_dir "../../images2/" \
    --control_dir "../../control_images2/" \
    --target_area 512*512 \
    --output_dir "cache" \
    --prompt_with_image