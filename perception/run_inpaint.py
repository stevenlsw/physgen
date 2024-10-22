import sys
import os
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

BASE_DIR = "Inpaint-Anything"
sys.path.append(os.path.join(BASE_DIR))

from lama_inpaint import inpaint_img_with_lama


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=20):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="../data/pig_ball", help="input path")
    parser.add_argument("--output", type=str, default=None, help="output directory")
    parser.add_argument("--dilate_kernel_size", type=int, default=20, help="Dilate kernel size")
    parser.add_argument("--lama_config", type=str, default="Inpaint-Anything/lama/configs/prediction/default.yaml",
                        help="The path to the config file of lama model.")
    parser.add_argument("--lama_ckpt", type=str, default="Inpaint-Anything/pretrained_models/big-lama", 
                        help="The path to the lama checkpoint.")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        image_path = args.input
    else:
        image_path = os.path.join(args.input, "original.png")
    output = args.output
    if output is None:
        output = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    os.makedirs(output, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = os.path.join(args.input, "original.png")
    img = load_img_to_array(image_path)
    seg_path = os.path.join(args.input, 'mask.png')
    seg_mask = cv2.imread(seg_path, 0)    
    mask = ((seg_mask > 0)*255).astype(np.uint8)
    
    if args.dilate_kernel_size is not None:
        mask = dilate_mask(mask, args.dilate_kernel_size) 

    img_inpainted = inpaint_img_with_lama(img, mask, args.lama_config, args.lama_ckpt, device=device)
    save_array_to_img(img_inpainted, os.path.join(output, "inpaint.png"))