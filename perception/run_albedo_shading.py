import argparse
import os
import glob
import numpy as np
from tqdm.auto import tqdm
from PIL import Image


from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models


def load_image(path, bits=8):
    np_arr = np.array(Image.open(path)).astype(np.single)
    return np_arr / float((2 ** bits) - 1)


def np_to_pil(img, bits=8):
    if bits == 8:
        int_img = (img * 255).astype(np.uint8)
    if bits == 16:
        int_img = (img * ((2 ** 16) - 1)).astype(np.uint16)

    return Image.fromarray(int_img)


def view_scale(img, p=100):
    return (img / np.percentile(img, p)).clip(0, 1)


def view(img, p=100):
    return view_scale(img ** (1/2.2), p=p)


def uninvert(x, eps=0.001, clip=True):
    if clip:
        x = x.clip(eps, 1.0)

    out = (1.0 / x) - 1.0
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../data/pig_ball", help="Input image or directory.")
    parser.add_argument("--output", type=str, default="../outputs/pig_ball", help="Output directory.")
    parser.add_argument("--vis", action="store_true", help="Visualize the results.")
    
    args = parser.parse_args()
   
    input = args.input
    EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
    if os.path.isfile(input) and os.path.splitext(input)[1].lower() in EXTENSION_LIST:
        test_files = [input]
        output = args.output
        if output is None:
            output = os.path.dirname(input)
        os.makedirs(output, exist_ok=True)
    else: 
        test_files = [os.path.join(args.input, "original.png")]
        output = args.output
        if output is None:
            output = input
    
    model = load_models('paper_weights')
    for rgb_path in tqdm(test_files, desc="Estimating Albedo & Shading", leave=True):
        
        # Read input image
        img = load_image(rgb_path, bits=8)
        # run the model on the image using R_0 resizing
        
        results = run_pipeline(model, img, resize_conf=0.0, maintain_size=True)

        albedo = results['albedo']
        inv_shd = results['inv_shading']
        
        shd = uninvert(inv_shd)
        shd_save_path = os.path.join(output, "shading.npy")
        np.save(shd_save_path, shd)
        
        if args.vis:
            save_intermediate = os.path.join(output, "intermediate")
            os.makedirs(save_intermediate, exist_ok=True)

            alb_save_path = os.path.join(save_intermediate, "albedo.npy")
            np.save(alb_save_path, albedo)
            np_to_pil(albedo).save(os.path.join(save_intermediate, 'albedo_vis.png'))
            np_to_pil(view(shd)).save(os.path.join(save_intermediate, 'shading_vis.png'))
        
        
        