import argparse
import os
import sys
import logging
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


BASE_DIR = "GeoWizard/geowizard"
sys.path.append(os.path.join(BASE_DIR))

from models.geowizard_pipeline import DepthNormalEstimationPipeline
from utils.seed_all import seed_all



if __name__=="__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Run MonoDepthNormal Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='lemonaddie/geowizard',
        help="pretrained model path from hugging face or local dir",
    )    

    parser.add_argument(
        "--domain",
        type=str,
        default='indoor',
        help="domain prediction",
    )   

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    
    # custom settings
    parser.add_argument("--input", type=str, default="../data/pig_ball", help="Input image or directory.")
    parser.add_argument("--output", type=str, default="../outputs/pig_ball", help="Output directory")
    parser.add_argument("--vis", action="store_true", help="Visualize the output.")
    
    args = parser.parse_args()
    
    checkpoint_path = args.pretrained_model_path
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size>15:
        logging.warning("long ensemble steps, low speed..")
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    domain = args.domain

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    
    if batch_size==0:
        batch_size = 1  # set default batchsize
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
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
        
    n_images = len(test_files)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found")
        exit(1)
    
    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # declare a pipeline
    pipe = DepthNormalEstimationPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)
    logging.info("loading pipeline whole successfully.")
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for rgb_path in tqdm(test_files, desc="Estimating Depth & Normal", leave=True):
            
            # Read input image
            input_image = Image.open(rgb_path)

            # predict the depth here
            pipe_out = pipe(input_image,
                denoising_steps = denoise_steps,
                ensemble_size= ensemble_size,
                processing_res = processing_res,
                match_input_res = match_input_res,
                domain = domain,
                color_map = color_map,
                show_progress_bar = True,
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored
            normal_pred: np.ndarray = pipe_out.normal_np
            normal_colored: Image.Image = pipe_out.normal_colored

            # Save as npy
            depth_npy_save_path = os.path.join(output, f"depth.npy")
            if os.path.exists(depth_npy_save_path):
                logging.warning(f"Existing file: '{depth_npy_save_path}' will be overwritten")
            np.save(depth_npy_save_path, depth_pred)

            normal_npy_save_path = os.path.join(output, f"normal.npy")
            if os.path.exists(normal_npy_save_path):
                logging.warning(f"Existing file: '{normal_npy_save_path}' will be overwritten")
            np.save(normal_npy_save_path, normal_pred)

            # Colorize
            if args.vis:
                save_intermediate = os.path.join(output, "intermediate")
                os.makedirs(output, exist_ok=True)
                os.makedirs(save_intermediate, exist_ok=True)


                depth_colored_save_path = os.path.join(save_intermediate, f"depth_vis.png")
                if os.path.exists(depth_colored_save_path):
                    logging.warning(
                        f"Existing file: '{depth_colored_save_path}' will be overwritten"
                    )
                depth_colored.save(depth_colored_save_path)

                normal_colored_save_path = os.path.join(save_intermediate, f"normal_vis.png")
                if os.path.exists(normal_colored_save_path):
                    logging.warning(
                        f"Existing file: '{normal_colored_save_path}' will be overwritten"
                    )
                normal_colored.save(normal_colored_save_path)
    print("Done.")