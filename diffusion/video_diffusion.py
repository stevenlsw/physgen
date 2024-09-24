import os
import sys
import numpy as np
from omegaconf import OmegaConf
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from einops import rearrange
from tqdm.auto import tqdm
from diffusers.models import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available


sys.path.insert(0, "SEINE")
from datasets import video_transforms
from diffusion import create_diffusion
from models.clip import TextEmbedder
from models import get_models
from utils import mask_generation_before


def prepare_input(relight_video_path, mask_video_path, 
                  image_h, image_w, latent_h, latent_w, device, use_fp16):
    relight_video = torch.load(relight_video_path).to(device)  # (f, 3, H, W)
    if relight_video.max() > 1.:
        relight_video = relight_video / 255.0
    if relight_video.shape[1] !=3: # (f, H, W, 3) -> (f, 3, H, W)
        relight_video = relight_video.permute(0, 3, 1, 2) # (f, 3, H, W)
    if relight_video.shape[-2:] != (image_h, image_w):
        relight_video = F.interpolate(relight_video, size=(image_h, image_w))
    relight_video = 2 * relight_video - 1 # [-1, 1]
    if use_fp16:
        relight_video = relight_video.to(dtype=torch.float16)
        
    mask_video = torch.load(mask_video_path).to(device).float() # (f, 1, H, W)
    if mask_video.ndim == 3: # (f, H, W) -> (f, 4, H, W)
        mask_video = mask_video.unsqueeze(1).repeat(1, 4, 1, 1) 
    elif mask_video.ndim == 4 and mask_video.shape[1] == 1: # (f, 1, H, W) -> (f, 4, H, W)
        mask_video = mask_video.repeat(1, 4, 1, 1) 
    elif mask_video.ndim == 4 and mask_video.shape[0] == 1: # (1, f, H, W) -> (f, 4, H, W)
        mask_video = mask_video.repeat(4, 1, 1, 1).permute(1, 0, 2, 3) # (f, 4, H, W)
    if mask_video.shape[-2:] != (latent_h, latent_w):
        mask_video = F.interpolate(mask_video, size=(latent_h, latent_w)) # (f, 4, h, w)
    mask_video = mask_video > 0.5
    mask_video = rearrange(mask_video, 'f c h w -> c f h w').contiguous()
    mask_video = mask_video.unsqueeze(0) # (1, 4, f, h, w)
    return relight_video, mask_video
    
    
def get_input(image_path, image_h, image_w, mask_type="first1", num_frames=16):
    transform_video = transforms.Compose([
            video_transforms.ToTensorVideo(), 
            video_transforms.ResizeVideo((image_h, image_w)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
    print(f'loading video from {image_path}')
    _, full_file_name = os.path.split(image_path)
    file_name, extension = os.path.splitext(full_file_name)
    if extension == '.jpg' or extension == '.jpeg' or extension == '.png':
        print("loading the input image")
        video_frames = []
        num = int(mask_type.split('first')[-1])
        first_frame = torch.as_tensor(np.array(Image.open(image_path), dtype=np.uint8, copy=True)).unsqueeze(0)
        for i in range(num):
            video_frames.append(first_frame)
        num_zeros = num_frames-num
        for i in range(num_zeros):
            zeros = torch.zeros_like(first_frame)
            video_frames.append(zeros)
        n = 0
        video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2) # f,c,h,w
        video_frames = transform_video(video_frames)
        return video_frames, n
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model checkpoint
    parser.add_argument("--ckpt", default="SEINE/pretrained/seine.pt")
    parser.add_argument("--pretrained_model_path", default="SEINE/pretrained/stable-diffusion-v1-4")
    # Model config
    parser.add_argument('--model', type=str, default='UNet', help='Model architecture to use.')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to process.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Resolution of the input images.')

    # Model speedup config
    parser.add_argument('--use_fp16', type=bool, default=True, help='Use FP16 for faster inference. Set to False if debugging with video loss.')
    parser.add_argument('--enable_xformers_memory_efficient_attention', type=bool, default=True, help='Enable xformers memory efficient attention.')

    # Sample config
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--run_time', type=int, default=13, help='Run time of the model.')
    parser.add_argument('--cfg_scale', type=float, default=8.0, help='Configuration scale factor.')
    parser.add_argument('--sample_method', type=str, default='ddim', choices=['ddim', 'ddpm'], help='Sampling method to use.')
    parser.add_argument('--do_classifier_free_guidance', type=bool, default=True, help='Enable classifier-free guidance.')
    parser.add_argument('--mask_type', type=str, default="first1", help='Type of mask to use.')
    parser.add_argument('--use_mask', type=bool, default=True, help='Whether to use a mask.')
    parser.add_argument('--num_sampling_steps', type=int, default=50, help='Number of sampling steps to perform.')
    parser.add_argument('--prompt', type=str, default=None, help='input prompt')
    parser.add_argument('--negative_prompt', type=str, default="", help='Negative prompt to use.')
    
    # Video diffusion config
    parser.add_argument('--denoise_strength', type=float, default=0.4, help='Denoise strength parameter.')
    parser.add_argument('--stop_idx', type=int, default=5, help='Stop index for the diffusion process.')
    parser.add_argument('--perception_input', type=str, default="../data/pool", help='input dir')
    parser.add_argument('--previous_output', type=str, default="../outputs/pool", help='previous output dir')
    
    # Parse the arguments
    args = parser.parse_args()

    output = args.previous_output
    os.makedirs(output, exist_ok=True)

    if args.seed:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path using --ckpt <path>")

    latent_h = args.image_size[0] // 8
    latent_w = args.image_size[1] // 8
    image_h = args.image_size[0]
    image_w = args.image_size[1]
    latent_h = latent_h
    latent_w = latent_w
    print('loading model')
    model = get_models(args).to(device)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    ckpt_path = args.ckpt 
    state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['ema']
    model.load_state_dict(state_dict)
    print('loading succeed')
    model.eval()
    
    pretrained_model_path = args.pretrained_model_path
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
    text_encoder = TextEmbedder(pretrained_model_path).to(device)
    
    sim_config_path = os.path.join(args.perception_input, "sim.yaml")
    config = OmegaConf.load(sim_config_path)
    objects = config.obj_info
    object_names = []
    for seg_id in objects:
        name = objects[seg_id]['label']
        object_names.append(name)
    object_names = list(set(object_names))
    if args.prompt is None:
        prompt = ", ".join(object_names)
    else:
        prompt = args.prompt
    print(f"input prompt: {prompt}")
    denoise_strength = getattr(config, 'denoise_strength', args.denoise_strength)
    
    if args.use_fp16:
        print('Warnning: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        
   
    relight_video_path = os.path.join(args.previous_output, "relight.pt")
    mask_video_path = os.path.join(args.previous_output, "mask_video.pt")
    relight_video, mask_video = prepare_input(relight_video_path, mask_video_path, image_h, image_w, latent_h, latent_w, device, args.use_fp16)
    
    with torch.no_grad():
        ref_latent = vae.encode(relight_video).latent_dist.sample().mul_(0.18215) # (f, 4, h, w)
        ref_latent = ref_latent.permute(1, 0, 2, 3).contiguous().unsqueeze(0) # (1, 4, f, h, w)
    
    image_path = os.path.join(args.perception_input, "original.png")
    video, reserve_frames = get_input(image_path, image_h, image_w, args.mask_type, args.num_frames)
    video_input = video.unsqueeze(0).to(device) # b,f,c,h,w
    b,f,c,h,w=video_input.shape
    mask = mask_generation_before(args.mask_type, video_input.shape, video_input.dtype, device) # b,f,c,h,w [1, 16, 3, 512, 512])
    masked_video = video_input * (mask == 0)

    if args.use_fp16:
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
   
    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
    
    if args.do_classifier_free_guidance:
        masked_video = torch.cat([masked_video] * 2)
        mask = torch.cat([mask] * 2)
        prompt_all = [prompt] + [args.negative_prompt] 
    else:
        masked_video = masked_video
        mask = mask
        prompt_all = [prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, 
                            class_labels=None, 
                            cfg_scale=args.cfg_scale,
                            use_fp16=args.use_fp16) # tav unet
    
    indices = list(range(diffusion.num_timesteps))[::-1]
    noise_level = int(denoise_strength * diffusion.num_timesteps)
    indices = indices[-noise_level:]
    
    latent = diffusion.q_sample(ref_latent, torch.tensor([indices[0]], device=device))
    latent = torch.cat([latent] * 2)
    stop_idx = args.stop_idx
    
    for idx, i in tqdm(enumerate(indices)):
            t = torch.tensor([indices[idx]] * masked_video.shape[0], device=device)
            with torch.no_grad():
                out = diffusion.ddim_sample(
                    model.forward_with_cfg,
                    latent,
                    t,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                    mask=mask,
                    x_start=masked_video,
                    use_concat=args.use_mask,
                )
               
                # update latent
                latent = out["sample"]
                if idx < len(indices)-stop_idx:
                    x = diffusion.q_sample(ref_latent, torch.tensor([indices[idx+1]], device=device))
                    pred_xstart = out["pred_xstart"]
    
                    weight = min(idx / len(indices), 1.0)
                    latent = (1 - mask_video.float()) * latent + mask_video.float() * ((1 - weight) * x + weight * latent)
                
    if args.use_fp16:
        latent = latent.to(dtype=torch.float16)
    latent = latent[0].permute(1, 0, 2, 3).contiguous() # (f, 4, h, w)
    video_output = vae.decode(latent / 0.18215).sample
    video_ = ((video_output * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
    save_video_path = os.path.join(output,  f'final_video.mp4')
    torchvision.io.write_video(save_video_path, video_, fps=7)
    print("all done!")