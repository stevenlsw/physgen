import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
from PIL import Image
import kornia as K
from omegaconf import OmegaConf


def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def writing_video(rgb_list, save_path: str, frame_rate: int = 30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = rgb_list[0].shape
    out = cv2.VideoWriter(save_path, fourcc, frame_rate, (w, h))

    for img in rgb_list:
        out.write(img)

    out.release()
    return


def prep_data(data_dir):
    
    img_path = os.path.join(data_dir, "original.png")
    mask_path = os.path.join(data_dir, "mask.png")
    inpaint_path = os.path.join(data_dir, "inpaint.png")
    img = np.array(Image.open(img_path)) / 255
    mask_img = np.array(Image.open(mask_path))
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]
    inpaint_img = np.array(Image.open(inpaint_path)).astype(np.float32) / 255
    return img, mask_img, inpaint_img
    

def fit_circle_from_mask(mask_image):
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contours found in the mask image.")
        return None
    max_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(max_contour)

    center = (x, y)
    return center, radius


def composite_trans(masked_src_imgs, trans, inpaint_img, active_seg_ids):
    """params:
    src_imgs: (N, C, H, W) 
    src_seg: (H, W) original segmentaion mask
    trans: (N, 2, 3)
    inpaint_img: (C, H, W)
    active_seg_ids: (N, ) record segmentaion id
    thre: threshold for foreground segmentaion mask
    return:
    
    """
    H, W = inpaint_img.shape[:2]
    out = K.geometry.warp_affine(masked_src_imgs, trans, (H, W)) # (N, C, H, W)
    src_binary_seg = (masked_src_imgs.sum(dim=1, keepdim=True) > 0).float() # (N, 1, H, W)
    out_binary_seg = K.geometry.warp_affine(src_binary_seg,  trans, (H, W)) # (N, C, H, W)
 
    foreground_msk = out_binary_seg.sum(dim=0).sum(dim=0) > 0 # (H, W)
    seg_map = torch.zeros((H, W)).long()
    seg_map[~foreground_msk] = 0
    seg_mask = out_binary_seg.sum(dim=1).argmax(dim=0) + 1 # (H, W)
    seg_map[foreground_msk] = seg_mask[foreground_msk] # 0 is background, 1~N is the fake segmentaion id
    
    num_classes = len(active_seg_ids) + 1
    binary_seg_map = F.one_hot(seg_map, num_classes=num_classes) # (H, W, N+1)
    binary_seg_map = binary_seg_map.permute(2, 0, 1).float() # (N+1, H, W)
    binary_seg_map = binary_seg_map.unsqueeze(dim=1) # (N, 1, H, W)
    
    inpaint_img = torch.from_numpy(inpaint_img).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    input = torch.cat([inpaint_img, out], dim=0) # (N+1, C, H, W)
    
    composite = (binary_seg_map * input).sum(0) # (C, H, W)
    
    composite = composite.permute(1, 2, 0).numpy() # (H, W, C)
    final_frame = (composite*255).astype(np.uint8)
    
    final_seg_map = torch.zeros((H, W)).long()
    for idx, seg_id in enumerate(active_seg_ids):
        final_seg_map[seg_map==idx+1] = seg_id
        
    final_seg_map = final_seg_map.numpy()
    
    return final_frame, final_seg_map


def list_to_numpy(data):
    if isinstance(data, list):
        try:
            return np.array(data)
        except ValueError:
            return data
    elif isinstance(data, dict) or isinstance(data, OmegaConf):
        for key in data:
            data[key] = list_to_numpy(data[key])
    return data