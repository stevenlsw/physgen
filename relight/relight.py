import os 
import argparse
import numpy as np
import copy
import torch.nn.functional as F
from tqdm import tqdm
import torch
import kornia as K

from relight_utils import get_light_coeffs,  generate_shd, writing_video


class Relight:
    def __init__(self, img, mask_img, nrm, shd): 
        # nrm: +X right, +Y down, +Z from screen to me, between 0 and 1    
        self.img = img # RGB between 0 and 1
        self.mask_img = mask_img
        self.nrm = nrm
        
        src_msks = []
        seg_ids = np.unique(mask_img)
        seg_ids.sort()
        for seg_id in np.unique(mask_img):
            if seg_id == 0:
                continue
            src_msk = (mask_img == seg_id)
            src_msks.append(src_msk)
        src_msks = np.stack(src_msks, axis=0) # (N, H, W)
        self.src_msks = torch.from_numpy(src_msks)
        self.num_classes = src_msks.shape[0]
        
        self.coeffs = get_light_coeffs(shd, self.nrm, self.img) # lambertian fitting
        shd[self.mask_img > 0] = generate_shd(self.nrm, self.coeffs, self.mask_img > 0)
        self.alb = (self.img ** 2.2) / shd[:, :, None].clip(1e-4)
        self.shd = shd
        
        
    def relight(self, comp_img, target_obj_mask, trans):
        # target_obj_mask (H, W) segmentation map
        # trans: (N, 2, 3)
        binary_seg_mask = []
        for seg_id in np.unique(self.mask_img):
            if seg_id == 0:
                continue
            seg_mask = target_obj_mask == seg_id
            binary_seg_mask.append(seg_mask)
            
        binary_seg_mask = np.stack(binary_seg_mask, axis=0) # (N, H, W)
        binary_seg_mask = torch.from_numpy(binary_seg_mask[:, None, :, :]) # (N, 1, H, W)
        
        assert binary_seg_mask.shape[0] == self.num_classes, "number of segmentation ids should be equal to the number of foreground objects"
        
        src_normal = torch.from_numpy(self.nrm).unsqueeze(dim=0).repeat(self.num_classes, 1, 1, 1) # (N, H, W, 3)
        rot_matrix = trans[:, :2, :2] # (N, 2, 2)
        full_rot = torch.cat([rot_matrix, torch.zeros(self.num_classes, 1, 2)], dim=1) # (N, 3, 2)
        full_rot = torch.cat([full_rot, torch.tensor([0, 0, 1]).unsqueeze(dim=0).unsqueeze(dim=-1).repeat(self.num_classes, 1, 1)], dim=2) # (N, 3, 3)
        
        nrm = (src_normal * 2.0) - 1.0
        nrm = nrm.reshape(self.num_classes, -1, 3) # (N, H*W, 3)
        trans_nrm = torch.bmm(nrm, full_rot) # the core rot is created by kornia, so there is no transpose # (N, H*W, 3)
        trans_nrm = (trans_nrm +1) / 2.0
        trans_nrm = trans_nrm.reshape(self.num_classes, self.img.shape[0], self.img.shape[1], 3) # (N, H, W, 3)
            
        trans_nrm = trans_nrm.permute(0, 3, 1, 2) # (N, 3, H, W)
        out = K.geometry.warp_affine(trans_nrm, trans, (mask_img.shape[0], mask_img.shape[1])) # (N, 3, H, W)
        target_nrm = (out * binary_seg_mask.float()).sum(dim=0) # (3, H, W)
        target_nrm = target_nrm.permute(1, 2, 0).numpy() # (H, W, 3)

        comp_nrm = copy.deepcopy(self.nrm) 
        comp_nrm[target_obj_mask > 0] = target_nrm[target_obj_mask > 0]
         
        comp_shd = copy.deepcopy(self.shd)
        comp_shd[target_obj_mask > 0] = generate_shd(comp_nrm, self.coeffs, target_obj_mask > 0)
        comp_alb = (comp_img ** 2.2) / comp_shd[:, :, None].clip(1e-4)
        
        # compose albedo from src
        assert trans.shape[0] == self.num_classes, "number of transformations should be equal to the number of foreground objects"
        src_alb = torch.from_numpy(self.alb).permute(2, 0, 1).unsqueeze(dim=0).repeat(self.num_classes, 1, 1, 1) # (N, 3, H, W)
        out = K.geometry.warp_affine(src_alb, trans, (self.mask_img.shape[0], self.mask_img.shape[1])) # (N, 3, H, W)
        
        foreground_alb = (out * binary_seg_mask.float()).sum(dim=0) # (3, H, W)
        foreground_alb = foreground_alb.permute(1, 2, 0).numpy() # (H, W, 3)
        comp_alb[target_obj_mask > 0, :] = foreground_alb[target_obj_mask > 0, :]
        
        compose_img = ((comp_alb * comp_shd[:, :, None])** (1/2.2)).clip(0, 1)

        return compose_img

 
def prepare_input(video_path, mask_video_path, trans_list_path):
    comp_video = torch.load(video_path) # (f, 3, H, W) (16, 3, 512, 512) between 0 and 255
    if comp_video.max().item() > 1.:
        comp_video = comp_video / 255.0
    if comp_video.shape[1] ==3: # (f, 3, H, W) -> (f, H, W, 3)
        comp_video = comp_video.permute(0, 2, 3, 1)
    T, H, W = comp_video.shape[:3]
    obj_masks = torch.load(mask_video_path).squeeze().float() # (f, H, W)
    trans_list = torch.load(trans_list_path)
    if trans_list.ndim == 3: # (f, 2, 3) -> (f, 1, 2, 3)
        trans_list = trans_list.unsqueeze(dim=1)
    if trans_list.shape[-2] == 3: # (f, *, 3, 3) -> (f, *, 2, 3)
        trans_list = trans_list[:, :, :2, :]
    assert comp_video.shape[0] == obj_masks.shape[0] == trans_list.shape[0], "video and mask should have the same length"
    comp_video = comp_video.numpy()
    if obj_masks.ndim == 4:
        obj_masks = obj_masks[:, :, 0] # (f, H, W)
    obj_masks = obj_masks.numpy() # (f, H, W)
    return comp_video, obj_masks, trans_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception_input", type=str, default="../data/pool", help='input dir')
    parser.add_argument('--previous_output', type=str, default="../outputs/pool", help='previous output dir')
    
    args = parser.parse_args()
    
    perception_input = args.perception_input
    previous_output = args.previous_output
    
    video_path = os.path.join(previous_output, "composite.pt")
    mask_video_path = os.path.join(previous_output, "mask_video.pt")
    trans_list_path = os.path.join(previous_output, "trans_list.pt")
    comp_video, obj_masks, trans_list = prepare_input(video_path, mask_video_path, trans_list_path)

    normal_path = os.path.join(perception_input, "normal.npy")
    shading_path = os.path.join(perception_input, "shading.npy")
    normal = np.load(normal_path) # (H, W, 3) between -1 and 1
    # convert geowizard normal to OMNI normal
    normal[:, :, 0]= -normal[:, :, 0]
    normal = (normal + 1) / 2.0
    shading = np.load(shading_path)
        
    
    output = args.previous_output
    os.makedirs(output, exist_ok=True)
    
    T = comp_video.shape[0]
    img = comp_video[0] # (H, W, 3)
    mask_img = obj_masks[0]
    model = Relight(img, mask_img, normal, shading)
   
    relight_list = [img]
    for time_idx in tqdm(range(1, T)):
        comp_img = comp_video[time_idx]
        target_obj_mask = obj_masks[time_idx] # segmentation msk
        trans = trans_list[time_idx] # (*, 2, 3)
        
        compose_img = model.relight(comp_img, target_obj_mask, trans)
        relight_list.append(compose_img)
       
    relight_video = np.stack(relight_list, axis=0)
    torch.save(torch.from_numpy(relight_video).permute(0, 3, 1, 2), f'{output}/relight.pt') # (0, 1)
    relight_video = (relight_video * 255).astype(np.uint8)
    writing_video(relight_video[..., ::-1], f'{output}/relight.mp4', frame_rate=7)
    print('done!')