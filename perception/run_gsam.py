import argparse
import os
import sys
import json
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


BASE_DIR = "Grounded-Segment-Anything"
sys.path.append(os.path.join(BASE_DIR))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


def load_image(image_path, return_transform=False):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    if not return_transform:
        return image_pil, image
    else:
        return image_pil, image, transform


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def save_mask_data(output_dir, mask_list, label_list, movable_dict):
    value = 0  # 0 for background
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy() == True] = value + idx + 1
    
    mask_img = mask_img.numpy().astype(np.uint8)
    Image.fromarray(mask_img).save(os.path.join(output_dir, "mask.png"))
    print("number of classes: ", len(np.unique(mask_img)))
    
    norm = matplotlib.colors.Normalize(vmin=np.min(mask_img), vmax=np.max(mask_img))
    norm_segmentation_map = norm(mask_img)
    cmap = "tab20"
    colormap = plt.get_cmap(cmap)
    colored_segmentation_map = colormap(norm_segmentation_map)
    colored_segmentation_map = (colored_segmentation_map[:, :, :3] * 255).astype(np.uint8)
    colored_segmentation_map[mask_img == 0] = [255, 255, 255]
    
    cv2.imwrite(os.path.join(output_dir, "vis_mask.jpg"), colored_segmentation_map[..., ::-1])
    print("seg visualization saved to ", os.path.join(output_dir, "vis_mask.jpg"))
    
    json_data = [{
        'value': value,
        'label': 'background',
        'movable': False
    }]
    for label in label_list:
        value += 1
        movable = movable_dict[label] if (label in movable_dict and movable_dict[label]) else False
        json_data.append({
            'value': value,
            'label': label,
            'movable': movable
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def masks_postprocess(masks, phrases, movable_dict):
    new_masks_list = []
    new_phrases_list = []
    for idx, (label, mask) in enumerate(zip(phrases, masks)):
        mask = mask.numpy()
        
        # post-processing label
        if (label != "background") or label not in movable_dict:
            found_keys = [key for key in movable_dict if key in label]
            if found_keys:
                phrases[idx] = found_keys[0]
                label = found_keys[0]   
        
        if label == "background": # background
            continue
         
        if label in movable_dict:
            if movable_dict[label]: # movable object
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 1:
                    new_masks_list.append(mask)
                    new_phrases_list.append(label)
                else:
                    for contour in contours:
                        component_mask = np.zeros_like(mask, dtype=np.uint8)
                        cv2.drawContours(component_mask, [contour], -1, 1, thickness=cv2.FILLED)
                        new_masks_list.append(component_mask)
                        new_phrases_list.append(label)
            else:
                new_masks_list.append(mask)
                new_phrases_list.append(label)
        else: # merge into background
            continue
    
    masks = np.stack(new_masks_list, axis=0)
    masks = torch.from_numpy(masks)
    assert len(masks) == len(new_phrases_list)
    return masks, new_phrases_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, default="Grounded-Segment-Anything/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="Grounded-Segment-Anything/sam_vit_h_4b8939.pth", help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    
    # custom setting
    parser.add_argument("--box_threshold", type=float, default=0.2, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.15, help="nms threshold")
    parser.add_argument('--disable_nms', action='store_true', help='Disable nms')
    
    parser.add_argument("--input", type=str, default="../data/pig_ball", help="input path")
    parser.add_argument("--output", type=str, default=None, help="output directory")
    parser.add_argument("--prompts_path", type=str, default=None, help="prompt file under same path as input")
    
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
   
    if os.path.isfile(args.input):
        image_path = args.input
    else:
        image_path = os.path.join(args.input, "original.png")
    output = args.output
    if output is None:
        output = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    os.makedirs(output, exist_ok=True)
    if args.prompts_path is None:
        prompts_path = os.path.join(args.input, "intermediate", "obj_movable.json")
    else:
        prompts_path = args.prompts_path
    
    with open(prompts_path, "r") as f:
        movable_dict = json.load(f)
    
    model = load_model(config_file, grounded_checkpoint, device=device)
    
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    image_pil, image_dino_input, transform = load_image(image_path, return_transform=True)
    
    text_prompt = ""
    for idx, tag in enumerate(movable_dict):
        
        if idx < len(movable_dict) - 1:
            text_prompt += tag + ". "
        else:
            text_prompt += tag
                    
    print(f"Text Prompt input: {text_prompt}")
    
    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image_dino_input, text_prompt, box_threshold, text_threshold, with_logits=False, device=device)
    
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu() # boxes_filt: xyxy
    
    # use NMS to handle overlapped boxes
    if not args.disable_nms:
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        scores = scores[nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")
            
    image_input = np.array(image_pil)
    predictor.set_image(image_input)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_input.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    masks = masks.cpu().squeeze(1) # (N, H, W)
    
    
    image_input2 = image_input.copy()
    image_pil2 = Image.fromarray(image_input2)

    for idx, (label, box, mask) in enumerate(zip(pred_phrases, boxes_filt, masks)):
        mask = mask.numpy()
        if label == "background": # background
            continue
        
        if label in movable_dict and movable_dict[label]: # movable object
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 1:
                image_input2[mask == True, :] = 0 # set as black
    

    image_pil2 = Image.fromarray(image_input2)
    image_dino_input2, _ = transform(image_pil2, None)
    
    boxes_filt2, scores2, pred_phrases2 = get_grounding_output(
        model, image_dino_input2, text_prompt, 0.8 * box_threshold, 0.8 * text_threshold, with_logits=False, device=device
    )
    
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt2.size(0)):
        boxes_filt2[i] = boxes_filt2[i] * torch.Tensor([W, H, W, H])
        boxes_filt2[i][:2] -= boxes_filt2[i][2:] / 2
        boxes_filt2[i][2:] += boxes_filt2[i][:2]

    boxes_filt2 = boxes_filt2.cpu() # boxes_filt: xyxy
    
    # use NMS to handle overlapped boxes
    if len(boxes_filt2) > 0:
        
        scores2 = 0.5 * scores2 # reduce the score of the second round prediction
        if not args.disable_nms:
            boxes_combine = torch.cat([boxes_filt, boxes_filt2], dim=0)
            phrases_combine = pred_phrases + pred_phrases2
            sc_combine = torch.cat([scores, scores2], dim=0)
            
            print(f"Before NMS: {boxes_filt2.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_combine, sc_combine, 1.25 * iou_threshold).numpy().tolist()
            nms_idx = [idx for idx in nms_idx if idx >= len(boxes_filt)]
            boxes_filt2 = boxes_combine[nms_idx]
            pred_phrases2 = [phrases_combine[idx] for idx in nms_idx]
            scores2 = sc_combine[nms_idx]
            print(f"After NMS: {boxes_filt2.shape[0]} boxes")
        
        if len(boxes_filt2) > 0:
            image_input2 = np.array(image_pil2)
            predictor.set_image(image_input2)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt2, image_input2.shape[:2]).to(device)
            masks2, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )
            masks2 = masks2.cpu().squeeze(1)

            pred_phrases = pred_phrases + pred_phrases2
            masks = torch.cat([masks, masks2], dim=0)
            masks, pred_phrases = masks_postprocess(masks, pred_phrases, movable_dict)    
        else:
            masks, pred_phrases = masks_postprocess(masks, pred_phrases, movable_dict)
    else:
        masks, pred_phrases = masks_postprocess(masks, pred_phrases, movable_dict)
            
    intermediate_dir = os.path.join(output, "intermediate") # save intermediate results and visualization
    os.makedirs(intermediate_dir, exist_ok=True)
    save_mask_data(intermediate_dir, masks, pred_phrases, movable_dict)

    