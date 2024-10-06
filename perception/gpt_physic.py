import os
import ast
from time import sleep
import cv2
import numpy as np
import pymunk
import openai
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from gpt.gpt_utils import find_json_response, encode_image, fit_polygon_from_mask, fit_circle_from_mask, compute_iou, is_mask_truncated


class GPTV_physics:
    def __init__(self, query_prompt="gpt/gpt_configs/physics/user.txt", retry_limit=0):
        with open(query_prompt, "r") as file:
            self.query = file.read().strip()
        self.retry_limit = retry_limit
     
    def call(self, image_path, mask, max_tokens=300, tmp_dir="./"):
        mask_path = os.path.join(tmp_dir, "tmp_msk.png")
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask)
        query_image = encode_image(image_path)
        mask_image = encode_image(mask_path)
        
        try_count = 0
        while True:
            response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{
            "role": "system",
            "content": self.query
            },
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_path[-3:]};base64,{query_image}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{mask_path[-3:]};base64,{mask_image}"
                }
                },
            ]
            }
            ],
            seed=100,
            max_tokens=max_tokens,
            )
            response = response["choices"][0]["message"]["content"]
            try:
                result = find_json_response(response)
                result = ast.literal_eval(result.replace(' ', '').replace('\n', ''))
                break
            except:
                print(f"Unknown response: {response}")
                try_count += 1
                if try_count > self.retry_limit:
                    raise ValueError(f"Over Limit: Unknown response: {response}")
                else:
                    print("Retrying after 1s.")
                    sleep(1)
        os.remove(mask_path)
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../data/pig_ball")
    parser.add_argument("--output", type=str, default="../outputs/pig_ball")
    parser.add_argument("--apikey_path", type=str, default="gpt/gpt_configs/my_apikey")
    args = parser.parse_args()
    
    with open(args.apikey_path, "r") as file:
        apikey = file.read().strip()
    
    openai.api_key = apikey
    gpt = GPTV_physics(query_prompt="gpt/gpt_configs/physics/user.txt")
    
    image_path = os.path.join(args.input, "original.png")
    mask_path = os.path.join(args.input, "mask.png")
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "physics.yaml")
    
    seg_mask = cv2.imread(mask_path, 0)
    seg_ids = np.unique(seg_mask)
    obj_info_list = {}
    for seg_id in seg_ids:
        if seg_id == 0:
            continue
        obj_info = {}
        mask = (seg_mask == seg_id)
        # fit primitive
        center, radius = fit_circle_from_mask((mask * 255).astype(np.uint8))
        center = tuple(map(int, center))
        radius = int(radius)
        pred_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(pred_mask, center, radius, (255), thickness=-1)
        area = np.count_nonzero(mask)
        pred_mask = pred_mask > 0
        iou = compute_iou(mask, pred_mask)
        if iou > 0.85:
            obj_info["primitive"] = 'circle'
        else:
            if is_mask_truncated(mask):
                continue
            obj_info["primitive"] = 'polygon'
            points = fit_polygon_from_mask(mask)
            points = tuple(map(tuple, points))
            polygon = pymunk.Poly(None, points)
            area = polygon.area
        
        result = gpt.call(image_path, mask)
        for key in result:
            if key == "mass":
                if obj_info['primitive'] == 'polygon':
                    density = result["mass"] / area
                    obj_info["mass"] = None
                    obj_info["density"] = density
                else:
                    obj_info["mass"] = result["mass"]
                    obj_info["density"] = None
            else:
                obj_info[key] = result[key] 
        
        obj_info_list[int(seg_id)] = obj_info
                 
    yaml = YAML()
    yaml_data = CommentedMap()
    
    yaml_data['obj_info'] = obj_info_list
    yaml_data.yaml_set_comment_before_after_key('obj_info', before="physics properties of each object")
    
    with open(save_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)
    
    print(f"GPT-4V Physics reasoning results saved to {save_path}")