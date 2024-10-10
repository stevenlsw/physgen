import os
import openai
import ast
from time import sleep
import json
from gpt.gpt_utils import find_json_response, encode_image


class GPTV_ram:
    def __init__(self, query_prompt="gpt/gpt_configs/movable/user.txt", retry_limit=3):
        
        with open(query_prompt, "r") as file:
            self.query = file.read().strip()
        self.retry_limit = retry_limit
        

    def call(self, image_path, max_tokens=300):
        query_image = encode_image(image_path)
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
                "type": "text",
                "text": f"{self.query}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_path[-3:]};base64,{query_image}"
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
                try_count += 1
                if try_count > self.retry_limit:
                    raise ValueError(f"Over Limit: Unknown response: {response}")
                else:
                    print("Retrying after 1s.")
                    sleep(1)
        return result
        
if __name__ == "__main__":
    import argparse
    import inflect
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="../data/domino/original.png")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--apikey_path", type=str, default="gpt/gpt_configs/my_apikey")
    args = parser.parse_args()
    
    with open(args.apikey_path, "r") as file:
        apikey = file.read().strip()
    
    openai.api_key = apikey
    gpt = GPTV_ram(query_prompt="gpt/gpt_configs/ram/user.txt")
    result = gpt.call(args.img_path) 
            
    if args.save_path is None:
        save_dir = os.path.join(os.path.dirname(args.img_path), "intermediate")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "obj_movable.json")
    else:
        save_path = args.save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    p = inflect.engine()
    for obj_name in result:
        singular = p.singular_noun(obj_name)
        if singular:
            result[singular] = result.pop(obj_name)
        else:
            continue
        
    with open(save_path, "w") as file:
        json.dump(result, file)
    
    print("result:", result)
    print(f"GPT4V image movable objects results saved to {save_path}")
