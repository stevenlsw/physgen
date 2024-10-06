# Perception


## Physics reasoning

- Install requirements
    ```bash
    pip install openai==0.28 ruamel.yaml
    ```
- Copy the OpenAI API key into `gpt/gpt_configs/my_apikey`.

- Physics reasoning requires the following input for each image:
     ```Shell
    image folder/ 
        ├── original.png
        ├── mask.png  # movable segmentation mask
    ```
- Run GPT-4V physical property reasoning by the following command:
    ```Shell
    python gpt_physic.py --input ../data/${name} --output ../outputs/${name}
    ```

- The output `physics.yaml` contains the physical properties and primitive shape of each object segment in the image. Note GPT-4V outputs may vary for different runs and differ from the original setting in `data/${name}/sim.yaml`. Users could adjust accordingly to each run output.


## Depth and Normal Estimation
- We use [GeoWizard](https://github.com/fuxiao0719/GeoWizard) to estimate depth and normal of input image. Follow [GeoWizard setup](https://github.com/fuxiao0719/GeoWizard/blob/main/README.md#%EF%B8%8F-setup) to install requirements. Recommend to create a new conda environment.

- Run GeoWizard on input image
    ```Shell
    python run_depth_normal.py --input ../data/${name} --output ../outputs/${name} --vis
    ```
- `depth.npy` and `normal.npy` are saved in `outputs/${name}`. Visualization of depth and normal are saved in `outputs/${name}/intermediate`.

    | **Input** | **Normal** | **Depth** 
    |:---------:|:----------------:|:----------:|
    | <img src="../data/pig_ball/original.png" alt="input" width="100"/> | <img src="../data/pig_ball/intermediate/normal_vis.png" alt="normal" width="100"/> | <img src="../data/pig_ball/intermediate/depth_vis.png" alt="normal" width="100"/> |


## Albedo and shading estimation
- We use [Intrinsic](https://github.com/compphoto/Intrinsic/tree/d9741e99b2997e679c4055e7e1f773498b791288) to infer albedo and shading of input image. Follow [Intrinsic setup](https://github.com/compphoto/Intrinsic/tree/d9741e99b2997e679c4055e7e1f773498b791288?tab=readme-ov-file#setup) to install requirements. Recommend to create a new conda environment.

- Run Intrinsic decomposition on input image
    ```Shell
    python run_albedo_shading.py --input ../data/${name} --output ../outputs/${name} --vis
    ```

- `shading.npy` are saved in `outputs/${name}`. Visualization of albedo and shading are saved in `outputs/${name}/intermediate`.

    | **Input** | **Albedo** | **Shading** 
    |:---------:|:----------------:|:----------:|
    | <img src="../data/pig_ball/original.png" alt="input" width="100"/> | <img src="../data/pig_ball/intermediate/albedo_vis.png" alt="albedo" width="100"/> | <img src="../data/pig_ball/intermediate/shading_vis.png" alt="shading" width="100"/> |

- [Intrinsic](https://github.com/compphoto/Intrinsic) has released updated trained model with better results. Feel free to use the updated model or any other model for better performance.
