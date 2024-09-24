<br />
<p align="center">

  <h1 align="center">PhysGen: Rigid-Body Physics-Grounded <br>Image-to-Video Generation</h1>

  <p align="center">
   ECCV, 2024
    <br />
    <a href="https://stevenlsw.github.io"><strong>Shaowei Liu</strong></a>
    ·
    <a href="https://jason718.github.io/"><strong>Zhongzheng Ren</strong></a>
    ·
    <a href="https://saurabhg.web.illinois.edu/"><strong>Saurabh Gupta*</strong></a>
    ·
    <a href="https://shenlong.web.illinois.edu/"><strong>Shenlong Wang*</strong></a>
    ·
  </p>

<p align="center"> 
<img src="assets/demo.gif" alt="Demo GIF" />
</p>

  <p align="center">
    <a href='https://drive.google.com/file/d/1JPtu11nhonYOydFl2Btf_y_n9zVT5QPP/view'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://stevenlsw.github.io/physgen/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    <a href='https://youtu.be/lCc1rHePEFQ' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
    </a>
  </p>

</p>
<br />

This repository contains the pytorch implementation for the paper [PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation](https://stevenlsw.github.io/physgen/), ECCV 2024. In this paper, we present a novel training-free image-to-video generation pipeline integrates physical simulation and generative video diffusion prior.

## Overview
![overview](assets/method.png)

## 📄 Table of Contents

- [Installation](#installation)
- [Quick Demo](#quick-demo)
- [Perception](#perception)
- [Simulation](#simulation)
- [Rendering](#rendering)
- [All-in-One command](#all-in-one-command)
- [Citation](#citation)


## Installation
- Clone this repository: 
    ```Shell
    git clone --recurse-submodules https://github.com/stevenlsw/physgen.git
    cd physgen
    ```
- Install requirements by the following commands:
    ```Shell
    conda create -n physgen python=3.9
    conda activate physgen
    pip install -r requirements.txt
    ```

## Quick Demo 
- Run image space dynamics simulation in just **3** seconds **without GPU and any displace device and additional setup** required!
    ```Shell
    export PYTHONPATH=$(pwd)
    name="pool"
    python simulation/animate.py --data_root data --save_root outputs --config data/${name}/sim.yaml 
    ```
- The output video should be saved in `outputs/${name}/composite.mp4`. Try set `name` to be `domino`, `balls`, `pig_ball` and `car` for other scenes exploration. The example outputs are shown below:

  <table>
    <tr>
      <td style="text-align:center;width: 33%"><b>Input Image</b></td>
      <td style="text-align:center;width: 33%"><b>Simulation</b></td>
      <td style="text-align:center;width: 33%"><b>Output Video</b></td>
    </tr>
    <tr>
      <td><img src="data/pool/original.png" alt="Pool Original Image" width="200"></td>
      <td><img src="assets/pool_sim.gif" alt="Pool Simulation GIF" width="200"></td>
      <td><img src="assets/pool_composite.gif" alt="Pool Composite GIF" width="200"></td>
    </tr>
    <tr>
      <td><img src="data/domino/original.png" alt="Domino Original Image" width="200"></td>
      <td><img src="assets/domino_sim.gif" alt="Domino Simulation GIF" width="200"></td>
      <td><img src="assets/domino_composite.gif" alt="Domino Composite GIF" width="200"></td>
    </tr>
  </table>



## Perception

<table>
  <tr>
    <th style="width: 16%">Input</th>
    <th style="width: 16%">Segmentation</th>
    <th style="width: 16%">Normal</th>
    <th style="width: 16%">Albedo</th>
    <th style="width: 16%">Shading</th>
    <th style="width: 16%">Inpainting</th>
  </tr>
  <tr>
    <td><img src="data/pig_ball/original.png" alt="input" width="100"></td>
    <td><img src="data/pig_ball/vis.png" alt="segmentation" width="100"></td>
    <td><img src="data/pig_ball/intermediate/normal_vis.png" alt="normal" width="100"></td>
    <td><img src="data/pig_ball/intermediate/albedo_vis.png" alt="albedo" width="100"></td>
    <td><img src="data/pig_ball/intermediate/shading_vis.png" alt="shading" width="100"></td>
    <td><img src="data/pig_ball/inpaint.png" alt="inpainting" width="100"></td>
  </tr>
</table>


## Simulation
- Simulation requires the following input for each image:
  ```Shell
  image folder/ 
    ├── original.png
    ├── mask.png  # segmentation mask
    ├── inpaint.png # background inpainting
    ├── sim.yaml # simulation configuration file
  ```

- `sim.yaml` specify the physical properties of each object and initial conditions (force and speed on each object). Please see `data/pig_ball/sim.yaml` for an example. Set `display` to `true` to visualize the simulation process with display device, set `save_snapshot` to `true` to save the simulation snapshots.
- Run the simulation by the following command:
  ```Shell
  cd simulation
  python animate.py --data_root ../data --save_root ../outputs --config ../data/${name}/sim.yaml
  ```
- The outputs are saved in `outputs/${name}` as follows:
  ```Shell
  output folder/
    ├── history.pkl # simulation history
    ├── composite.mp4 # composite video
    |── composite.pt # composite video tensor
    ├── mask_video.pt # foreground masked video tensor
    ├── trans_list.pt # objects transformation list tensor
  ```

## Rendering

### Relighting
- Relighting requires the following input:
  ```Shell
  image folder/ # 
    ├── normal.npy # normal map
    ├── shading.npy # shading map by intrinsic decomposition
  previous output folder/
    ├── composite.pt # composite video
    ├── mask_video.pt # foreground masked video tensor
    ├── trans_list.pt # objects transformation list tensor

  ```
- The `perception_input` is the image folder contains the perception result. The `previous_output` is the output folder from the previous simulation step.
- Run the relighting by the following command:
  ```Shell
  cd relight
  python relight.py --perception_input ../data/${name} --previous_output ../outputs/${name} 
  ```
- The output `relight.mp4` and `relight.pt` is the relighted video and tensor. 
- Compare between composite video and relighted video:
  <table>
    <tr>
      <td style="text-align:center;width: 33%"><b>Input Image</b></td>
      <td style="text-align:center;width: 33%"><b>Composite Video</b></td>
      <td style="text-align:center;width: 33%"><b>Relight Video</b></td>
    </tr>
    <tr>
      <td style="text-align:center;"><img src="data/pig_ball/original.png" alt="Original Input Image" width="200"></td>
      <td style="text-align:center;"><img src="assets/pig_ball_composite.gif" alt="Pig Ball Composite GIF" width="200"></td>
      <td style="text-align:center;"><img src="assets/pig_ball_relight.gif" alt="Pig Ball Relight GIF" width="200"></td>
    </tr>
  </table>


### Video Diffusion Rendering
- Download the [SEINE](https://github.com/Vchitect/SEINE/) model follow [instruction](https://github.com/Vchitect/SEINE/tree/main?tab=readme-ov-file#download-our-model-and-t2i-base-model)

  ```Shell
  # install git-lfs beforehand
  mkdir -p diffusion/SEINE/pretrained
  git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 diffusion/SEINE/pretrained
  wget -P diffusion/SEINE/pretrained https://huggingface.co/Vchitect/SEINE/resolve/main/seine.pt
  ```

- The video diffusion rendering requires the following input:
  ```Shell
  image folder/ # 
    ├── original.png # input image
    ├── sim.yaml # simulation configuration file (optional)
  previous output folder/
    ├── relight.pt # composite video
    ├── mask_video.pt # foreground masked video tensor
  ```
- Run the video diffusion rendering by the following command:
  ```Shell
  cd diffusion
  python video_diffusion.py --perception_input ../data/${name} --previous_output ../outputs/${name} 
  ```
  `denoise_strength` and `prompt` could be adjusted in the above script. `denoise_strength` controls the amount of noise added, 0 means no denoising, 1 means denoise from scratch with lots of variance to the input image. `prompt` is the input prompt for video diffusion model, we use default foreground object names from perception model as prompt.


- The output `final_video.mp4` is the rendered video.

- Compare between relight video and diffuson rendered video:
    <table>
      <tr>
        <td style="text-align:center;width: 33%"><b>Input Image</b></td>
        <td style="text-align:center;width: 33%"><b>Composite Video</b></td>
        <td style="text-align:center;width: 33%"><b>Relight Video</b></td>
      </tr>
      <tr>
        <td style="text-align:center;"><img src="data/car/original.png" alt="Original Input Image" width="200"></td>
        <td style="text-align:center;"><img src="assets/car_relight.gif" alt="Pig Ball Composite GIF" width="200"></td>
        <td style="text-align:center;"><img src="assets/car_final.gif" alt="Pig Ball Relight GIF" width="200"></td>
      </tr>
    </table>


## All-in-One command
We integrate the simulation, relighting and video diffusion rendering in one script. Please follow the [Video Diffusion Rendering](#video-diffusion-rendering) to download the SEINE model first.
```Shell
bash scripts/run_demo.sh ${name}
```


## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@inproceedings{liu2023contactgen,
  title={ContactGen: Generative Contact Modeling for Grasp Generation},
  author={Liu, Shaowei and Zhou, Yang and Yang, Jimei and Gupta, Saurabh and Wang, Shenlong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```


## Acknowledgement
* [Grounded-Segment-Anything
](https://github.com/IDEA-Research/Grounded-Segment-Anything) for segmentation in [perception](#perception) 
* [GeoWizard
](https://github.com/fuxiao0719/GeoWizard) for depth and normal estimation in [perception](#perception) 
* [Intrinsic](https://github.com/compphoto/Intrinsic/) for intrinsic image decomposition in in [perception](#perception) 
* [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) for image inpainting in [perception](#perception)
* [Pymunk](https://github.com/viblo/pymunk) for physics simulation in [simulation](#simulation)
* [SEINE](https://github.com/Vchitect/SEINE/) for video diffusion in [rendering](#rendering) 
