#!/bin/bash 

NAME=$1

cd simulation
python animate.py --config ../data/${NAME}/sim.yaml --save_root ../outputs

cd ../relight
python relight.py --perception_input ../data/${NAME} --previous_output  ../outputs/${NAME}

cd ../diffusion
python video_diffusion.py --perception_input ../data/${NAME} --previous_output ../outputs/${NAME}