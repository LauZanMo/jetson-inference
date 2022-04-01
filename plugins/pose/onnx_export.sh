#!/bin/bash

set -e 

python3 onnx_export.py --arch densenet121 --topology tasks/human_pose/human_pose.json --input models/densenet121_baseline_att_256x256_B_epoch_160.pth --output models/pose-densenet121-body.onnx

python3 onnx_export.py --arch resnet18 --topology tasks/human_pose/human_pose.json --input models/resnet18_baseline_att_224x224_A_epoch_249.pth --output models/pose-resnet18-body.onnx

python3 onnx_export.py --arch resnet18 --topology models/hand_pose.json --input models/hand_pose_resnet18_att_244_244.pth --output models/pose-resnet18-hand.onnx