#
# converts a saved PyTorch model to ONNX format
# 
import os
import json
import argparse
import pprint

import torch
import trt_pose.models


# parse command line
parser = argparse.ArgumentParser()

parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'densenet121'], help="model architecture")
parser.add_argument('--input', type=str, default='', required=True, help="path to input PyTorch .pth model")
parser.add_argument('--output', type=str, default='', required=True, help="desired path of converted ONNX model")
parser.add_argument('--topology', type=str, default='', required=True, help="path to skeleton topology .json file")

args = parser.parse_args() 
print(args)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# load the topology
print('loading topology from ' + args.topology)

with open(args.topology, 'r') as f:
    topology = json.load(f)

pprint.pprint(topology)

num_parts = len(topology['keypoints'])
num_links = len(topology['skeleton'])

print('num parts', num_parts)
print('num links', num_links)

# create the model
if args.arch == 'resnet18':
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links, pretrained=False)   #.cuda().eval()
    resolution = 224
elif args.arch == 'densenet121':
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links, pretrained=False)    #.cuda().eval()
    resolution = 256
else:
    raise ValueError(f'invalid --arch specified ({args.arch})')
    
print('created model ' + args.arch)

model.to(device)
model.eval()

# load the model checkpoint
print('loading checkpoint:  ' + args.input)
model.load_state_dict(torch.load(args.input))
model.eval()
print(model)

# create example image data
print('input size:  {:d}x{:d}'.format(resolution, resolution))
input = torch.zeros((1, 3, resolution, resolution)).to(device)

# export the model
input_names = ["input"]
output_names = ["cmap", "paf"]

print('exporting model to ONNX...')
torch.onnx.export(model, input, args.output, verbose=True, do_constant_folding=True, input_names=input_names, output_names=output_names)    #opset_version=10, 
print('model exported to:  {:s}'.format(args.output))


