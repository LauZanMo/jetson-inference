#!/bin/bash

set -e 

TRTEXEC=/usr/src/tensorrt/bin/trtexec
WORKSPACE=2048

$TRTEXEC --onnx=models/pose-densenet121-body.onnx --fp16 --workspace=$WORKSPACE --verbose | tee models/pose-densenet121-body.benchmark.txt
$TRTEXEC --onnx=models/pose-resnet18-body.onnx --fp16 --workspace=$WORKSPACE --verbose | tee models/pose-resnet18-body.benchmark.txt
$TRTEXEC --onnx=models/pose-resnet18-hand.onnx --fp16 --workspace=$WORKSPACE --verbose | tee models/pose-resnet18-hand.benchmark.txt