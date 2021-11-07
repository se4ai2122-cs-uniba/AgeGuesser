#!/bin/bash

source ./venv/bin/activate
python3 train.py --img 320 --batch 32 --epochs 1 --data "dataset/yolo/data.yaml" --cfg ./models/yolov5s.yaml --weights "yolov5s.pt" --name results  --cache
python3 detect.py --weights "runs/train/results/weights/best.pt" --source "dataset/yolo/test/images"