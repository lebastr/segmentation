#!/usr/bin/env bash
python src/train.py --input_size=260 --output_size=52 --model_dir=./models/ --dataset_dir=/home/lebedev/datasets/road_segmentation_dataset/dataset1 $*
