#!/bin/bash
# This script is for training an image diffusion model on GF3 dataset
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_P2P_DISABLE=1
export OPENAI_LOGDIR=./log
python scripts/image_train.py --data_dir_opt ../../Dataset/scaling/train/opt --data_dir_sar ../../Dataset/scaling/train/sar --lr 1e-4 --weight_decay 1e-3 --batch_size 12 --diffusion_steps 2000 --noise_schedule linear --image_size 256

