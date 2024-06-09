#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_P2P_DISABLE=1
export OPENAI_LOGDIR=./result
python scripts/image_sample_realtime.py --model_path ./model_opt2sar/model100000.pt --batch_size 64 --num_samples 10000 --timestep_respacing 250 --diffusion_steps 2000 --noise_schedule linear --image_size 256