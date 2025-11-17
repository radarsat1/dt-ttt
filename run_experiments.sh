#!/bin/bash

uv run python train.py --exp_name offline_rtg \
   --num_epochs 200 --no-online_data_generation --no-use_advantage

uv run python train.py --exp_name online_rtg \
   --num_epochs 200 --online_data_generation --no-use_advantage

uv run python train.py --exp_name offline_advantage \
   --num_epochs 200 --no-online_data_generation --use_advantage

uv run python train.py --exp_name online_advantage \
   --num_epochs 200 --use_advantage --online_data_generation
