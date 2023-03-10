#!/bin/bash

# pretraining + hold (aka sequential).
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-09-30_11-54-42_pretrain_1_s0_train \
  --name pretrain_1 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-09-30_11-54-44_pretrain_2_s0_train \
  --name pretrain_2 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-05_01-06-08_pretrain_3_s0_train \
  --name pretrain_3 \
  BENCHMARK.SAVE_RESULT True

# pretraining + without-hold (aka simultaneous).
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-09-30_11-54-42_pretrain_1_s0_train \
  --without-hold \
  --name pretrain_1 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-09-30_11-54-44_pretrain_2_s0_train \
  --without-hold \
  --name pretrain_2 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-05_01-06-08_pretrain_3_s0_train \
  --without-hold \
  --name pretrain_3 \
  BENCHMARK.SAVE_RESULT True

# finetuning + hold (aka sequential).
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-14_03-01-32_finetune_1_s0_train \
  --name finetune_1 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-16_12-51-46_finetune_4_s0_train \
  --name finetune_4 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-16_08-48-30_finetune_5_s0_train \
  --name finetune_5 \
  BENCHMARK.SAVE_RESULT True

# finetuning + without-hold (aka simultaneous).
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-14_03-01-32_finetune_1_s0_train \
  --without-hold \
  --name finetune_1 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-16_12-51-46_finetune_4_s0_train \
  --without-hold \
  --name finetune_4 \
  BENCHMARK.SAVE_RESULT True
GADDPG_DIR=GA-DDPG CUDA_VISIBLE_DEVICES=0 python examples/test.py \
  --model-dir output/cvpr2023_models/2022-10-16_08-48-30_finetune_5_s0_train \
  --without-hold \
  --name finetune_5 \
  BENCHMARK.SAVE_RESULT True
