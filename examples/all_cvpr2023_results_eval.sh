#!/bin/bash

# pretraining + hold (aka sequential). ("w/o fintuning" in Tab. 2)
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-10_10-59-15_handover-sim2real-hold_pretrain_1_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-10_11-29-32_handover-sim2real-hold_pretrain_2_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-10_12-00-00_handover-sim2real-hold_pretrain_3_s0_test

# pretraining + without-hold (aka simultaneous). ("w/o finetuning simult." in Tab. 2)
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-10_13-04-12_handover-sim2real-wo-hold_pretrain_1_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-10_13-30-31_handover-sim2real-wo-hold_pretrain_2_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-10_13-56-35_handover-sim2real-wo-hold_pretrain_3_s0_test

# finetuning + hold (aka sequential). (Tab. 1, and "Ours" in Tab. 2)
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-07_03-50-02_handover-sim2real-hold_finetune_1_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-07_05-26-26_handover-sim2real-hold_finetune_4_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-07_09-07-20_handover-sim2real-hold_finetune_5_s0_test

# finetuning + without-hold (aka simultaneous). (Tab. 1 and "Ours simult." in Tab. 2)
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-09_16-02-29_handover-sim2real-wo-hold_finetune_1_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-09_17-27-28_handover-sim2real-wo-hold_finetune_4_s0_test
python handover-sim/examples/evaluate_benchmark.py \
  --res_dir results/cvpr2023_results/2022-11-09_17-55-43_handover-sim2real-wo-hold_finetune_5_s0_test
