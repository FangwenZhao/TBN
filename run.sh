#!/usr/bin/env bash
GPU_ID=1
data_dir=./data/office31

loss=none
seed=1
# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config ./Config.yaml --seed $seed --transfer_loss $loss    --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee ./results/office31/test_0/seed_1/D2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config ./Config.yaml --seed $seed --transfer_loss $loss   --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee ./results/office31/test_0/seed_1/D2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config ./Config.yaml --seed $seed --transfer_loss $loss   --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee ./results/office31/test_0/seed_1/A2D.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config ./Config.yaml --seed $seed --transfer_loss $loss  --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee ./results/office31/test_0/seed_1/A2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config ./Config.yaml --seed $seed --transfer_loss $loss   --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee ./results/office31/test_0/seed_1/W2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config ./Config.yaml --seed $seed --transfer_loss $loss   --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee ./results/office31/test_0/seed_1/W2D.log
