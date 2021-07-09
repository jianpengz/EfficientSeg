#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM train.py \
--train_list='list/MyoPS2020/train.txt' \
--val_list='list/MyoPS2020/val.txt' \
--snapshot_dir='snapshots/MyoPS2020/EfficientSeg-B1/' \
--input_size='288,288' \
--compound_coef=1 \
--batch_size=64 \
--FP16=True \
--num_gpus=4 \
--num_epochs=500 \
--start_epoch=0 \
--save_pred_every=10 \
--patience=3 \
--learning_rate=1e-4 \
--num_classes=6 \
--num_workers=16 \
--random_mirror=True \
--random_scale=True
