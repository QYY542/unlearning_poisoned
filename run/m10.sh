#!/bin/bash

EPOCHS=30
REPEAT_NUM=900
NUM_PER_LABEL=8
MODEL=resnet18
DATASET=FashionMNIST
POISON_TYPE=flipped_label
N_SHADOWS=30
PKEEP=0.5
SAVEDIR=exp
FIXED_LABEL=0
USE_ORIGINAL_LABEL=--use_original_label

# TARGET_SAMPLES=(23103 2563)
TARGET_SAMPLES=(23103)

for TARGET_SAMPLE in "${TARGET_SAMPLES[@]}"; do
    for SHADOW_ID in $(seq 0 $((N_SHADOWS-1))); do
        python main.py --epochs $EPOCHS --n_shadows $N_SHADOWS --shadow_id $SHADOW_ID --model $MODEL --dataset $DATASET --pkeep $PKEEP --savedir $SAVEDIR --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE --repeat_num $REPEAT_NUM --fixed_label $FIXED_LABEL $USE_ORIGINAL_LABEL --num_per_label $NUM_PER_LABEL
    done
done


