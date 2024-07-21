#!/bin/bash

EPOCHS=20
REPEAT_NUM=10
TARGET_SAMPLE=4
MODEL=resnet18
POISON_TYPE=fixed_label
N_SHADOWS=10
PKEEP=0.5
SAVEDIR=exp/cifar10
FIXED_LABEL=0
USE_ORIGINAL_LABEL=--use_original_label

# for SHADOW_ID in {0..9}
# do
#     python main.py --epochs $EPOCHS --n_shadows $N_SHADOWS --shadow_id $SHADOW_ID --model $MODEL --pkeep $PKEEP --savedir $SAVEDIR --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE --repeat_num $REPEAT_NUM --fixed_label $FIXED_LABEL $USE_ORIGINAL_LABEL
# done

python plot.py --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE
