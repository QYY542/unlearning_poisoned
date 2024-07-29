#!/bin/bash

EPOCHS=30
REPEAT_NUM=2000
TARGET_SAMPLE=11
MODEL=resnet18
POISON_TYPE=random_samples
N_SHADOWS=20
PKEEP=0.5
SAVEDIR=exp/cifar10
FIXED_LABEL=0
USE_ORIGINAL_LABEL=--use_original_label

for SHADOW_ID in {0..19}
do
    python main.py --epochs $EPOCHS --n_shadows $N_SHADOWS --shadow_id $SHADOW_ID --model $MODEL --pkeep $PKEEP --savedir $SAVEDIR --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE --repeat_num $REPEAT_NUM --fixed_label $FIXED_LABEL $USE_ORIGINAL_LABEL
done

python plot.py --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE
python calculate.py --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE
