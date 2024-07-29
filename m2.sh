#!/bin/bash

EPOCHS=30
REPEAT_NUM=200
TARGET_SAMPLE=0
MODEL=resnet18
POISON_TYPE=fixed_label
N_SHADOWS=30
PKEEP=0.5
SAVEDIR=exp/cifar10
FIXED_LABEL=5

for SHADOW_ID in {0..29}
do
    python main.py --epochs $EPOCHS --n_shadows $N_SHADOWS --shadow_id $SHADOW_ID --model $MODEL --pkeep $PKEEP --savedir $SAVEDIR --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE --repeat_num $REPEAT_NUM --fixed_label $FIXED_LABEL
done

python plot.py --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE
python calculate.py --poison_type $POISON_TYPE --target_sample $TARGET_SAMPLE
