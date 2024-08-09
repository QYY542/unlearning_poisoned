#!/bin/bash
TARGET_SAMPLE=23104


# python plot.py --poison_type random_label --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type random_label --target_sample $TARGET_SAMPLE

# python plot.py --poison_type fixed_label --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type fixed_label --target_sample $TARGET_SAMPLE



# python plot.py --poison_type flipped_label --dataset cifar10 --model resnet18 --target_sample $TARGET_SAMPLE
python plot.py --poison_type flipped_label --dataset cifar100 --model resnet18 --target_sample $TARGET_SAMPLE
# python plot.py --poison_type flipped_label --dataset FashionMNIST --model resnet18 --target_sample $TARGET_SAMPLE
# python plot.py --poison_type flipped_label --dataset FashionMNIST --model vgg16 --target_sample $TARGET_SAMPLE




# python plot.py --poison_type flipped_label --dataset FashionMNIST --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type flipped_label --target_sample $TARGET_SAMPLE

# python plot.py --poison_type random_samples --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type random_samples --target_sample $TARGET_SAMPLE