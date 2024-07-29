#!/bin/bash
TARGET_SAMPLE=11


# python plot.py --poison_type random_label --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type random_label --target_sample $TARGET_SAMPLE

# python plot.py --poison_type fixed_label --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type fixed_label --target_sample $TARGET_SAMPLE

# python plot.py --poison_type flipped_label --target_sample $TARGET_SAMPLE
# python calculate.py --poison_type flipped_label --target_sample $TARGET_SAMPLE

python plot.py --poison_type random_samples --target_sample $TARGET_SAMPLE
python calculate.py --poison_type random_samples --target_sample $TARGET_SAMPLE