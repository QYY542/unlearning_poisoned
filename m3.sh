python main.py --epochs 30 --n_shadows 10 --shadow_id 0 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 1 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 2 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 3 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 4 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 5 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 6 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 7 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 8 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 
python main.py --epochs 30 --n_shadows 10 --shadow_id 9 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --target_sample 4 --repeat_num 200 --fixed_label 0 --use_original_label 

python plot.py --poison_type flipped_label --target_sample 4
python calculate.py --poison_type flipped_label --target_sample 4
