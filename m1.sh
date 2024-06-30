# python main.py --epochs 10 --n_shadows 2 --shadow_id 0 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --num_to_poison 5 --repeat_num 10
# python main.py --epochs 10 --n_shadows 2 --shadow_id 0 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type fixed_label --num_to_poison 5 --fixed_label 0 --repeat_num 10
# python main.py --epochs 10 --n_shadows 2 --shadow_id 0 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type flipped_label --num_to_poison 5 --fixed_label 0 --num_to_flip 10 --repeat_num 10

python main.py --epochs 10 --n_shadows 2 --shadow_id 0 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 5 --repeat_num 10
python main.py --epochs 10 --n_shadows 2 --shadow_id 1 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 5 --repeat_num 10 
# python3 inference.py --savedir exp/cifar10
# python3 score.py --savedir exp/cifar10
# python3 plot.py --savedir exp/cifar10

