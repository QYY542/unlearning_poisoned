python main.py --epochs 20 --n_shadows 2 --shadow_id 0 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200
python main.py --epochs 20 --n_shadows 2 --shadow_id 1 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200 
# python main.py --epochs 20 --n_shadows 10 --shadow_id 2 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200
# python main.py --epochs 20 --n_shadows 10 --shadow_id 3 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200 
# python main.py --epochs 20 --n_shadows 10 --shadow_id 4 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200
# python main.py --epochs 20 --n_shadows 10 --shadow_id 5 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200 
# python main.py --epochs 20 --n_shadows 10 --shadow_id 6 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200
# python main.py --epochs 20 --n_shadows 10 --shadow_id 7 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200 
# python main.py --epochs 20 --n_shadows 10 --shadow_id 8 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200
# python main.py --epochs 20 --n_shadows 10 --shadow_id 9 --model resnet18 --pkeep 0.5 --savedir exp/cifar10 --poison_type random_uniform --target_sample 0 --repeat_num 200 

python plot.py --poison_type random_uniform --target_sample 0
python calculate.py --poison_type random_uniform --target_sample 0

