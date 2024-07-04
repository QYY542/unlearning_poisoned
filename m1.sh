python main.py --epochs 50 --n_shadows 8 --shadow_id 0 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100
python main.py --epochs 50 --n_shadows 8 --shadow_id 1 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100 
python main.py --epochs 50 --n_shadows 8 --shadow_id 2 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100
python main.py --epochs 50 --n_shadows 8 --shadow_id 3 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100 
python main.py --epochs 50 --n_shadows 8 --shadow_id 4 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100
python main.py --epochs 50 --n_shadows 8 --shadow_id 5 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100 
python main.py --epochs 50 --n_shadows 8 --shadow_id 6 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100
python main.py --epochs 50 --n_shadows 8 --shadow_id 7 --model resnet18 --pkeep 0.5 --savedir exp/cifar10/random_uniform --poison_type random_uniform --num_to_poison 2 --repeat_num 100 

python plot.py --savedir exp/cifar10/random_uniform --sample_index 0
python plot.py --savedir exp/cifar10/random_uniform --sample_index 1
python plot.py --savedir exp/cifar10/random_uniform --sample_index 2
python plot.py --savedir exp/cifar10/random_uniform --sample_index 3
python plot.py --savedir exp/cifar10/random_uniform --sample_index 4

