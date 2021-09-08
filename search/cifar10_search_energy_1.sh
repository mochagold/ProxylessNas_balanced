#python imagenet_arch_search.py --path './result/' --width_stages '32,16,32,64,160,320' --n_cell_stages '2,3,4,3,3,1' --stride_stages '1,2,2,1,2,1' --gpu 1 --dataset cifar10 --target_hardware flops --grad_reg_loss_type 'add#linear'
python imagenet_arch_search.py --path './result/cifar10_mobile_PE12x14/' --yaml_path '/data/share/yaml/EyerisPE12x14.yaml' --width_stages '32,64,128' --n_cell_stages '3,3,3' --stride_stages '1,2,2' --gpu 0 --dataset cifar10 --target_hardware mobile --grad_reg_loss_type 'add#linear' --grad_reg_loss_lambda 0  --grad_binary_mode 'full_v2' 2>&1 | tee cifar10_mobile_PE12x14.log
