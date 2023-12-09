#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pde-deq

noise_levels="0"
depths="1 2 4 8"
frame_no=3
seeds="0 1 2"

data_base_path='/path/to/PDE_datasets/'
model_base_path='/path/to/ckpts/'
wandb_prefix=prefix_for_wandb_logging

model=non-wt
for depth in $depths
    do
    for seed in $seeds 
        do 
        for nl in $noise_levels
            do 
            CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_ns.py  \
                                        model=$model \
                                        pde_type=ss_navier_stokes \
                                        noise_level=$nl  \
                                        epochs=500 \
                                        ntrain=4500 \
                                        ntest=500 \
                                        width=32 \
                                        lr=0.005 \
                                        seed=$seed\
                                        train=True \
                                        batch_size=32 \
                                        depth_per_block=$depth \
                                        add_mlp=True \
                                        logging_freq=100 \
                                        use_wandb=True \
                                        wandb_prefix=$wandb_prefix \
                                        frame_number=[$frame_no] \
                                        normalize=True \
                                        res=255 \
                                        sub=2 \
                                        add_noise_to_inputs=False \
                                        data_base_path=$data_base_path \
                                        model_base_path=$model_base_path \

            done
        done
    done 

noise_levels="0.001 0.004"
depths="1 2 4"
seeds="0 1 2"
model=non-wt
for depth in $depths
    do
    for seed in $seeds 
        do 
        for nl in $noise_levels
            do 
            CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_ns.py  \
                                        model=$model \
                                        pde_type=ss_navier_stokes \
                                        noise_level=$nl  \
                                        epochs=500 \
                                        ntrain=4500 \
                                        ntest=500 \
                                        width=32 \
                                        lr=0.005 \
                                        seed=$seed\
                                        train=True \
                                        batch_size=32 \
                                        depth_per_block=$depth \
                                        add_mlp=True \
                                        logging_freq=100 \
                                        use_wandb=True \
                                        wandb_prefix=$wandb_prefix \
                                        frame_number=[$frame_no] \
                                        normalize=True \
                                        res=255 \
                                        sub=2 \
                                        add_noise_to_inputs=False \
                                        data_base_path=$data_base_path \
                                        model_base_path=$model_base_path \

            done
        done
    done 

noise_levels="0.004"
depths="1 2 4"
seeds="0 1 2"
for depth in $depths
    do
    for seed in $seeds 
        do 
        for nl in $noise_levels
            do 
            CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_ns.py  \
                                        model=$model \
                                        pde_type=ss_navier_stokes \
                                        noise_level=$nl  \
                                        epochs=500 \
                                        ntrain=4500 \
                                        ntest=500 \
                                        width=32 \
                                        lr=0.005 \
                                        seed=$seed\
                                        train=True \
                                        batch_size=32 \
                                        depth_per_block=$depth \
                                        add_mlp=True \
                                        logging_freq=100 \
                                        use_wandb=True \
                                        wandb_prefix=$wandb_prefix \
                                        frame_number=[$frame_no] \
                                        normalize=True \
                                        res=255 \
                                        sub=2 \
                                        add_noise_to_inputs=True \
                                        data_base_path=$data_base_path \
                                        model_base_path=$model_base_path \

            done
        done
    done 
