#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pde-deq


seeds="0 1 2"
pde_type=darcy_flow
add_noise_to_inputs=False
nl=0
add_mlp=True
lr_scheduler=cosine
noisy_data=False
### Train FNO-WT on clean data
for seed in $seeds
        do 
        CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_darcy.py  \
                                model=wt \
                                noise_level=$nl  \
                                add_noise_to_inputs=$add_noise_to_inputs  \
                                epochs=500 \
                                ntrain=1024 \
                                lr_scheduler=$lr_scheduler \
                                ntest=500 \
                                seed=$seed \
                                train=True \
                                dataset=piececonst_r421_N1024_smooth1.mat\
                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                batch_size=32 \
                                depth_per_block=1 \
                                pde_type=darcy_flow \
                                add_mlp=$add_mlp \
                                solver_steps=12 \
                                logging_freq=100 \
                                noisy_data=$noisy_data \
                                use_wandb=False \
                                res=421 \
                                sub=5
        done

### Train FNO-WT for noisy targets
seeds="0 1 2"
add_noise_to_inputs=False
noisy_data=True
nl=0.001
for seed in $seeds
        do 
        CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_darcy.py  \
                                model=wt \
                                noise_level=$nl  \
                                add_noise_to_inputs=$add_noise_to_inputs  \
                                epochs=500 \
                                ntrain=1024 \
                                ntest=500 \
                                seed=$seed \
                                train=True \
                                dataset=piececonst_r421_N1024_smooth1.mat\
                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                batch_size=32 \
                                depth_per_block=1 \
                                pde_type=darcy_flow \
                                add_mlp=True \
                                solver_steps=12 \
                                logging_freq=100 \
                                noisy_data=False \
                                use_wandb=False \
                                res=421 \
                                sub=5
        done

### Train FNO-WT for noisy inputs
seeds="0 1 2"
add_noise_to_inputs=True
noisy_data=True
nl=0.001
for seed in $seeds
        do 
        CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_darcy.py  \
                                model=wt \
                                noise_level=$nl  \
                                add_noise_to_inputs=$add_noise_to_inputs  \
                                epochs=500 \
                                ntrain=1024 \
                                ntest=500 \
                                seed=$seed \
                                train=True \
                                dataset=piececonst_r421_N1024_smooth1.mat\
                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                batch_size=32 \
                                depth_per_block=1 \
                                pde_type=darcy_flow \
                                add_mlp=True \
                                solver_steps=12 \
                                logging_freq=100 \
                                noisy_data=False \
                                use_wandb=False \
                                res=421 \
                                sub=5
        done