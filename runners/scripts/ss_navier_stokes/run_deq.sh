#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pde-deq

pde_type=ss_navier_stokes
add_noise_to_inputs=False
model=deq
tau=0.8
pg_steps=3
res=255
sub=2
seeds="0 1 2"
noise_levels="0 0.001 0.004"
depths="1"
frame_no=3

data_base_path='/path/to/PDE_datasets/'
model_base_path='/path/to/ckpts/'
wandb_prefix=prefix_for_wandb_logging

for depth in $depths
    do
    for seed in $seeds 
        do 
        for nl in $noise_levels
            do
            CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_ns.py  \
                            model=$model \
                            noise_level=$nl \
                            epochs=500 \
                            ntrain=4500 \
                            ntest=500 \
                            train=True \
                            seed=$seed \
                            width=32 \
                            lr=5e-3 \
                            batch_size=32 \
                            depth_per_block=$depth \
                            pde_type=$pde_type \
                            add_mlp=True \
                            logging_freq=100 \
                            use_pg=True \
                            tau=$tau \
                            pg_steps=$pg_steps \
                            solver=anderson \
                            solver_steps=24\
                            eps=1e-6\
                            pretrain_iter_steps=12\
                            noisy_data=True \
                            use_wandb=True \
                            wandb_prefix=$wandb_prefix \
                            frame_number=[$frame_no] \
                            res=$res \
                            sub=$sub \
                            normalize=True \
                            add_noise_to_inputs=False \
                            data_base_path=$data_base_path \
                            model_base_path=$model_base_path \
                            model_save_folder_path=models_$pde_type/$model\_$norm_type/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/tau_$tau\_pg_steps$pg_steps 
            done
        done 
    done

add_noise_to_inputs=True  ### Set this to True or False 
noise_levels="0.001 0.004"
seeds="0 1 2"
for depth in $depths
    do
    for seed in $seeds 
        do 
        for nl in $noise_levels
            do
            CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_ns.py  \
                            model=$model \
                            noise_level=$nl \
                            epochs=500 \
                            ntrain=4500 \
                            ntest=500 \
                            train=True \
                            seed=$seed \
                            width=32 \
                            lr=5e-3 \
                            batch_size=32 \
                            depth_per_block=$depth \
                            pde_type=$pde_type \
                            add_mlp=True \
                            logging_freq=100 \
                            use_pg=True \
                            tau=$tau\
                            pg_steps=$pg_steps\
                            solver_steps=16\
                            pretrain_iter_steps=12\
                            noisy_data=True \
                            use_wandb=True \
                            wandb_prefix=$wandb_prefix \
                            frame_number=[$frame_no] \
                            res=$res\
                            sub=$sub\
                            normalize=True \
                            add_noise_to_inputs=$add_noise_to_inputs \
                            data_base_path=$data_base_path \
                            model_base_path=$model_base_path \
                            model_save_folder_path=models_$pde_type/$model\_$norm_type/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/tau_$tau\_pg_steps$pg_steps
            done
        done 
    done
