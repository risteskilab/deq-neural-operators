#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pde-deq

pde_type=ss_navier_stokes
add_noise_to_inputs=False
model=wt
res=255
sub=2
seeds="0 1 2"
noise_levels="0"
depths="1"
frame_no=3
# data_base_path=/path/to/dataset
# model_base_path=/path/to/model/folder/

data_base_path='/scratch/apokle/full_kolmogorov_wavenum_1_mv_5'
model_base_path='/project_data/datasets/apokle_exp/deq-pde/FNO_ckpts/full_kolmogorov_wavenum_1_mv_5'
wandb_prefix=fno_orig_visc_0.001_global_norm_std

# data_base_path='/scratch/apokle/full_type_noise__forcing_kolmogorov_visc_0.01_mv_5_peak_wave_num_1'
# model_base_path='/project_data/datasets/apokle_exp/deq-pde/FNO_ckpts/full_type_noise__forcing_kolmogorov_visc_0.01_mv_5_peak_wave_num_1'
# wandb_prefix=fno_orig_visc_0.01_global_norm_std

# for depth in $depths
#     do
#     for seed in $seeds 
#         do 
#         for nl in $noise_levels
#             do
#             CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_ns.py  \
#                             model=$model \
#                             noise_level=$nl \
#                             epochs=500 \
#                             ntrain=4500 \
#                             ntest=500 \
#                             train=True \
#                             seed=$seed \
#                             width=32 \
#                             lr=5e-3 \
#                             batch_size=32 \
#                             depth_per_block=1 \
#                             pde_type=$pde_type \
#                             add_mlp=True \
#                             logging_freq=100 \
#                             solver_steps=6\
#                             pretrain_iter_steps=6 \
#                             noisy_data=True \
#                             use_wandb=True \
#                             wandb_prefix=$wandb_prefix \
#                             frame_number=[$frame_no] \
#                             res=$res \
#                             sub=$sub \
#                             normalize=True \
#                             add_noise_to_inputs=False \
#                             data_base_path=$data_base_path \
#                             model_base_path=$model_base_path \
#                             model_save_folder_path=models_$pde_type/$model\_$norm_type/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/tau_$tau\_pg_steps$pg_steps
#             done
#         done 
#     done

seeds="1 2"
add_noise_to_inputs=False
noise_levels="0.004"
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
                            depth_per_block=1 \
                            pde_type=$pde_type \
                            add_mlp=True \
                            logging_freq=100 \
                            use_pg=True \
                            tau=$tau\
                            pg_steps=$pg_steps\
                            solver_steps=16\
                            pretrain_iter_steps=8\
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

seeds="0 1 2"
add_noise_to_inputs=True
noise_levels="0.004"
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
                            depth_per_block=1 \
                            pde_type=$pde_type \
                            add_mlp=True \
                            logging_freq=100 \
                            use_pg=True \
                            tau=$tau\
                            pg_steps=$pg_steps\
                            solver_steps=12\
                            pretrain_iter_steps=8\
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

