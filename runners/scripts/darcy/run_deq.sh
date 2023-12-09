#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pde-deq

pde_type=darcy_flow

model=deq
tau_vals="0.5" 
pg_steps=1
res=421
sub=5
seeds="2"
nl=0

lr_scheduler=cosine
add_mlp=True

data_base_path='/path/to/PDE_datasets/'

# Train FNO-DEQ on clean data
add_noise_to_inputs=True
noisy_data=False
for seed in $seeds
    do
    for tau in $tau_vals
        do 
        python fourier_2d_runner_darcy.py  \
                                model=$model \
                                noise_level=$nl \
                                add_noise_to_inputs=$add_noise_to_inputs  \
                                epochs=500 \
                                ntrain=1024 \
                                ntest=500 \
                                seed=$seed \
                                train=False \
                                data_base_path=$data_base_path \
                                dataset=piececonst_r421_N1024_smooth1.mat\
                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                batch_size=1 \
                                lr_scheduler=$lr_scheduler \
                                depth_per_block=1 \
                                pde_type=$pde_type \
                                add_mlp=$add_mlp \
                                logging_freq=100 \
                                use_pg=True \
                                tau=$tau\
                                pg_steps=$pg_steps\
                                solver_steps=2\
                                pretrain_iter_steps=8\
                                noisy_data=$noisy_data\
                                use_wandb=False \
                                res=$res\
                                sub=$sub\
                                eps=1e-6\
                                model_save_folder_path=models_$pde_type/$model/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/tau_$tau\_pg_steps$pg_steps \
                                ckpt=lr_0.001_ep_500_ntrain_1024_2023-06-17_22-08-07

        done
    done

### Train FNO-DEQ on noisy input data
seeds="0 1 2"
add_noise_to_inputs=True
noisy_data=True
nl=0.001

for seed in $seeds
    do
    for tau in $tau_vals
        do 
        python fourier_2d_runner.py  \
                                model=$model \
                                noise_level=$nl \
                                add_noise_to_inputs=$add_noise_to_inputs  \
                                epochs=500 \
                                ntrain=1024 \
                                ntest=500 \
                                seed=$seed \
                                train=True \
                                dataset=piececonst_r421_N1024_smooth1.mat\
                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                batch_size=32 \
                                lr_scheduler=$lr_scheduler \
                                depth_per_block=1 \
                                pde_type=$pde_type \
                                add_mlp=$add_mlp \
                                logging_freq=100 \
                                use_pg=True \
                                tau=$tau\
                                pg_steps=$pg_steps\
                                solver_steps=32\
                                pretrain_iter_steps=8\
                                noisy_data=$noisy_data\
                                use_wandb=False \
                                res=$res\
                                sub=$sub\
                                eps=1e-6\
                                model_save_folder_path=models_$pde_type/$model/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/tau_$tau\_pg_steps$pg_steps
        done
    done

### Train FNO-DEQ for noisy targets
seeds="0 1 2"
add_noise_to_inputs=False
noisy_data=True
nl=0.001
for seed in $seeds
    do
    for tau in $tau_vals
        do 
        python fourier_2d_runner.py  \
                                model=$model \
                                noise_level=$nl \
                                add_noise_to_inputs=$add_noise_to_inputs  \
                                epochs=500 \
                                ntrain=1024 \
                                ntest=500 \
                                seed=$seed \
                                train=True \
                                dataset=piececonst_r421_N1024_smooth1.mat\
                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                batch_size=32 \
                                lr_scheduler=$lr_scheduler \
                                depth_per_block=1 \
                                pde_type=$pde_type \
                                add_mlp=$add_mlp \
                                logging_freq=100 \
                                use_pg=True \
                                tau=$tau\
                                pg_steps=$pg_steps\
                                solver_steps=32\
                                pretrain_iter_steps=8\
                                noisy_data=$noisy_data\
                                use_wandb=False \
                                res=$res\
                                sub=$sub\
                                eps=1e-6\
                                model_save_folder_path=models_$pde_type/$model/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/tau_$tau\_pg_steps$pg_steps
        done
    done