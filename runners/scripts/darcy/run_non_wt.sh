#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pde-deq

pde_type=darcy_flow
model=non-wt
add_noise_to_inputs=False
lr_scheduler="cosine"
res=421
sub=5
seeds="0 1 2"
add_mlp=True
depths="1 2 4"
noise_levels="0"

##### Train FNO++ models on clean data
for nl in $noise_levels
        do 
        for depth in $depths
                do
                for seed in $seeds
                        do 
                        CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_darcy.py  \
                                                model=$model \
                                                noise_level=$nl  \
                                                add_noise_to_inputs=$add_noise_to_inputs \
                                                epochs=500 \
                                                ntrain=1024 \
                                                ntest=500 \
                                                seed=$seed \
                                                train=True \
                                                lr_scheduler=$lr_scheduler \
                                                dataset=piececonst_r421_N1024_smooth1.mat\
                                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                                batch_size=32 \
                                                depth_per_block=$depth \
                                                pde_type=darcy_flow \
                                                add_mlp=True \
                                                logging_freq=100 \
                                                noisy_data=False \
                                                use_wandb=False \
                                                res=$res \
                                                sub=$sub \
                                                model_save_folder_path=models_$pde_type/$model/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/ \

                        done
                done
        done

##### Train FNO++ models on noisy input data
add_noise_to_inputs=True
noise_levels="0.001"
for nl in $noise_levels
        do 
        for depth in $depths
                do
                for seed in $seeds
                        do 
                        CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_darcy.py  \
                                                model=$model \
                                                noise_level=$nl  \
                                                add_noise_to_inputs=$add_noise_to_inputs  \
                                                epochs=500 \
                                                ntrain=1024 \
                                                ntest=500 \
                                                seed=$seed \
                                                train=True \
                                                lr_scheduler=$lr_scheduler \
                                                dataset=piececonst_r421_N1024_smooth1.mat\
                                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                                batch_size=32 \
                                                depth_per_block=$depth \
                                                pde_type=darcy_flow \
                                                add_mlp=$add_mlp \
                                                logging_freq=100 \
                                                noisy_data=True \
                                                use_wandb=False \
                                                res=$res \
                                                sub=$sub \
                                                model_save_folder_path=models_$pde_type/$model/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/
                        done
                done
        done

##### Train FNO++ models on noisy observation data
add_noise_to_inputs=False
noise_levels="0.001"
for nl in $noise_levels
        do 
        for depth in $depths
                do
                for seed in $seeds
                        do 
                        CUDA_VISIBLE_DEVICES=0 python fourier_2d_runner_darcy.py  \
                                                model=$model \
                                                noise_level=$nl  \
                                                add_noise_to_inputs=$add_noise_to_inputs  \
                                                epochs=500 \
                                                ntrain=1024 \
                                                ntest=500 \
                                                seed=$seed \
                                                train=True \
                                                lr_scheduler=$lr_scheduler \
                                                dataset=piececonst_r421_N1024_smooth1.mat\
                                                test_dataset=piececonst_r421_N1024_smooth2.mat\
                                                batch_size=32 \
                                                depth_per_block=$depth \
                                                pde_type=darcy_flow \
                                                add_mlp=$add_mlp \
                                                logging_freq=100 \
                                                noisy_data=True \
                                                use_wandb=False \
                                                res=$res \
                                                sub=$sub \
                                                model_save_folder_path=models_$pde_type/$model/add_noise_to_input_$add_noise_to_inputs/noise_level_$nl\_res$res\_sub$sub/
                        done
                done
        done
