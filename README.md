# Deep Equilibrium Based Neural Operators for Steady State PDEs [NeurIPS 2023]

## Getting Started
1. Create conda environment and install all packages from `requirements.txt`
```
conda create --name <environment_name> --file requirements.txt
conda activate <environment_name>
```

2. Download datasets for Darcy Flow released by Li et al. 2021 [[Follow this google drive link](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)]
3. Download datasets for Navier-Stokes from this [link](https://drive.google.com/drive/folders/1790NVbM6IPaQNKQNQQG93LcF3YxJCTOk?usp=sharing).

## Training Models

### Training and evaluating models on Darcy Flow data

All bash scripts used for training and evaluating models on Darcy flow are located at `darcy_flow/scripts`.

First, update the following variables in `darcy_flow/configs/config.yml` and in the script, if necessary:
1. `data_base_path` : path to dataset 
2. `model_base_path`: path to folder to store checkpoints

Command to train FNO-DEQ would be `bash scripts/run_deq.sh`. 

Similar commands can be used to train other models. Refer to the following table choose an appropriate script:

| Model  | File  |
|---|---|
| FNO    | [run_non_wt_no_inj.sh](runners/scripts/darcy_flow/run_non_wt_no_inj.sh)|
| FNO++  | [run_non_wt.sh](runners/scripts/darcy_flow/run_non_wt.sh) |
| FNO-WT | [run_wt.sh](runners/scripts/darcy_flow/run_wt.sh) |
| FNO-DEQ | [run_deq.sh](runners/scripts/darcy_flow/run_deq.sh) |

To evaluate with a pretrained checkpoint, set `train=False` in the script, and set `ckpt` to the pretrained checkpoint in config.

Note: wandb logging is disabled by default. You can enable it by setting `use_wandb=True` in the scripts. 

### Training models and evaluating models on Navier-Stokes data

All bash scripts used for training and evaluating models on Darcy flow are located at `steady_state_navier_stokes/scripts`.

First, update the following variables in `steady_state_navier_stokes/configs/config.yml` and in the script, if necessary:
1. `data_base_path` : path to dataset 
2. `model_base_path`: path to folder to store checkpoints

Command to train FNO-DEQ would be `bash scripts/run_deq.sh`. 

Please use the following scripts to train models:

| Model  | File  |
|---|---|
| FNO    | [run_non_wt_no_inj.sh](runners/scripts/ss_navier_stokes/run_non_wt_no_inj.sh)|
| FNO++  | [run_non_wt.sh](runners/scripts/ss_navier_stokes/run_non_wt.sh) |
| FNO-WT | [run_wt.sh](runners/scripts/ss_navier_stokes/run_wt.sh) |
| FNO-DEQ | [run_deq.sh](runners/scripts/ss_navier_stokes/run_deq.sh) |

To evaluate with a pretrained checkpoint, set `train=False` in the script, and set `ckpt` to the pretrained checkpoint in config.
