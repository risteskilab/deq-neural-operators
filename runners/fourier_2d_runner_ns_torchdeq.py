import sys
import os
import seaborn as sns 

sys.path.append("lib/")
sys.path.append("../")

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import random

from utils.utilities3 import *
from optimizer.adam import Adam
from timeit import default_timer

from models.fourier_2d_deq_alt import FNO2dDEQ

import wandb
import matplotlib.pyplot as plt 

def get_model(args):
    if args.model == 'deq' or args.model == 'wt':
        model = FNO2dDEQ(
            modes1=args.modes,
            modes2=args.modes, 
            width=args.width,
            args=args,
            block_depth=args.depth_per_block,
            add_mlp=args.add_mlp,
            in_channels=args.in_channels, 
            out_channels=args.out_channels, 
            normalize=args.normalize).cuda()
    else:
        raise ValueError(f"Unknown model {args.model}")
    return model


#### Non autoregressive training and testing
#### Used for Darcy flow 2D
def train(args, train_loader, val_loader, y_normalizer, model, t_train=101, initial_step=10):
    print(count_params(model))

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iterations = args.epochs*(args.ntrain//args.batch_size)
    if args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations)
    elif args.lr_schedule == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1, total_iters=0)
    else:
        ValueError("Incorrect option for Lr Schedule")

    myloss = LpLoss(size_average=False)
    if y_normalizer is not None:
        y_normalizer.cuda()
    train_step = 0
    ### set to -1 for DEQ mode
    if 'deq' in args.model:
        iters = -1
    else:
        iters = args.solver_steps

    model_folder_path = os.path.join(
            args.model_base_path,
            args.model_save_folder_path)
    os.makedirs(model_folder_path, exist_ok=True)
    cm = sns.color_palette("icefire", as_cmap=True)

    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0

        for data in train_loader:
            x, y = data[0].cuda(), data[1].cuda()
            grid = data[2].cuda() if len(data) == 3 else None
            batch_size, s = x.shape[0], x.shape[1]

            if len(x.shape) < 4:
                x = x.unsqueeze(-1)

            optimizer.zero_grad()

            out = model(x, grid=grid,
                        train_step=train_step, 
                        iters=iters, 
                        eps=args.eps)

            if y_normalizer is not None:
                out = y_normalizer.decode(out.squeeze())
                y = y_normalizer.decode(y)
            
            train_step += 1

            l2 = myloss(out.contiguous().view(batch_size, -1), 
                        y.contiguous().view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_l2 += l2.item()
            if args.lr_schedule == 'cosine':
                scheduler.step()

        if args.lr_schedule == 'step':
            scheduler.step()
       
        model.eval()
        test_l2 = 0.0
        # No noise on validation set!!!
        with torch.no_grad():
            for data in val_loader:
                xx, yy  = data[0].cuda(), data[1].cuda()
                grid = data[2].cuda() if len(data) == 3 else None
                batch_size = xx.shape[0]
                s = xx.shape[1]
                if len(xx.shape) < 4:
                    xx = xx.unsqueeze(-1)

                # out = model(xx, grid=grid, iters=iters).reshape(batch_size, s, s)
                out = model(xx, grid=grid, iters=iters)
                if y_normalizer is not None:
                    out = y_normalizer.decode(out.squeeze())

                test_l2 += myloss(out.contiguous().view(batch_size, -1), 
                                  yy.contiguous().view(batch_size, -1)).item()
                
        train_l2 /= args.ntrain
        test_l2 /= args.ntest

        t2 = default_timer()
        print(ep, t2-t1, train_l2, test_l2)

        if args.use_wandb:
            log_dict = {
                "train l2": train_l2,
                "val l2": test_l2,
                "lr_val": scheduler.get_last_lr()[0]
                }
            wandb.log(log_dict)

        if ep % args.logging_freq == 0:
            ckpt_path = f"{model_folder_path}/checkpoint_{ep}.pth"
            torch.save({'model': model.state_dict()}, ckpt_path)

    model_save_path = f'{model_folder_path}/{args.ckpt}.pth'
    print(f'will save model to {model_save_path}')
    torch.save({'model': model.state_dict()}, model_save_path)

def test(args, test_loader, y_normalizer, model):
    print(count_params(model))

    myloss = LpLoss(size_average=False)
    if y_normalizer is not None:
        y_normalizer.cuda()
    ### set to -1 for DEQ mode
    if 'deq' in args.model:
        iters = -1
    else:
        iters = args.solver_steps

    model_folder_path = os.path.join(
            args.model_base_path,
            args.model_save_folder_path)
    model_save_path = f'{model_folder_path}/{args.ckpt}.pth'
    saved_model = torch.load(f'{model_save_path}')
    model.load_state_dict(saved_model['model'])

    model.eval()

    # all_abs_trace = 0
    # all_rel_trace = 0
    test_l2 = 0
    # Don't add any noise to observations at test time too
    print(f"Testing on {args.ntest} clean samples")
    with torch.no_grad():
        for data in test_loader:
            xx, yy = data[0].cuda(), data[1].cuda()
            grid = data[2].cuda() if len(data) == 3 else None

            batch_size = xx.shape[0]
            s = xx.shape[1]

            if len(xx.shape) < 4:
                xx = xx.unsqueeze(-1)

            out = model(xx, 
                        grid=grid, 
                        iters=iters, 
                        eps=args.eps, 
                        wandb_log=False)

            if y_normalizer is not None:
                out = y_normalizer.decode(out)
            test_l2 += myloss(out.contiguous().view(batch_size, -1), 
                              yy.contiguous().view(batch_size, -1)).item()
            # all_abs_trace += abs_trace
            # all_rel_trace += rel_trace

    if args.use_wandb:
        wandb.log({
            "test l2": test_l2/args.ntest
        })
    print(f"Test Loss {test_l2/args.ntest}")
    # print(f"Abs trace {all_abs_trace/args.ntest}")
    # print(f"Abs trace {all_rel_trace/args.ntest}")

@hydra.main(version_base=None, config_path="./configs/ss_navier_stokes", config_name="config")
def main(args: DictConfig):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 

    print(args)
    if args.use_wandb:
        wandb_config = OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True)
        run_name = (
                f"{args.model}_train_{args.ntrain}"
                f"bz_{args.batch_size}_noise_level_{args.noise_level}_lr_{args.lr}_torchdeq"
        )
        group_name = (
                f"{args.wandb_prefix}_{args.model}"
        )
        wandb.init(
                project=f"Final-Steady-State-PDE-DEQ-{args.pde_type.lower()}-Test",
                group=group_name,
                name=run_name,
                config=wandb_config
        )

    model = get_model(args)
    

    from data.navier_stokes_dataloader import  load_data_orig
    train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_data_orig(args)

    if args.train:
        train(args, train_loader, val_loader, y_normalizer, model)
    test(args, test_loader, y_normalizer, model)
    print("Done!")

if __name__=='__main__':
    main()
