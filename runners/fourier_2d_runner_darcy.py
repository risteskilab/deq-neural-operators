import sys
import os
import seaborn as sns 

sys.path.append("lib/")
sys.path.append("../")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from utils.utilities3 import *
from optimizer.adam import Adam
from timeit import default_timer

from models.fourier_2d_deq_shallow import FNO2dDEQShallow
from models.fourier_2d_deq import FNO2dDEQ
from models.fourier_2d_deep import FNO2dDeep
from models.fourier_2d_deep_no_inj import FNO2d
from models.fourier_2d_deep_shallow import FNO2dDeepSmall
from data.mat_dataloader_darcy_2d import load_data_orig


import wandb

def get_model(args):
    if args.model == 'deq' or args.model == 'wt':
        model = FNO2dDEQ(
            args.modes,
            args.modes, 
            width=args.width, 
            f_solver=args.solver, 
            b_solver=args.solver, 
            f_thres=args.solver_steps, 
            b_thres=args.solver_steps,
            block_depth=args.depth_per_block,
            add_mlp=args.add_mlp,
            pretrain_steps=args.pretrain_steps,
            pretrain_iter_steps=args.pretrain_iter_steps,
            in_channels=args.in_channels, 
            out_channels=args.out_channels,
            use_pg=args.use_pg,
            tau=args.tau,
            pg_steps=args.pg_steps
            ).cuda()
    elif args.model == 'non-wt':
        model = FNO2dDeep(args.modes,
                          args.modes, 
                          width=args.width, 
                          block_depth=args.depth_per_block,
                          add_mlp=args.add_mlp).cuda()
    elif args.model == 'non-wt-no-inj':
        model = FNO2d(args.modes, 
                      args.modes, 
                      width=args.width, 
                      block_depth=args.depth_per_block,
                      add_mlp=args.add_mlp).cuda()
    elif args.model == 'shallow-deq' or args.model == 'shallow-wt':
        model = FNO2dDEQShallow(
            args.modes,
            args.modes, 
            width=args.width, 
            f_solver=args.solver, 
            b_solver=args.solver, 
            f_thres=args.solver_steps, 
            b_thres=args.solver_steps,
            block_depth=args.depth_per_block,
            add_mlp=args.add_mlp,
            pretrain_steps=args.pretrain_steps,
            pretrain_iter_steps=args.pretrain_iter_steps).cuda()
    elif args.model == 'shallow-non-wt':
        model = FNO2dDeepSmall(args.modes,
                          args.modes, 
                          width=args.width, 
                          block_depth=args.depth_per_block,
                          add_mlp=args.add_mlp).cuda()
    elif args.model == 'simple-deq' or args.model == 'simple-wt':
        model = FNO2dSimpleDEQ(
            args.modes,
            args.modes, 
            width=3,  # We are not assuming any channels 
            # width=1,  # we are not going to append a grid either in this case
            f_solver=args.solver, 
            b_solver=args.solver, 
            f_thres=args.solver_steps, 
            b_thres=args.solver_steps,
            block_depth=args.depth_per_block,
            add_mlp=args.add_mlp,
            pretrain_steps=args.pretrain_steps,
            pretrain_iter_steps=args.pretrain_iter_steps).cuda()
    else:
        raise ValueError(f"Unknown model {args.model}")
    return model


def train(args, train_loader, val_loader, y_normalizer, model, t_train=101, initial_step=10):
    print(count_params(model))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iterations = args.epochs*(args.ntrain//args.batch_size)
    if args.lr_scheduler == 'cosine':
        print("Using cosine scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError(f"Unknown LR scheduler {args.lr_scheduler}")

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

    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            s = x.shape[1]

            optimizer.zero_grad()

            out = model(x, 
                        train_step=train_step, 
                        iters=iters, 
                        eps=args.eps)
            out = out.reshape(batch_size, s, s)

            if y_normalizer is not None:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            
            train_step += 1

            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_l2 += l2.item()

            if args.lr_scheduler == 'cosine':
                scheduler.step()

        if args.lr_scheduler == 'step':
            scheduler.step()

        model.eval()
        test_l2 = 0.0
        # No noise on validation set!!!
        with torch.no_grad():
            for xx, yy in val_loader:
                batch_size = xx.shape[0]
                s = xx.shape[1]
                xx, yy = xx.cuda(), yy.cuda()
                out = model(xx, iters=iters).reshape(batch_size, s, s)
                if y_normalizer is not None:
                    out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), yy.view(batch_size, -1)).item()
                
        train_l2 /= args.ntrain
        test_l2 /= args.ntest

        t2 = default_timer()
        print(ep, t2-t1, train_l2, test_l2)
        if args.use_wandb:
            wandb.log({
                "train l2": train_l2,
                "val l2": test_l2
            })
        if ep % args.logging_freq == 0:
            ckpt_path = f"{model_folder_path}/checkpoint_{ep}.pth"
            torch.save(model, ckpt_path)

    model_save_path = f'{model_folder_path}/{args.ckpt}.pth'
    print(f'will save model to {model_save_path}')
    torch.save(model, model_save_path)

def test(args, test_loader, y_normalizer, model):
    print(count_params(model))

    myloss = LpLoss(size_average=False)
    if y_normalizer is not None:
        y_normalizer.cuda()
    ### set to -1 for DEQ mode
    if 'deq' in args.model:
        #iters = -1
        iters = args.solver_steps
    else:
        iters = args.solver_steps

    model_folder_path = os.path.join(
            args.model_base_path,
            args.model_save_folder_path)
    model_save_path = f'{model_folder_path}/{args.ckpt}.pth'
    saved_model = torch.load(f'{model_save_path}')
    model.load_state_dict(saved_model.state_dict())

    model.eval()
    cm = sns.color_palette("icefire", as_cmap=True)
    test_l2 = 0
    # Don't add any noise to observations at test time too
    print(f"Testing on {args.ntest} clean samples")
    with torch.no_grad():
        for xx, yy in test_loader:
            batch_size = xx.shape[0]
            s = xx.shape[1]
            xx, yy = xx.cuda(), yy.cuda()
            out = model(xx, iters=iters, eps=args.eps, wandb_log=False)
            out = out.reshape(batch_size, s, s)
            if y_normalizer is not None:
                out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), yy.view(batch_size, -1)).item()
    if args.use_wandb:
        wandb.log({
            "test l2": test_l2/args.ntest
        })
    print(f"Test Loss {test_l2/args.ntest}")

@hydra.main(config_path="./configs/darcy", config_name="config")
def main(args: DictConfig):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(args)
    if args.use_wandb:
        wandb_config = OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True)
        run_name = (
                f"{args.model}_train_{args.ntrain}"
                f"bz_{args.batch_size}_ep_{args.epochs}_lr_{args.lr}"
                f"_{args.solver}{args.solver_steps}_eps_{args.eps}"
        )
        group_name = (
                f"{args.lr_scheduler}_lr_{args.model}_noise_level_{args.noise_level}_"
                f"noise2inp_{args.add_noise_to_inputs}_res_{args.res}_sub_{args.sub}"
        )
        wandb.init(
                project=f"Final-Steady-State-PDE-DEQ-{args.pde_type.lower()}",
                group=group_name,
                name=run_name,
                config=wandb_config
        )

    model = get_model(args)
    
    train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_data_orig(args)

    if args.train:
        train(args, train_loader, val_loader, y_normalizer, model)
    test(args, test_loader, y_normalizer, model)
    print("Done!")

if __name__=='__main__':
    main()
