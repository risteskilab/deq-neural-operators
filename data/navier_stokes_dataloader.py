from typing import List
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt
import glob
import sys
import random
sys.path.append("lib/")
sys.path.append("../")

from utils.utilities3 import UnitGaussianNormalizer

NOISE_LEVELS = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 
                1e-4, 1e-3, 2e-3, 4e-3, 1e-2, 1e-1, 5e-1, 1]

DATASET_STAT_DICT = {
    'full_kolmogorov_wavenum_1_mv_5': {
        'x': {'mean': 0, 'var': 0.0030435901135206223},
        'y': {'mean': 0, 'var': 0.11162283271551132}
    },
    'full_type_noise__forcing_kolmogorov_visc_0.0001_mv_5_peak_wave_num_1': {
        'x': {'mean': 0, 'var': 0.0034190488513559103},
        'y': {'mean': 0, 'var': 0.11464705318212509}
    },
    'full_type_noise__forcing_kolmogorov_visc_0.01_mv_5_peak_wave_num_1': {
        'x': {'mean': 0, 'var': 0.001515210373327136},
        'y': {'mean': 0, 'var': 0.08820350468158722}
    }
}

class FolderDataloader(Dataset):
    def __init__(
        self, 
        frame_number: List[int] = [-1], 
        saved_folder: str ='../data/',
        file_prefix: str = 'example',
        num_examples: int = 1000,
        data_source: str = 'jax_cfd',
        input_time_steps: int = None,
        output_time_steps: int = -1,
        sub_resolution: int = 1,
        time_resolution: int = 1, 
        noise_level: int = 0,
        is_valid: bool = False,
        is_test: bool = False,
        valid_examples: float = 1000,
        file_type: str = '.npy',
        test_base_index: int = 0,
        precomp_stats_file: str = None, 
        pix_normalize: bool = True,
        add_noise_to_inp: bool = False,
        ):

        if frame_number[0] < 0:
            self.no_frame = True
        else:
            assert max(frame_number) < 5, "Frame number cannot be greater than 5"
            assert min(frame_number) > 0, "Frame number cannot be greater than 0"
            self.no_frame = False

        assert data_source in ['jax_cfd'], "invalid datasource"
        
        self.frame_number = frame_number
        self.saved_folder = saved_folder
        self.file_prefix = file_prefix
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.is_valid = is_valid
        self.is_test = is_test
        self.valid_examples = valid_examples
        self.sub_resolution = sub_resolution
        self.time_resolution = time_resolution
        self.file_type = file_type
        self.noise_level = noise_level
        self.data_source = data_source
        self.precomp_stats_file = precomp_stats_file
        self.pix_normalize = pix_normalize
        self.add_noise_to_inp = add_noise_to_inp
        
        print(f"Reading from {self.saved_folder} ")
        self.filenames = self.get_filenames(
                file_folder=self.saved_folder,
                file_type=file_type,)
        assert len(self.filenames) > 0, "Files not read from directory"

        if is_test:
            self.num_examples = np.min([num_examples, len(self.filenames)])
            self.base_idx = test_base_index
        else:
            if is_valid:
                self.num_examples = valid_examples
                self.base_idx = len(self.filenames)  - valid_examples
            else:
                self.num_examples = np.min(
                        [num_examples, len(self.filenames) - valid_examples])
                self.base_idx = 0
        assert num_examples > 0, "num_examples cannot be 0"

        self.noise_list = self.get_noise_list(
                num_examples=self.num_examples,
                noise_level=noise_level)


    def get_filenames(self, file_folder: str, file_type: str):
        path = os.path.join(file_folder, f'{self.file_prefix}_*{file_type}')
        filenames = glob.glob(path)
        return filenames

    def get_meshgrid(self):
        grid_list = np.load(os.path.join(self.saved_folder, 'grid.npy'))
        grid_np = np.stack(grid_list, axis=-1)
        grid = torch.tensor(grid_np)[::self.sub_resolution, ::self.sub_resolution]
        return grid

    def get_noise_list(self, num_examples: int, noise_level: float):
        noise_list = []
        valid_noise_levels = [nl for nl in NOISE_LEVELS if nl <= self.noise_level]
        samples_per_noise_level = num_examples // len(valid_noise_levels) + 1
        for nl_idx, st in enumerate(range(0, num_examples, samples_per_noise_level)):
            if nl_idx < len(valid_noise_levels):
                print(f"Adding gaussian noise to inputs with std "
                      f"{np.sqrt(valid_noise_levels[nl_idx])}")
                noise_list = (noise_list 
                              + [valid_noise_levels[nl_idx]] 
                              * samples_per_noise_level)
        noise_list = noise_list[:num_examples]
        assert len(noise_list) == num_examples, "noise not covered for the entire array"
        return noise_list
    
    def __getitem__(self, index):
        real_index = self.base_idx + index
        filename = os.path.join(
            self.saved_folder, f'{self.file_prefix}_{real_index:05}{self.file_type}')
        grid = None
        data = np.load(filename, allow_pickle=True)
        grid = self.get_meshgrid()
        if self.no_frame:
            x = data.item()['f_curl'].transpose(1,2,0)
            y = data.item()['w'].transpose(1,2,0)
        else:
            fm = np.random.choice(self.frame_number)
            x = data.item()['f'][fm].transpose(1, 2, 0)
            y = data.item()['w'][fm]
        x = x[::self.sub_resolution, 
                ::self.sub_resolution]
        y = y[::self.sub_resolution, 
                ::self.sub_resolution]

        if (not self.is_valid) and (not self.is_test):
            if self.add_noise_to_inp:
                x += np.random.normal(loc=0, 
                                        scale=np.sqrt(self.noise_list[index]),
                                        size=x.shape)
            else:
                y += np.random.normal(loc=0, 
                            scale=np.sqrt(self.noise_list[index]),
                            size=y.shape)
        
        if self.pix_normalize:
            if False:
                x = (x - x.mean())/ x.std()
                y = (y - y.mean()) / y.std()

            saved_folder = self.saved_folder.split("/")[-1]
            # NOTE (TM) Addition
            x = ((x - DATASET_STAT_DICT[saved_folder]['x']['mean']) 
                 / np.sqrt(DATASET_STAT_DICT[saved_folder]['x']['var']))
            y = ((y - DATASET_STAT_DICT[saved_folder]['y']['mean']) 
                 / np.sqrt(DATASET_STAT_DICT[saved_folder]['y']['var']))
        
        if grid is not None:
            return (torch.tensor(x, dtype=torch.float32), 
                    torch.tensor(y, dtype=torch.float32), 
                    grid)
        else:
            return (torch.tensor(x, dtype=torch.float32), 
                    torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return self.num_examples

def load_data_orig(args):
    # max permissoble variance in added gaussian
    # noise_levels = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 
    #                 1e-2, 1e-1, 5e-1, 1, 2.5, 5, 7.5, 10]
    if args.test_base_index < 0:
        test_base_index = args.ntrain + 1
    else:
        test_base_index = args.test_base_index


    train_data = FolderDataloader(
                                frame_number=args.frame_number,
                                saved_folder=args.data_base_path,
                                noise_level=args.noise_level,
                                num_examples=args.ntrain,
                                valid_examples=args.ntest,
                                sub_resolution=args.sub,
                                precomp_stats_file=args.precomp_stats_file,
                                add_noise_to_inp=args.add_noise_to_inputs
                                )
    ## val data and test data should be clean
    val_data = FolderDataloader(
                                frame_number=args.frame_number,
                                saved_folder=args.data_base_path,
                                noise_level=0,
                                num_examples=args.ntrain,
                                sub_resolution=args.sub,
                                valid_examples=args.ntest,
                                is_valid=True,
                                is_test=False,
                                precomp_stats_file=args.precomp_stats_file
                                )
    test_data = FolderDataloader(frame_number=args.frame_number,
                                saved_folder=args.data_base_path,
                                noise_level=0,
                                num_examples=args.ntest,
                                test_base_index=test_base_index,
                                sub_resolution=args.sub,
                                valid_examples=args.ntest,
                                is_valid=False,
                                is_test=True,
                                precomp_stats_file=args.precomp_stats_file
                                )

    args.ntrain = len(train_data)
    args.ntest = len(test_data) 

    print("Total training samples", args.ntrain, "testing ", args.ntest)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               num_workers=args.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=False)
    
    x_normalizer, y_normalizer = None, None
    
    # if args.precomp_stats_file is not None:
    #     x_normalizer = UnitGaussianNormalizer(
    #           mean=norm_stats['mean_x'], std=norm_stats['std_x'])
    #     y_normalizer = UnitGaussianNormalizer(mean=norm_stats['mean_y'], std=norm_stats['std_y'])

    return train_loader, val_loader, test_loader, x_normalizer, y_normalizer

if __name__=='__main__':
    from attrdict import AttrDict
    args = AttrDict({
        'batch_size': 5000,
        'num_workers': 1,
        'noise_level': 0,
        'data_base_path': '/home/tmarwah/projects/jax-equations/src/steady_state_ns/data_total_wavenum_1_mv_5',
        #'data_base_path': '/project_data/datasets/apokle_exp/deq-pde/PDE_datasets/jax_cfd_ns/data_with_forcing',
        'frame_number': [1],
        'sub': 4,
        'ntrain': 5000,
        'ntest': 500
    })
    train_loader, val_loader, test_loader,_,_ = load_data_orig(args)
    train_data = next(iter(train_loader))
    mean_x = torch.mean(train_data[0], 0)
    std_x = torch.std(train_data[0], 0)
    mean_y = torch.mean(train_data[1], 0)
    std_y = torch.std(train_data[1], 0)
    torch.save({'mean_x': mean_x, 
                'std_x': std_x,
                'mean_y': mean_y, 
                'std_y': std_y,
                }, f"ns_forcing_{args.ntrain}_{args.sub}_{args.frame_number[0]}_wavenum_1_mv_5.pt")
