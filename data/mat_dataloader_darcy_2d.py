import os
from utils.utilities3 import *

# ################ Original data loader from FNO 
def load_data_orig(args):
    BASE_PATH = args.data_base_path
    ntrain = args.ntrain
    ntest = args.ntest

    r = args.sub
    h = int(((args.res - 1)/r) + 1)
    s = h

    # max permissoble variance in added gaussian
    noise_levels = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 
                                    1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1]

    valid_noise_levels = [nl for nl in noise_levels if nl <= args.noise_level]

    samples_per_noise_level = ntrain // len(valid_noise_levels)

    reader = MatReader(os.path.join(BASE_PATH, f'{args.dataset}'))
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

    if args.test_dataset is not None:
        reader.load_file(os.path.join(BASE_PATH, args.test_dataset))
        x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
        y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]
    else:
        ValueError("Need to provide path for test dataset")


    # If noisy data is true, add some noise to observations
    if args.noisy_data:
        if not args.add_noise_to_inputs:
            for nl_idx, st in enumerate(range(0, ntrain, samples_per_noise_level)):
                if nl_idx < len(valid_noise_levels):
                    print(f"Adding gaussian noise to input with std "
                                f"{np.sqrt(valid_noise_levels[nl_idx])}")
                    x_train[st:st+samples_per_noise_level] += torch.normal(
                            mean=0,
                            std=np.sqrt(valid_noise_levels[nl_idx]),
                            size=x_train[st:st+samples_per_noise_level].shape)
        else:
            for nl_idx, st in enumerate(range(0, ntrain, samples_per_noise_level)):
                if nl_idx < len(valid_noise_levels):
                    print(f"Adding gaussian noise to observations with std "
                                f"{np.sqrt(valid_noise_levels[nl_idx])}")
                    y_train[st:st+samples_per_noise_level] += torch.normal(
                            mean=0,
                            std=np.sqrt(valid_noise_levels[nl_idx]),
                            size=y_train[st:st+samples_per_noise_level].shape)

    x_normalizer, y_normalizer = None, None
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    # Reshape to add the channel information since we'll be using conv
    x_train = x_train.reshape(ntrain,s,s,1) 
    x_test  = x_test.reshape(ntest,s,s,1)

    # Should this be before or after adding Gaussian noise to labels?
    # Rerun experiments
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    train_data = torch.utils.data.TensorDataset(x_train, y_train) 
    val_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    
    args.ntrain = len(train_data)
    args.ntest = len(test_data)

    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=args.batch_size, 
            shuffle=False)
    test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=args.batch_size, 
            shuffle=False)
    return train_loader, val_loader, test_loader, x_normalizer, y_normalizer