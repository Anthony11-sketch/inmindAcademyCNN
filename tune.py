"""
Hyperparameter optimization for CIFAR-10 CNN.

Supports: random search, Bayesian (TPE), and pruning (ASHA-like early stopping).
Use --sampler random for random search, default is TPE (Bayesian).
Use --pruner to enable early stopping (MedianPruner = ASHA-like).
"""
import argparse
import os
import yaml

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from model import SimpleNet
from train import set_seed, get_transforms, evaluate, _SubsetWithTransform

# Load base config
with open('config.yaml', 'r') as f:
    BASE_CONFIG = yaml.safe_load(f)


def get_tune_loaders(subset_ratio=1.0, batch_size=128, num_workers=4):
    """Get loaders; subset_ratio < 1 uses a smaller subset for cheap tuning."""
    data_dir = BASE_CONFIG['paths']['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    transform_train = get_transforms(augment=True)
    transform_eval = get_transforms(augment=False)

    dataset_train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )
    dataset_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_eval,
    )

    val_split = BASE_CONFIG['hyperparameters'].get('val_split', 0.1)
    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    subset_train, subset_val = random_split(dataset_train_full, [n_train, n_val])

    if subset_ratio < 1.0:
        n_sub = max(1, int(len(subset_train) * subset_ratio))
        indices = torch.randperm(len(subset_train))[:n_sub].tolist()
        subset_train = Subset(subset_train.dataset, [subset_train.indices[i] for i in indices])

    dataset_train = _SubsetWithTransform(subset_train, transform_train)
    dataset_val = _SubsetWithTransform(subset_val, transform_eval)

    kw = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2,
    }
    loader_train = DataLoader(dataset_train, shuffle=True, **kw)
    loader_val = DataLoader(dataset_val, shuffle=False, **kw)
    loader_test = DataLoader(dataset_test, shuffle=False, **kw)
    return loader_train, loader_val, loader_test


def train_trial(model, loader_train, loader_val, criterion, device, params, n_epochs, use_amp, trial=None):
    optimizer_name = params.get('optimizer', 'SGD')
    lr = params['lr']
    weight_decay = params.get('weight_decay', 5e-4)

    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        momentum = params.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Simple cosine schedule over n_epochs
    def lr_lambda(ep):
        import math
        return 0.5 * (1 + math.cos(math.pi * ep / n_epochs))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(n_epochs):
        model.train()
        for inputs, labels in loader_train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        scheduler.step()

        val_loss, val_acc = evaluate(model, loader_val, criterion, device)

        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return val_acc


def objective(trial):
    set_seed(BASE_CONFIG.get('seed', 42))

    # Sample hyperparameters (log-uniform for lr, etc.)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'AdamW'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.8, 0.95) if optimizer == 'SGD' else 0.9

    params = {
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'weight_decay': weight_decay,
        'momentum': momentum,
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = BASE_CONFIG.get('use_amp', False) and device.type == 'cuda'

    loader_train, loader_val, loader_test = get_tune_loaders(
        subset_ratio=args.subset_ratio,
        batch_size=batch_size,
        num_workers=BASE_CONFIG['hyperparameters'].get('num_workers', 4),
    )

    model = SimpleNet().to(device)
    label_smoothing = BASE_CONFIG['hyperparameters'].get('label_smoothing', 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    n_epochs = args.epochs_per_trial

    val_acc = train_trial(model, loader_train, loader_val, criterion, device, params, n_epochs, use_amp, trial)
    return val_acc


parser = argparse.ArgumentParser()
parser.add_argument('--n-trials', type=int, default=20, help='Number of trials')
parser.add_argument('--epochs-per-trial', type=int, default=5, help='Epochs per trial (small for cheap search)')
parser.add_argument('--subset-ratio', type=float, default=0.2, help='Use subset of train data (0.2 = 20%% for fast ranking)')
parser.add_argument('--sampler', choices=['random', 'tpe'], default='tpe', help='random or TPE (Bayesian)')
parser.add_argument('--pruner', action='store_true', help='Enable pruning (ASHA-like early stopping)')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


def main():
    set_seed(args.seed)

    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.sampler == 'tpe' else optuna.samplers.RandomSampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner() if args.pruner else optuna.pruners.NopPruner()

    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print('\n--- Best trial ---')
    print(f'Val acc: {study.best_value:.2f}%')
    print('Params:', study.best_params)

    # Optionally save best params to a yaml for manual copy to config.yaml
    out_path = 'tune_best_params.yaml'
    with open(out_path, 'w') as f:
        yaml.dump(study.best_params, f)
    print(f'Saved best params to {out_path}')


if __name__ == '__main__':
    main()
