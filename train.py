import os

# Ensure a usable temp directory for PyTorch (fixes FileNotFoundError when /tmp is missing/unwritable)
_fallback_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
os.makedirs(_fallback_tmp, exist_ok=True)
os.environ.setdefault("TMPDIR", _fallback_tmp)

import random
import time
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import SimpleNet, ResNet20, ResNet18CIFAR, WideResNet28_2

# Load config (YAML for easy editing)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def set_seed(seed: int):
    """Fix reproducibility: set seeds and deterministic flags."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Faster for fixed-size images (fine with same input size per run)


class CutoutTransform:
    """Cutout applied after ToTensor+Normalize."""

    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):
        c, h, w = img.shape
        y = random.randint(0, max(0, h - self.size))
        x = random.randint(0, max(0, w - self.size))
        img = img.clone()
        img[:, y : y + self.size, x : x + self.size] = 0.0
        return img


def get_transforms(augment: bool = False, cutout_size: int = 0):
    """Get transforms: augment=True for training, False for val/test."""
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    if augment:
        aug_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *base,
        ]
        if cutout_size > 0:
            aug_list.append(CutoutTransform(size=cutout_size))
        return transforms.Compose(aug_list)
    return transforms.Compose(base)


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Apply different transforms to train vs val subset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def get_loaders():
    data_dir = config['paths']['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    cutout_size = int(config['hyperparameters'].get('cutout_size', 0))
    transform_train = get_transforms(augment=True, cutout_size=cutout_size)
    transform_eval = get_transforms(augment=False)

    # Load raw (no transform) so we can apply different transforms for train vs val
    # Same root = single download; train=True/False selects split
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

    val_split = config['hyperparameters'].get('val_split', 0.1)
    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    subset_train, subset_val = random_split(dataset_train_full, [n_train, n_val])

    dataset_train = _SubsetWithTransform(subset_train, transform_train)
    dataset_val = _SubsetWithTransform(subset_val, transform_eval)

    hp = config['hyperparameters']
    num_workers = hp.get('num_workers', 0)
    kw = {
        'batch_size': hp['batch_size'],
        'num_workers': num_workers,
        'pin_memory': hp.get('pin_memory', False) and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kw['persistent_workers'] = hp.get('persistent_workers', False)
        kw['prefetch_factor'] = hp.get('prefetch_factor', 2)

    dataloader_train = DataLoader(dataset_train, shuffle=True, **kw)
    dataloader_val = DataLoader(dataset_val, shuffle=False, **kw)
    dataloader_test = DataLoader(dataset_test, shuffle=False, **kw)
    return dataloader_train, dataloader_val, dataloader_test


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])


def evaluate(model, dataloader, criterion, device, use_tta=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if use_tta:
                out1 = model(images)
                out2 = model(torch.flip(images, [-1]))
                outputs = (out1 + out2) / 2
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total, 100 * correct / total


def train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device, use_amp, scaler=None, ema=None):
    model.train()
    epochs = config['hyperparameters']['epochs']
    hp = config['hyperparameters']

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}", leave=True, unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    if not torch.isfinite(loss):
                        print("Non-finite loss detected, skipping batch")
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if not torch.isfinite(loss):
                        print("Non-finite loss detected, skipping batch")
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                if ema is not None:
                    ema.update(model)

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (i + 1)})

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        avg_train_loss = running_loss / len(dataloader_train)
        val_loss, val_acc = evaluate(model, dataloader_val, criterion, device)
        print(f"Epoch {epoch+1} | LR: {lr:.6f} | Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}%")

    print('Finished Training')


def test(model, dataloader_test, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {acc:.2f}%')
    return acc


def build_model(device):
    model_name = config.get('model', 'SimpleNet')
    if model_name == 'ResNet20':
        model = ResNet20(num_classes=10).to(device)
    elif model_name == 'ResNet18CIFAR':
        model = ResNet18CIFAR(num_classes=10).to(device)
    elif model_name == 'WideResNet28_2':
        dropout = float(config['hyperparameters'].get('dropout', 0.0))
        model = WideResNet28_2(num_classes=10, dropout=dropout).to(device)
    else:
        model = SimpleNet().to(device)

    if config.get('use_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)
    return model


def build_optimizer(model):
    hp = config['hyperparameters']
    name = hp.get('optimizer', 'SGD')
    lr = float(hp['lr'])
    weight_decay = float(hp.get('weight_decay', 0))

    if name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    momentum = float(hp.get('momentum', 0.9))
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def build_scheduler(optimizer):
    hp = config['hyperparameters']
    sched = hp.get('scheduler')
    epochs = int(hp['epochs'])
    warmup = int(hp.get('warmup_epochs', 0))

    if sched == 'cosine':
        def lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / warmup
            prog = (ep - warmup) / (epochs - warmup)
            return 0.5 * (1 + __import__('math').cos(__import__('math').pi * prog))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if sched == 'step':
        milestones = hp.get('lr_milestones', [60, 120])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return None


def main():
    seed = config.get('seed', 42)
    set_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataloader_train, dataloader_val, dataloader_test = get_loaders()

    model = build_model(device)

    label_smoothing = float(config['hyperparameters'].get('label_smoothing', 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    use_amp = config.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    use_ema = config.get('use_ema', False)
    ema = EMA(model, decay=float(config.get('ema_decay', 0.999))) if use_ema else None

    use_tta = config.get('use_tta', False)

    t0 = time.perf_counter()
    train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device, use_amp, scaler, ema)
    train_time = time.perf_counter() - t0

    eval_model = model
    if ema is not None:
        ema.apply(model)
        eval_model = model
    test_loss, test_acc = evaluate(eval_model, dataloader_test, criterion, device, use_tta=use_tta)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')
    print(f'Total training time: {train_time:.1f}s')

    os.makedirs(os.path.dirname(config['paths']['model_path']) or '.', exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_path'])
    print(f"Model saved to {config['paths']['model_path']}")


if __name__ == '__main__':
    main()
