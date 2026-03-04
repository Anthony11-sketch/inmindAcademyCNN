import os

# Ensure a usable temp directory for PyTorch (fixes FileNotFoundError when /tmp is missing/unwritable)
_fallback_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
os.makedirs(_fallback_tmp, exist_ok=True)
os.environ.setdefault("TMPDIR", _fallback_tmp)

import random
import time
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import WideResNet28_2

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


def get_transforms(augment: bool = False, cutout_size: int = 0, use_randaugment: bool = False):
    """Get transforms: augment=True for training, False for val/test."""
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    if augment:
        aug_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if use_randaugment:
            aug_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        aug_list.extend(base)
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


def get_loaders(use_randaugment=False):
    """Build dataloaders. use_randaugment overrides config for phased training."""
    data_dir = config['paths']['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    cutout_size = int(config['hyperparameters'].get('cutout_size', 0))
    if use_randaugment:
        cutout_size = 0  # Don't stack RandAugment + Cutout
    transform_train = get_transforms(augment=True, cutout_size=cutout_size, use_randaugment=use_randaugment)
    transform_eval = get_transforms(augment=False)

    # Standard CIFAR split: train=50k, test=10k (no random val split)
    dataset_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )
    dataset_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_eval,
    )

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
    dataloader_test = DataLoader(dataset_test, shuffle=False, **kw)
    return dataloader_train, dataloader_test


def cutmix_data(inputs, labels, alpha=1.0):
    """CutMix augmentation. Returns mixed inputs, label_a, label_b, lam."""
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    W, H = inputs.size(3), inputs.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return inputs, labels, labels[index], lam


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


def train(model, dataloader_train_a, dataloader_train_b, dataloader_eval, criterion, optimizer, scheduler, device, use_amp, scaler=None, ema=None, cutmix_alpha=0.0, start_randaugment_epoch=30, start_cutmix_epoch=30, model_path=None, use_tta=False):
    model.train()
    epochs = config['hyperparameters']['epochs']
    hp = config['hyperparameters']
    use_cutmix = cutmix_alpha > 0
    cutmix_prob = float(config.get('cutmix_prob', 0.5))
    cutmix_alpha_end = float(config.get('cutmix_alpha_end', 1.0))
    cutmix_alpha_ramp_epochs = int(config.get('cutmix_alpha_ramp_epochs', 0))
    best_acc = -1.0

    for epoch in range(epochs):
        # Phase A: RandomCrop+Flip only. Phase B: RandAugment + CutMix
        dataloader_train = dataloader_train_b if epoch >= start_randaugment_epoch else dataloader_train_a
        epochs_since_cutmix = max(0, epoch - start_cutmix_epoch)
        if cutmix_alpha_ramp_epochs <= 0 or epochs_since_cutmix >= cutmix_alpha_ramp_epochs:
            eff_cutmix_alpha = cutmix_alpha_end
        else:
            eff_cutmix_alpha = cutmix_alpha + (cutmix_alpha_end - cutmix_alpha) * (epochs_since_cutmix / cutmix_alpha_ramp_epochs)
        use_amp_this_epoch = use_amp and device.type == 'cuda'

        running_loss = 0.0
        with tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}", leave=True, unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)

                do_cutmix = False
                lam = 1.0
                labels_b = labels
                if use_cutmix and (epoch >= start_cutmix_epoch) and (np.random.rand() < cutmix_prob):
                    do_cutmix = True
                    inputs = inputs.clone()
                    inputs, labels, labels_b, lam = cutmix_data(inputs, labels, eff_cutmix_alpha)

                optimizer.zero_grad()

                if use_amp_this_epoch:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(inputs)
                        if do_cutmix:
                            loss = lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, labels_b)
                        else:
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
                    if do_cutmix:
                        loss = lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, labels_b)
                    else:
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
        eval_loss, eval_acc = evaluate(model, dataloader_eval, criterion, device, use_tta=use_tta)
        print(f"Epoch {epoch+1} | LR: {lr:.6f} | Train loss: {avg_train_loss:.3f} | Eval loss: {eval_loss:.3f} | Eval acc: {eval_acc:.2f}%")

        # Save best checkpoint by eval accuracy
        if model_path and eval_acc > best_acc:
            best_acc = eval_acc
            os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best checkpoint (acc: {eval_acc:.2f}%)")

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
    dropout = float(config['hyperparameters'].get('dropout', 0.0))
    model = WideResNet28_2(num_classes=10, dropout=dropout).to(device)

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
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


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

    dataloader_train_a, dataloader_test = get_loaders(use_randaugment=False)
    dataloader_train_b, _ = get_loaders(use_randaugment=True)

    model = build_model(device)
    print("MODEL: WideResNet28_2 | PARAMS:", sum(p.numel() for p in model.parameters()))

    label_smoothing = float(config['hyperparameters'].get('label_smoothing', 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    use_amp = config.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    use_ema = config.get('use_ema', False)
    ema = EMA(model, decay=float(config.get('ema_decay', 0.999))) if use_ema else None

    use_tta = config.get('use_tta', False)
    cutmix_alpha = float(config.get('cutmix_alpha', 0.0))
    start_randaugment_epoch = int(config.get('start_randaugment_epoch', 30))
    start_cutmix_epoch = int(config.get('start_cutmix_epoch', 30))

    t0 = time.perf_counter()
    train(model, dataloader_train_a, dataloader_train_b, dataloader_test, criterion, optimizer, scheduler, device, use_amp, scaler, ema, cutmix_alpha=cutmix_alpha, start_randaugment_epoch=start_randaugment_epoch, start_cutmix_epoch=start_cutmix_epoch, model_path=config['paths']['model_path'], use_tta=use_tta)
    train_time = time.perf_counter() - t0

    eval_model = model
    if ema is not None:
        ema.apply(model)
        eval_model = model
    test_loss, test_acc = evaluate(eval_model, dataloader_test, criterion, device, use_tta=use_tta)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')
    print(f'Total training time: {train_time:.1f}s')

    # Best checkpoint already saved during training; save final model
    os.makedirs(os.path.dirname(config['paths']['model_path']) or '.', exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_path'])
    print(f"Model saved to {config['paths']['model_path']}")


if __name__ == '__main__':
    main()
