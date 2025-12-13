import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import random


def get_dataloaders(
    data_dir,
    batch_size=64,
    img_size=224,
    num_workers=4,
):

    train_gt = os.path.join(data_dir, "Train/gt")
    train_input = os.path.join(data_dir, "Train/input")

    val_gt = os.path.join(data_dir, "Test/gt")
    val_input = os.path.join(data_dir, "Test/input")

    train_input_paths = sorted(os.listdir(train_input))
    train_gt_paths    = sorted(os.listdir(train_gt))

    val_input_paths = sorted(os.listdir(val_input))
    val_gt_paths    = sorted(os.listdir(val_gt))

    train_pairs = [
        (os.path.join(train_input, f_in),
         os.path.join(train_gt,    f_gt))
        for f_in, f_gt in zip(train_input_paths, train_gt_paths)
    ]

    val_pairs = [
        (os.path.join(val_input, f_in),
         os.path.join(val_gt,    f_gt))
        for f_in, f_gt in zip(val_input_paths, val_gt_paths)
    ]

    def collate_pairs(batch):
        # batch is: [(inp_path, gt_path), ...]
        inputs = [item[0] for item in batch]
        gts    = [item[1] for item in batch]
        return inputs, gts

    train_loader = DataLoader(
        train_pairs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pairs
    )

    val_loader = DataLoader(
        val_pairs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pairs
    )

    return train_loader, val_loader


# ------------------------------
# TRAIN FUNCTION
# ------------------------------


def train(model, loader, criterion, optimizer, device, img_size=384):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    total_images = 0

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    for inputs_paths, gts_paths in loader:
        # Load images and move to device
        inputs = torch.stack([transform(Image.open(p).convert("RGB")) for p in inputs_paths]).to(device)
        gts    = torch.stack([transform(Image.open(p).convert("RGB")) for p in gts_paths]).to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, gts)
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * inputs.size(0)

        # Compute PSNR
        mse = torch.mean((preds - gts) ** 2, dim=[1,2,3])  # per image
        psnr = 10 * torch.log10(1.0 / mse)                 # assuming images in [0,1]
        running_psnr += psnr.sum().item()

        total_images += inputs.size(0)

    avg_loss = running_loss / total_images
    avg_psnr = running_psnr / total_images

    return avg_loss, avg_psnr


# ------------------------------
# EVAL FUNCTION
# ------------------------------


def eval_model(model, loader, criterion, device, img_size=384):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    total_images = 0

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    with torch.no_grad():
        for inputs_paths, gts_paths in loader:
            # Load images and move to device
            inputs = torch.stack([transform(Image.open(p).convert("RGB")) for p in inputs_paths]).to(device)
            gts    = torch.stack([transform(Image.open(p).convert("RGB")) for p in gts_paths]).to(device)

            preds = model(inputs)
            loss = criterion(preds, gts)

            running_loss += loss.item() * inputs.size(0)

            # Compute PSNR
            mse = torch.mean((preds - gts) ** 2, dim=[1,2,3])
            psnr = 10 * torch.log10(1.0 / mse)
            running_psnr += psnr.sum().item()

            total_images += inputs.size(0)

    avg_loss = running_loss / total_images
    avg_psnr = running_psnr / total_images

    return avg_loss, avg_psnr
