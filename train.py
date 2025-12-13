import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision

from utils.loss_utils import L1Loss, EdgeLoss, FrequencyLoss
from utils.train_utils import get_dataloaders, train, eval_model
from .model import create


class ComplexLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # Pixel Loss
        self.pixel_loss = L1Loss(loss_weight=1.0, reduction='mean')

        # Edge Loss
        self.edge_loss = EdgeLoss(rank=device, loss_weight=50, criterion="l1", reduction="mean")

        # Frequency Loss
        self.frequency_loss = FrequencyLoss(loss_weight=1.0, criterion="l1", reduction="mean")

    def forward(self, enhanced, target, outside=None):
        """
        enhanced: model output
        target: ground truth
        outside: optional input for enhance_loss, defaults to enhanced
        """
        if outside is None:
            outside = enhanced

        total_loss = 0
        total_loss += self.pixel_loss(enhanced, target)
        total_loss += self.edge_loss(enhanced, target)
        total_loss += self.frequency_loss(enhanced, target)

        return total_loss


def main():
    data_dir = "/content/drive/MyDrive/LOL-v2"
    batch_size = 16
    img_size = 256
    epochs = 100
    lr = 5e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    train_loader, val_loader = get_dataloaders(
        data_dir, batch_size, img_size
    )

    model = create().to(device)
    criterion = ComplexLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.9))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_losses = []
    val_losses = []

    train_metrics = []
    val_metrics = []

    for epoch in range(1, epochs + 1):
        train_loss, train_psnr = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr = eval_model(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"[Epoch {epoch}] Train Metric: {train_psnr:.4f} | Val Metric: {val_psnr:.4f}")

        train_losses.append(train_loss)
        train_metrics.append(train_psnr)

        val_losses.append(val_loss)
        val_metrics.append(val_psnr)

        scheduler.step()

    torch.save(model.state_dict(), "/content/drive/MyDrive/grace_model_100.pth")
    print("Saved model -> model.pth")

    return train_losses, val_losses, train_metrics, val_metrics


if __name__ == "__main__":
    train_losses, val_losses, train_metrics, val_metrics = main()
