import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


def plot_loss(train_losses, val_losses):
    """
    Plots the train and validation loss.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="red")
    plt.title("Train and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics(train_metrics, val_metrics):
    """
    Plots the train and validation metrics.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label="Train Metric", color="blue")
    plt.plot(val_metrics, label="Val Metric", color="red")
    plt.title("Train and Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.show()


def image():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create().to(device)

    checkpoint = torch.load('/content/drive/MyDrive/grace_model_wblur.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    img = Image.open('/content/drive/MyDrive/LOL-v2/Test/input/00690.png').convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)  # move to device
    with torch.no_grad():
        output = model(input_tensor)

    output_image = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img_out = Image.fromarray((output_image * 255).astype(np.uint8))

    img_out.save('/content/drive/MyDrive/LOL-v2/00690_grace.png')
    img_out.show()
