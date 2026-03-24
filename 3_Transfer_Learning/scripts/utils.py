import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                    # 3_Transfer_Learning/
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# ImageNet normalization stats (used by all pretrained torchvision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_device():
    """Detect best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dataloaders(batch_size=64, num_workers=2):
    """
    Returns (train_loader, val_loader, test_loader, class_names) for Sports dataset.

    - Train: data augmentation (flip, crop) + ImageNet normalization
    - Val/Test: resize + center crop + ImageNet normalization

    Uses proper 3-split: train for weight updates, val for early stopping,
    test for final evaluation only (used once).
    """
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_test_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"Classes: {len(class_names)}")
    return train_loader, val_loader, test_loader, class_names


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate model on a dataset. Returns (avg_loss, accuracy%)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return running_loss / len(loader), 100. * correct / total


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ── Plot History ─────────────────────────────────────────────────────────────

def plot_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses,   label='Val Loss',   marker='o')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label='Train Acc', marker='o')
    ax2.plot(epochs, val_accs,   label='Val Acc',   marker='o')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(model, loader, device, class_names, save_path=None):
    """
    Build and display a confusion matrix.
    Rows = true labels, Columns = predicted labels.
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs   = model(images)
            _, preds  = outputs.max(1)
            for t, p in zip(labels.numpy(), preds.cpu().numpy()):
                cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()

    # Print most-confused pairs
    print("\nTop confused pairs (true → predicted):")
    off_diag = [(cm[i][j], i, j) for i in range(num_classes)
                                  for j in range(num_classes) if i != j]
    for count, true, pred in sorted(off_diag, reverse=True)[:10]:
        if count > 0:
            print(f"  {class_names[true]} → {class_names[pred]} : {count} times")


# ── Wrong Prediction Gallery ─────────────────────────────────────────────────

def plot_wrong_predictions(model, loader, device, class_names, n=16, save_path=None):
    """
    Collect up to n wrong predictions and display them in a grid.
    Title of each subplot: 'T:{true}  P:{pred}'
    """
    wrong_images = []
    wrong_true   = []
    wrong_pred   = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images_gpu = images.to(device)
            outputs  = model(images_gpu)
            _, preds = outputs.max(1)

            # Compare on CPU
            preds_cpu = preds.cpu()
            mask      = preds_cpu.ne(labels)

            wrong_images.append(images[mask])       # keep on CPU for plotting
            wrong_true.append(labels[mask])
            wrong_pred.append(preds_cpu[mask])

            if sum(len(w) for w in wrong_images) >= n:
                break

    wrong_images = torch.cat(wrong_images)[:n]
    wrong_true   = torch.cat(wrong_true)[:n]
    wrong_pred   = torch.cat(wrong_pred)[:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i, ax in enumerate(axes.flat):
        if i < len(wrong_images):
            img = denormalize(wrong_images[i]).permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"T:{class_names[wrong_true[i].item()]}\nP:{class_names[wrong_pred[i].item()]}",
                         fontsize=8, color='red')
        ax.axis('off')

    plt.suptitle(f'Wrong Predictions ({len(wrong_images)} total)', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()
