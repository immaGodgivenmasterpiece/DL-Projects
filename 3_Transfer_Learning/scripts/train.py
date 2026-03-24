"""
Train v1: Feature Extraction — freeze ResNet18 backbone, train only the FC head.

Uses proper 3-split: val_loader for early stopping, test_loader for final evaluation only.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import SportsClassifier
from utils import (get_device, get_dataloaders, train_one_epoch, evaluate,
                   plot_history, plot_confusion_matrix, plot_wrong_predictions)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                    # 3_Transfer_Learning/
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "v1_feature_extraction")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")


# ── Training Loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a .pth checkpoint to resume training from")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Data — proper 3-split (train/val/test)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=64)

    # Model — frozen backbone, only FC layer is trainable
    model = SportsClassifier(num_classes=100, freeze_backbone=True).to(device)

    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"Loaded checkpoint: {args.resume_from}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Model is on: {next(model.parameters()).device}")

    # Loss & Optimizer — only FC params (no point passing frozen params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)

    # Hyperparams
    EPOCHS  = 30
    PATIENCE = 5

    # History
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    # Early stopping state
    best_acc = 0.0
    counter  = 0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
        print(f"  Val   Loss: {vl_loss:.4f} | Val   Acc: {vl_acc:.2f}%")

        # Early stopping based on val accuracy
        if vl_acc > best_acc:
            best_acc = vl_acc
            counter  = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (acc: {best_acc:.2f}%)")
        else:
            counter += 1
            print(f"  patience: {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print(f"\nEarly stopping. Best Val Acc: {best_acc:.2f}%")
                break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print("\nBest model loaded!")

    # Final evaluation on test set — used only once
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Acc: {test_acc:.2f}%")

    # Visualize
    plot_history(train_losses, val_losses, train_accs, val_accs,
                 save_path=os.path.join(RESULTS_DIR, "loss_acc_curves.png"))
    plot_confusion_matrix(model, test_loader, device, class_names,
                          save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_wrong_predictions(model, test_loader, device, class_names, n=16,
                           save_path=os.path.join(RESULTS_DIR, "wrong_predictions.png"))


if __name__ == "__main__":
    main()
