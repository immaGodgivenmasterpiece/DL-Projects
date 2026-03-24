"""
Train v2: Fine-Tuning — unfreeze all layers, train end-to-end with small LR.

Loads the best model from v1 (feature extraction) as starting point.
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
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "v2_finetune")
os.makedirs(RESULTS_DIR, exist_ok=True)

V1_MODEL_PATH   = os.path.join(PROJECT_DIR, "results", "v1_feature_extraction", "best_model.pth")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")


# ── Training Loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a .pth checkpoint to resume from (default: v1 best model)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Data — proper 3-split (train/val/test)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=64)

    # Model — start frozen, then load v1 weights and unfreeze
    model = SportsClassifier(num_classes=100, freeze_backbone=True).to(device)

    checkpoint = args.resume_from or V1_MODEL_PATH
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"Loaded checkpoint: {checkpoint}")

    # Unfreeze all layers for fine-tuning
    for param in model.resnet.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (all layers unfrozen)")
    print(f"Model is on: {next(model.parameters()).device}")

    # Loss & Optimizer — small LR to avoid destroying pretrained knowledge
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
    print(f"Fine-Tuned Test Acc: {test_acc:.2f}%")

    # Visualize
    plot_history(train_losses, val_losses, train_accs, val_accs,
                 save_path=os.path.join(RESULTS_DIR, "loss_acc_curves.png"))
    plot_confusion_matrix(model, test_loader, device, class_names,
                          save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_wrong_predictions(model, test_loader, device, class_names, n=16,
                           save_path=os.path.join(RESULTS_DIR, "wrong_predictions.png"))


if __name__ == "__main__":
    main()
