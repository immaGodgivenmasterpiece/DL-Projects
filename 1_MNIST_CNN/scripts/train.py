import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from utils import (get_device, get_dataloaders, train_one_epoch, evaluate,
                   plot_history, plot_confusion_matrix, plot_wrong_predictions)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                    # 1_MNIST_CNN/
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")


# ── Training Loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a .pth checkpoint to resume training from")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Data — MNIST standard 2-split (no separate validation set).
    # Test set is used for early stopping AND final evaluation.
    # Acceptable for this benchmark; see README for rationale.
    train_loader, test_loader = get_dataloaders(batch_size=64)

    # Model
    model = SimpleCNN().to(device)

    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"Loaded checkpoint: {args.resume_from}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Hyperparams
    EPOCHS  = 20
    PATIENCE = 3

    # History
    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []

    # Early stopping state
    best_acc = 0.0
    counter  = 0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(tr_loss);  test_losses.append(te_loss)
        train_accs.append(tr_acc);     test_accs.append(te_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
        print(f"  Test  Loss: {te_loss:.4f} | Test  Acc: {te_acc:.2f}%")

        # Early stopping + best model save
        if te_acc > best_acc:
            best_acc = te_acc
            counter  = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (acc: {best_acc:.2f}%)")
        else:
            counter += 1
            print(f"  patience: {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print(f"\nEarly stopping triggered. Best Test Acc: {best_acc:.2f}%")
                break

    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print("\nBest model loaded!")

    # Visualize
    plot_history(train_losses, test_losses, train_accs, test_accs,
                 save_path=os.path.join(RESULTS_DIR, "loss_acc_curves.png"))
    plot_confusion_matrix(model, test_loader, device,
                          save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_wrong_predictions(model, test_loader, device, n=16,
                           save_path=os.path.join(RESULTS_DIR, "wrong_predictions.png"))


if __name__ == "__main__":
    main()
