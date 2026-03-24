import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SportsClassifier(nn.Module):
    """
    ResNet18-based classifier for 100 sports classes.
    Pretrained on ImageNet, with the final FC layer replaced.

    ResNet18 pipeline:
        Input (N, 3, 224, 224)
        → Conv1 + BN + ReLU + MaxPool  → (N, 64, 56, 56)
        → Layer1: BasicBlock × 2       → (N, 64, 56, 56)
        → Layer2: BasicBlock × 2       → (N, 128, 28, 28)
        → Layer3: BasicBlock × 2       → (N, 256, 14, 14)
        → Layer4: BasicBlock × 2       → (N, 512, 7, 7)
        → AdaptiveAvgPool              → (N, 512, 1, 1)
        → FC: 512 → 100               ← replaced for our task
    """
    def __init__(self, num_classes=100, freeze_backbone=True):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace the final FC layer: 512 → num_classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":
    model = SportsClassifier(num_classes=100, freeze_backbone=True)
    print(model)
    dummy = torch.randn(1, 3, 224, 224)
    print("Output shape:", model(dummy).shape)  # should be [1, 100]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
