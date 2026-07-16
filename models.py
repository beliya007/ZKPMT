"""
MLP model definitions for zkPMT experiments.
Three model scales: MLP-Small, MLP-Medium, MLP-Large
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Piecewise-linear approximation of exp(x) used in zkPMT ──────────────────
def approx_exp(x):
    """First-order Taylor approximation of e^x clipped to [0,1]."""
    return torch.clamp(0.5 + x, 0.0, 1.0)


def approx_softmax(logits):
    """Softmax using piecewise-linear exp approximation."""
    exp_vals = approx_exp(logits)
    return exp_vals / (exp_vals.sum(dim=-1, keepdim=True) + 1e-8)


# ── Standard MLP (used by all baselines) ────────────────────────────────────
class MLPSmall(nn.Module):
    """2 FC layers, hidden=128, ~10K params. Suitable for Iris."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MLPMedium(nn.Module):
    """3 FC layers, hidden=256/128, ~50K params. Suitable for MNIST."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPLarge(nn.Module):
    """5 FC layers, hidden=512/256/128/64, ~200K params. Suitable for CIFAR-10."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# ── Approximated MLP used by zkPMT (piecewise-linear activations) ────────────
class MLPSmallApprox(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))          # ReLU is exact piecewise-linear
        return self.fc2(x)               # output logits; softmax approximated externally


class MLPMediumApprox(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPLargeApprox(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# ── ResNet50 (Extra scale) ──────────────────────────────────────────────────
class ResNet50Model(nn.Module):
    """Standard ResNet50 adapted for CIFAR-10 (32x32 input)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        # Adapt for 32x32: change conv1 and remove maxpool
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Flattened input (B, 3072) -> (B, 3, 32, 32)
        if x.dim() == 2 and x.size(1) == 3072:
            x = x.view(-1, 3, 32, 32)
        return self.resnet(x)


class ResNet50Approx(nn.Module):
    """Approximated ResNet50 (using ReLU activations)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        if x.dim() == 2 and x.size(1) == 3072:
            x = x.view(-1, 3, 32, 32)
        return self.resnet(x)


MODEL_CONFIGS = {
    # (model_class_standard, model_class_approx, input_dim, num_classes, dataset_name)
    "Small+Iris":     (MLPSmall,  MLPSmallApprox,  4,   3,  "Iris"),
    "Medium+MNIST":   (MLPMedium, MLPMediumApprox, 784, 10, "MNIST"),
    "Large+CIFAR-10": (MLPLarge,  MLPLargeApprox,  3072, 10, "CIFAR-10"),
    "ResNet50+CIFAR-10": (ResNet50Model, ResNet50Approx, 3072, 10, "CIFAR-10"),
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())
