"""
Dataset loading utilities.
Returns (X_train, y_train, X_test, y_test) as PyTorch tensors,
flattened and normalized, ready for MLP input.
"""

import torch
import numpy as np


def load_iris():
    """Iris: 150 samples, 4 features, 3 classes."""
    from sklearn.datasets import load_iris as _load
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = _load()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return (torch.tensor(X_tr), torch.tensor(y_tr),
            torch.tensor(X_te), torch.tensor(y_te))


def load_mnist():
    """MNIST: 60K train, 10K test, 28×28 grayscale → 784-dim."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_ds = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform)
        X_tr = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
        y_tr = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
        X_te = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
        y_te = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
        return X_tr, y_tr, X_te, y_te
    except Exception:
        # Fallback: synthetic data with same dimensions
        print("[WARN] MNIST download failed, using synthetic data.")
        return _synthetic(60000, 784, 10)


def load_cifar10():
    """CIFAR-10: 50K train, 10K test, 32×32×3 → 3072-dim."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_ds = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform)
        X_tr = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
        y_tr = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
        X_te = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
        y_te = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
        return X_tr, y_tr, X_te, y_te
    except Exception:
        print("[WARN] CIFAR-10 download failed, using synthetic data.")
        return _synthetic(50000, 3072, 10)


def _synthetic(n_train, input_dim, num_classes):
    X_tr = torch.randn(n_train, input_dim)
    y_tr = torch.randint(0, num_classes, (n_train,))
    X_te = torch.randn(1000, input_dim)
    y_te = torch.randint(0, num_classes, (1000,))
    return X_tr, y_tr, X_te, y_te


LOADERS = {
    "Iris":     load_iris,
    "MNIST":    load_mnist,
    "CIFAR-10": load_cifar10,
}
