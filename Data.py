import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


def load_data():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define transforms (normalization helps convergence)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-10 training and test datasets
    train_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split into training (45k) and validation (5k)
    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
