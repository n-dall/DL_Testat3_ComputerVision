import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import Data

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze
            nn.Flatten(),             # Shape: (B, C)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x).unsqueeze(2).unsqueeze(3)  # Shape: (B, C, 1, 1)
        return x * scale



class CNN(nn.Module):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32×32×32
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64×32×32
        self.bn2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)  # 64×16×16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128×16×16
        self.bn3 = nn.BatchNorm2d(128)

        self.se1 = SEBlock(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 256×16×16
        self.bn4 = nn.BatchNorm2d(256)

        self.pool2 = nn.MaxPool2d(2, 2)  # 256×8×8

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 256×8×8
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 128×8×8
        self.bn6 = nn.BatchNorm2d(128)

        self.se2 = SEBlock(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 128×8×8
        self.bn7 = nn.BatchNorm2d(128)


        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 128×8×8
        self.bn8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 128×8×8
        self.bn9 = nn.BatchNorm2d(128)

        self.se3 = SEBlock(128)

        self.pool3 = nn.MaxPool2d(2, 2)  # 128×4×4

        self.dropout = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se1(x)
        x = self.bn4(self.conv4(x))
        x_res1 = self.pool2(x)

        x = self.bn5(self.conv5(x_res1))
        x = F.relu(x + x_res1)
        x = self.bn6(self.conv6(x))
        x_res2 = self.se2(x)

        x = self.bn7(self.conv7(x_res2))
        x = F.relu(x + x_res2)

        x_res3 = self.bn8(self.conv8(x))
        x = self.bn9(self.conv9(x_res3))
        x = self.se3(x)
        x = F.relu(x + x_res3)
        x = self.pool3(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = F.relu(self.dropout(self.fc3(x)))
        x = self.fc4(x)

        return x

    def fit(self, train_loader, val_loader, device, epochs=10, lr=0.001):
        """
        Train the model on the given Train Dataset
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        self.to(device)
        train_accuracy = []
        train_loss = []
        validation_accuracy = []
        validation_loss = []

        for epoch in range(epochs):
            self.train()

            train_acc, trainloss = self.evaluate(train_loader, device, optimizer, train=True)
            train_accuracy.append(train_acc)
            train_loss.append(trainloss)

            val_acc, val_loss = self.evaluate(val_loader, device, None, train=False)
            validation_accuracy.append(val_acc)
            validation_loss.append(val_loss)

            print(f"Epoch {epoch + 1}, Training Loss: {trainloss:.4f}, Validation Loss: {val_loss:.4f}")

        return train_accuracy, train_loss, validation_accuracy, validation_loss

    def evaluate(self, loader, device, optimizer, train=True):
        """
        Calculate the model's accuracy & loss on the given dataset.
        If train is True, this also updates the weights with the supplied optimizer.
        """
        context = nullcontext() if train else torch.no_grad()
        if not train:
            self.eval()

        correct = 0
        total = 0
        running_loss = 0.0

        with context:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        accuracy = 100 * correct / total
        loss = running_loss / len(loader)
        return accuracy, loss