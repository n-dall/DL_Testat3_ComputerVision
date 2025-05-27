import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

class FCNN(nn.Module):
    criterion = nn.CrossEntropyLoss()
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.fc7(x)

    def fit(self, train_loader, val_loader, device, epochs=10, lr=0.001):
        """
        Train the model on the given Train Dataset
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
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
        loss = running_loss/len(loader)
        return accuracy, loss
