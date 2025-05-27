import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from FCNN import FCNN
from CNN import CNN
from Data import load_data
matplotlib.use('TkAgg')


def accuracy_loss_plot(train_accuracy, val_accuracy, train_loss, val_loss):
    epochs = range(1, len(train_accuracy) + 1)

    plt.figure(figsize=(12, 6))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracies')
    plt.legend()
    plt.grid(True)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data()

    # TODO: Make both models a bit more complex and retrain

    # Fully Connected NN
    fcnn = FCNN()
    fcnn_train_acc, fcnn_train_loss, fcnn_val_acc, fcnn_val_loss = [], [], [], []
    if os.path.exists("fcnn.pth"):
        print("Loading saved fully connected neural network from file")
        fcnn.load_state_dict(torch.load("fcnn.pth", weights_only=True))
    else:
        print("Training fully connected neural network...")
        fcnn_train_acc, fcnn_train_loss, fcnn_val_acc, fcnn_val_loss = fcnn.fit(train_loader, val_loader, device, epochs=50)
        torch.save(fcnn.state_dict(), "fcnn.pth")

    fcnn_test_acc, fcnn_test_loss = fcnn.evaluate(test_loader, device, None, train=False)
    print("Final test accuracy of FCNN: " + str(fcnn_test_acc) + "%, Final test loss of FCNN: " + str(fcnn_test_loss))
    accuracy_loss_plot(fcnn_train_acc, fcnn_val_acc, fcnn_train_loss, fcnn_val_loss)

    # CNN
    cnn = CNN()
    cnn_train_acc, cnn_train_loss, cnn_val_acc, cnn_val_loss = [], [], [], []
    if os.path.exists("cnn.pth"):
        print("\nLoading saved convolutional neural network from file")
        cnn.load_state_dict(torch.load("cnn.pth", weights_only=True))
    else:
        print("\nTraining convolutional neural network...")
        cnn_train_acc, cnn_train_loss, cnn_val_acc, cnn_val_loss = cnn.fit(train_loader, val_loader, device, epochs=60)
        torch.save(cnn.state_dict(), "cnn.pth")

    cnn_test_acc, cnn_test_loss = cnn.evaluate(test_loader, device, None, train=False)
    print("Final test accuracy of CNN: " + str(cnn_test_acc) + "%, Final test loss of CNN: " + str(cnn_test_loss))
    accuracy_loss_plot(cnn_train_acc, cnn_val_acc, cnn_train_loss, cnn_val_loss)


if __name__ == "__main__":
    main()