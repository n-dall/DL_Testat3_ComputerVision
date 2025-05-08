import torch
from FCNN import FCNN
from CNN import CNN
from Data import load_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Save to file
    train_loader, val_loader, test_loader = load_data()


    # TODO: Optimize fit() and save models to file
    # TODO: Graph accuracies over epochs
    # Fully Connected NN
    fcnn = FCNN()
    fcnn.fit(train_loader, val_loader, device, epochs=50)
    print("Final test accuracy of fully connected neural network: " + fcnn.evaluate(test_loader, device))

    # CNN
    cnn = CNN()
    cnn.fit(train_loader, val_loader, device, epochs=50)
    print("Final test accuracy of convolutional neural network: " + cnn.evaluate(test_loader, device))


if __name__ == "__main__":
    main()