import time

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from nn import NN
from function import one_hot_encode


if __name__ == "__main__":
    lr = 0.0001
    epoch = 10
    batch_size = 32

    train_data = datasets.MNIST(root="../datasets",
                                train=True,
                                download=True,
                                transform=transforms.ToTensor(),
                                target_transform=one_hot_encode)
    test_data = datasets.MNIST(root="../datasets",
                               train=False,
                               download=True,
                               transform=transforms.ToTensor(),
                               target_transform=one_hot_encode)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)

    input_size = 28 * 28
    hidden_size = input_size // 2
    label_size = 10

    model = NN(input_size, hidden_size, label_size)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)

    train_accuracy_list = []
    test_accuracy_list = []

    start = time.time()

    for epoch in range(epoch):
        print("[Epoch: {:>4}] {:0.5g} %".format(epoch + 1, 0.), end="")

        sum_cost = 0
        train_correct = 0
        train_complete = 0
        train_total = train_data.__len__()

        model.train()
        for i, data in enumerate(train_loader):
            train_x, train_y = data

            optimizer.zero_grad()

            hypothesis = model(train_x)
            cost = criterion(hypothesis, train_y)
            cost.backward()

            optimizer.step()

            sum_cost += cost.item()
            _, predicted = torch.max(hypothesis.data, 1)
            train_complete += train_y.shape[0]
            train_correct += (predicted == torch.argmax(train_y, dim=1)).sum().item()

            print("\r[Epoch: {:>4}] {:0.5g} %".format(epoch + 1, 100 * train_complete / train_total), end="")

        accuracy = 100 * train_correct / train_total
        train_accuracy_list.append(accuracy)
        print("\r[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, sum_cost / train_total))
        print("Train Accuracy: {0:.3f} %".format(accuracy))

        test_num = torch.zeros(label_size)
        test_correct = torch.zeros(label_size)
        test_complete = 0
        test_total = test_data.__len__()

        model.eval()
        with torch.no_grad():
            print("Test Progress: {:0.5g} %".format(0.), end="")
            for data in test_loader:
                test_x, test_y = data

                outputs = model(test_x)

                _, predicted = torch.max(outputs.data, 1)
                test_complete += test_y.shape[0]

                is_correct = torch.unsqueeze(predicted == torch.argmax(test_y, dim=1), dim=1)
                test_num += test_y.sum(dim=0)
                test_correct += (is_correct * test_y).sum(dim=0)

                print("\rTest Progress: {:0.5g} %".format(100 * test_complete / test_total), end="")

        test_accuracy = 100 * torch.sum(test_correct) / test_total
        test_accuracy_list.append(test_accuracy)
        print("\rTest Accuracy: {0:.3f} %".format(test_accuracy))

        test_class_accuracy = 100 * test_correct / test_num
        for i, class_accuracy in enumerate(test_class_accuracy):
            print("class {0:d}: {1:.3f} %".format(i, class_accuracy))

    end = time.time()
    print("총 학습 시간 : {0:f}".format(end - start))

    plt.plot(test_accuracy_list, '-r', label="Test Accuracy")
    plt.plot(train_accuracy_list, '-b', label="Training Accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.text(1, min(min(test_accuracy_list), min(train_accuracy_list)) + 1,
             "Pytorch\n" +
             "lr = {0:f}".format(lr))
    plt.show()
