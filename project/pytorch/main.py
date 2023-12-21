import time

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from nn import NN
from function import one_hot_encode


def train(model, criterion, optimizer, loader, cost_list, accuracy_list):
    print("[Epoch: {0:>4d}] {1:.5g} %".format(epoch + 1, 0.), end="")

    sum_cost = 0
    train_correct = 0
    train_complete = 0
    train_total = train_data.__len__()

    model.train()
    for i, data in enumerate(loader):
        train_x, train_y = data

        optimizer.zero_grad()

        hypothesis = model(train_x)
        cost = criterion(hypothesis, train_y)
        cost.backward()

        optimizer.step()

        sum_cost += cost.item()
        _, predicted = torch.max(hypothesis.data, 1)
        train_complete += train_y.shape[0]
        train_correct += torch.sum(torch.eq(predicted, torch.argmax(train_y, dim=1))).item()

        print("\r[Epoch: {0:>4d}] {1:.5g} %".format(epoch + 1, 100 * train_complete / train_total), end="")

    cost = sum_cost / train_total
    accuracy = 100 * train_correct / train_total

    cost_list.append(cost)
    accuracy_list.append(accuracy)

    print("\r[Epoch: {0:>4d}]".format(epoch + 1))
    print("Train Cost = {0:>.9f}".format(cost))
    print("Train Accuracy = {0:.3f} %".format(accuracy))


def test(model, criterion, loader, cost_list, accuracy_list):
    target_size = len(loader.dataset.classes)

    cost_sum = 0
    test_num = torch.zeros(target_size)
    test_correct = torch.zeros(target_size)
    test_complete = 0
    test_total = test_data.__len__()

    model.eval()
    with torch.no_grad():
        print("Test Progress: {0:.5g} %".format(0.), end="")
        for data in loader:
            test_x, test_y = data

            outputs = model(test_x)
            cost = criterion(outputs, test_y)
            _, predicted = torch.max(outputs.data, 1)

            cost_sum += cost.item()
            test_complete += test_y.shape[0]

            is_correct = torch.unsqueeze(torch.eq(predicted, torch.argmax(test_y, dim=1)), dim=1)
            test_num += torch.sum(test_y, dim=0)
            test_correct += torch.sum(is_correct * test_y, dim=0)

            print("\rTest Progress: {0:.5g} %".format(100 * test_complete / test_total), end="")

    cost = cost_sum / test_total
    total_accuracy = 100 * torch.sum(test_correct) / test_total
    class_accuracy = 100 * test_correct / test_num

    cost_list.append(cost)
    accuracy_list.append(total_accuracy)

    print("\rTest Cost = {0:>.9f}".format(cost))
    print("Test Accuracy = {0:.3f} %".format(total_accuracy))
    for i, accuracy in enumerate(class_accuracy):
        print("class {0:d}: {1:.3f} %".format(i, accuracy))


def show_graph(train_list, test_list, axis_name):
    plt.plot(train_list, "-b", label="Train")
    plt.plot(test_list, "-r", label="Test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(axis_name)
    plt.text(0, min(min(train_list), min(test_list)),
             "Pytorch, " + type(optimizer).__name__ + "\n" +
             "hidden_size = {0:d}, lr = {1:f}".format(hidden_size, lr))
    plt.show()


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

    input_size = train_data.data.size(1) * train_data.data.size(2)
    hidden_size = input_size // 2
    label_size = len(train_data.classes)

    model = NN(input_size, hidden_size, label_size)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)

    train_cost_list = []
    test_cost_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    start = time.time()

    for epoch in range(epoch):
        train(model, criterion, optimizer, train_loader, train_cost_list, train_accuracy_list)
        test(model, criterion, test_loader, test_cost_list, test_accuracy_list)

    end = time.time()
    print("총 학습 시간 : {0:f}".format(end - start))

    show_graph(train_cost_list, test_cost_list, "Cost")
    show_graph(train_accuracy_list, test_accuracy_list, "Accuracy")
