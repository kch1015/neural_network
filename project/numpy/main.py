import time

import numpy as np
import matplotlib.pyplot as plt

from project.datasets.MNIST.mnist import load_mnist
from utils.sampler import Sampler
from nn import NN
from layers.cross_entropy_loss import CrossEntropyLoss
from optim import *
from utils.backward import Backward


def train(model, criterion, optimizer, sampler, cost_list, accuracy_list):
    print("[Epoch: {0:>4d}] {1:0.5g} %".format(epoch + 1, 0.), end="")

    cost_sum = 0
    train_correct = 0
    train_complete = 0
    train_total = train_x.shape[0]

    model.train()
    for i, data in enumerate(sampler):
        batch_x, batch_y = data

        hypothesis = model(batch_x)
        cost = criterion(hypothesis, batch_y)

        backward = Backward(model)
        backward.backward(criterion.backward())

        optimizer.step()

        cost_sum += cost.item()
        train_complete += batch_y.shape[0]
        train_correct += np.sum(np.equal(np.argmax(hypothesis, axis=1), np.argmax(batch_y, axis=1))).item()

        print("\r[Epoch: {0:>4d}] {1:0.5g} %".format(epoch + 1, 100 * train_complete / train_total), end="")

    cost = cost_sum / train_total
    accuracy = 100 * train_correct / train_total

    cost_list.append(cost)
    accuracy_list.append(accuracy)

    print("\r[Epoch: {0:>4d}]".format(epoch + 1))
    print("Train Cost = {0:>.9f}".format(cost))
    print("Train Accuracy = {0:.3f} %".format(accuracy))


def test(model, criterion, sampler, cost_list, accuracy_list):
    print("Test Progress: {0:0.5g} %".format(0.), end="")

    cost_sum = 0
    test_num = np.zeros(label_size)
    test_correct = np.zeros(label_size)
    test_complete = 0
    test_total = test_x.shape[0]

    model.eval()
    for data in sampler:
        batch_x, batch_y = data

        outputs = model(batch_x)
        cost = criterion(outputs, batch_y)

        cost_sum += cost.item()
        test_complete += batch_y.shape[0]

        is_correct = np.expand_dims(np.equal(np.argmax(outputs, axis=1), np.argmax(batch_y, axis=1)), axis=1)
        test_num += np.sum(batch_y, axis=0)
        test_correct += np.sum(is_correct * batch_y, axis=0)

        print("\rTest Progress: {0:0.5g} %".format(100 * test_complete / test_total), end="")
    
    cost = cost_sum / test_total
    total_accuracy = 100 * np.sum(test_correct) / test_total
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
             "Numpy, {0:s}\n".format(type(optimizer).__name__) +
             # "He = True, batch_norm = True\n" +
             "hidden_size = {0:d}, lr = {1:f}".format(hidden_size, lr))
    plt.show()


if __name__ == "__main__":
    lr = 0.0001
    epoch = 10
    batch_size = 32
    
    (train_x, train_y), (test_x, test_y) = load_mnist(one_hot_label=True)

    train_sampler = Sampler(train_x, train_y, batch_size, shuffle=True)
    test_sampler = Sampler(test_x, test_y, batch_size, shuffle=False)

    input_size = train_x.shape[1]
    hidden_size = input_size // 2
    label_size = train_y.shape[1]

    model = NN(input_size, hidden_size, label_size)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)
    
    train_cost_list = []
    test_cost_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    start = time.time()

    for epoch in range(epoch):
        train(model, criterion, optimizer, train_sampler, train_cost_list, train_accuracy_list)
        test(model, criterion, test_sampler, test_cost_list, test_accuracy_list)

    end = time.time()
    print("총 학습 시간 : {0:f}".format(end - start))

    show_graph(train_cost_list, test_cost_list, "Cost")
    show_graph(train_accuracy_list, test_accuracy_list, "Accuracy")
