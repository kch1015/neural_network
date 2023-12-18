import time

import numpy as np
import matplotlib.pyplot as plt

from project.datasets.MNIST.mnist import load_mnist
from utils.sampler import Sampler
from nn import NN
from layers.cross_entropy_loss import CrossEntropyLoss
from optim import Adam
from utils.backward import Backward


if __name__ == "__main__":
    lr = 0.0002
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

    train_accuracy_list = []
    test_accuracy_list = []

    start = time.time()

    for epoch in range(epoch):
        print("[Epoch: {:>4}] {:0.5g} %".format(epoch + 1, 0.), end="")

        sum_cost = 0
        train_correct = 0
        train_complete = 0
        train_total = train_x.shape[0]

        for i, data in enumerate(train_sampler):
            batch_x, batch_y = data

            optimizer.zero_grad()

            hypothesis = model(batch_x)
            cost = criterion(hypothesis, batch_y)

            backward = Backward(model)
            backward.backward(criterion.backward())

            optimizer.step()

            sum_cost += cost.item()
            predicted = np.argmax(hypothesis, axis=1)
            train_complete += batch_y.shape[0]
            train_correct += np.sum(predicted == np.argmax(batch_y, axis=1)).item()

            print("\r[Epoch: {:>4}] {:0.5g} %".format(epoch + 1, 100 * train_complete / train_total), end="")

        accuracy = 100 * train_correct / train_total
        train_accuracy_list.append(accuracy)
        print("\r[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, sum_cost / train_total))
        print("Train Accuracy: {0:.3f} %".format(accuracy))

        test_num = np.zeros(label_size)
        test_correct = np.zeros(label_size)
        test_complete = 0
        test_total = test_x.shape[0]

        print("Test Progress: {:0.5g} %".format(0.), end="")
        for data in test_sampler:
            batch_x, batch_y = data

            outputs = model(batch_x)

            predicted = np.argmax(outputs, 1)
            test_complete += batch_y.shape[0]

            is_correct = np.expand_dims(predicted == np.argmax(batch_y, axis=1), axis=1)
            test_num += np.sum(batch_y, axis=0)
            test_correct += np.sum(is_correct * batch_y, axis=0)

            print("\rTest Progress: {:0.5g} %".format(100 * test_complete / test_total), end="")

        test_accuracy = 100 * np.sum(test_correct) / test_total
        test_accuracy_list.append(test_accuracy)
        print("\rTest Accuracy: {0:.3f} %".format(test_accuracy))

        test_class_accuracy = 100 * test_correct / test_num
        for i, class_accuracy in enumerate(test_class_accuracy):
            print("class {0:d}: {1:.3f} %".format(i, class_accuracy))

    end = time.time()
    print("총 학습 시간 : {}".format(end - start))

    plt.plot(test_accuracy_list, '-r', label="Test Accuracy")
    plt.plot(train_accuracy_list, '-b', label="Training Accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.text(1, min(min(test_accuracy_list), min(train_accuracy_list)) + 1,
             "isTorch = True\n" +
             "lr = {0:f}".format(lr))
    plt.show()
