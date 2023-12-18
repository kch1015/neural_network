from project.datasets.MNIST.mnist import load_mnist
from project.numpy.utils.sampler import Sampler


if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = load_mnist(one_hot_label=False)

    train_sampler = Sampler(train_x, train_y, batch_size=32, shuffle=True)
    test_sampler = Sampler(test_x, test_y, batch_size=32, shuffle=False)

    count = 0
    for i, data in enumerate(test_sampler):
        batch_x, batch_y = data

        print(batch_x.shape)
        print(batch_y)
        count += batch_y.shape[0]
        print(count)




