import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from adaline import Adaline
import stats
import constants

def part_a():
    stats.demo_generate_random_values()

    stats.demo_cities()

    stats.demo_hospital()

def part_b():
    SETOSA = 0
    VERSICOLOR = 1
    TRAINSET_SIZE = 50

    data = pd.read_csv(constants.DATASET_ROOT + 'iris.csv', sep=',', header=0).sample(frac=1)
    
    transform = np.vectorize(lambda s: SETOSA if s == 'Iris-setosa' else VERSICOLOR)

    train_d = transform(data[:TRAINSET_SIZE][['Species']].to_numpy())
    train_x = data[:TRAINSET_SIZE][['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()

    test_d = transform(data[TRAINSET_SIZE:][['Species']].to_numpy())
    test_x = data[TRAINSET_SIZE:][['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()

    p = Perceptron(4, 1, constants.EPOCHS)

    err = p.train(train_x, train_d)

    plt.xlabel('epochs')
    plt.ylabel('errors')
    plt.plot(range(constants.EPOCHS), err)
    plt.show()

    print(p.test(test_x, test_d))

    # a = Adaline(4, 1, constants.EPOCHS, 100)
    # print(a.train(train_x, train_d))
    # print(a.test(test_x, test_d))


def main():
    pass

    # part_a()

    part_b()


if __name__ == "__main__":
    main()
