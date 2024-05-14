import numpy as np

class Perceptron:
    def __init__(self, n, beta, epochs):
        # size of inputs
        self.__n = n

        # learning rate
        self.__beta = beta

        self.__epochs = epochs

        # weights vector
        self.__weights = np.random.rand(self.__n, 1)
        
    @staticmethod
    def __step(x: np.array):
        return np.array([1 if item >=0 else 0 for item in x])

    def train(self, x: np.array, d: np.array):
        err = np.zeros(self.__epochs, dtype=int)
        for i in range(self.__epochs):
            for j in range(len(x)):
                y = self.__step(np.dot(x[j], self.__weights))
                if y[0] != d[j][0]:
                    err[i] += 1
                    # delta rule
                    self.__weights += self.__beta * (d[j][0] - y[0]) * x[j].reshape(-1, 1)
            if err[i] == 0:
                break
        return err

    def weights(self):
        return self.__weights

    def test(self, x: np.array, d: np.array):
        err = 0
        for i in range(len(x)):
            if d[i][0] != self.__step(np.dot(x[i], self.__weights)):
                err += 1
        return err
