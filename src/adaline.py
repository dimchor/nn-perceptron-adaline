import numpy as np

class Adaline:
    def __init__(self, n, beta, epochs, mmse):
        # size of inputs
        self.__n = n

        # learning rate
        self.__beta = beta

        self.__epochs = epochs

        # mininum mean squared error
        self.__mmse = mmse

        # weights vector
        self.__weights = np.random.rand(self.__n, 1)
        
    @staticmethod
    def __linear(x: np.array):
        return x

    def train(self, x: np.array, d: np.array):
        mse = np.zeros(self.__epochs, dtype=float)
        for i in range(self.__epochs):
            err = 0.
            for j in range(len(x)):
                y = self.__linear(np.dot(x[j], self.__weights))

                # delta rule
                delta = d[j][0] - y[0]

                err += delta ** 2

                self.__weights += self.__beta * delta * x[j].reshape(-1, 1)

            mse[i] = err / len(x)

            if err <= self.__mmse:
                break

        return mse


    def weights(self):
        return self.__weights

    def test(self, x: np.array, d: np.array):
        err = 0
        for i in range(len(x)):
            if abs(d[i][0] - self.__linear(np.dot(x[i], self.__weights))) > self.__mmse:
                err += 1
        return err
