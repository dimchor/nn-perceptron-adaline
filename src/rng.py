import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

def generate_random_values():
    rng = np.random.default_rng()
    size = 10000

    # example of random number generation using a discrete distribution
    n = 1000
    p = .5
    values = rng.binomial(n, p, size)
    bins = np.arange(min(values), max(values))
    plt.title("Binomial distribution")
    plt.xlabel("number")
    plt.ylabel("probability")
    plt.hist(values, bins, color='g')
    plt.show()

    # example of random number generation using a continuous distribution
    mu = 500
    sig = 200
    values = rng.normal(mu, sig, size)
    bins = np.arange(min(values), max(values))
    plt.title("Normal distribution")
    plt.xlabel("number")
    plt.ylabel("probability")
    plt.hist(values, bins, color='r')
    plt.show()
