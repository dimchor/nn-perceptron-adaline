import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import scipy.io as sio
import matplotlib.pyplot as plt

import constants

def mean_median_mode(values):
    return np.mean(values), np.median(values), sp.mode(values)

def variance_stddev(values):
    return np.var(values), np.std(values)

def demo_cities():
    data = sio.loadmat(constants.DATASET_ROOT + 'cities.mat')
    names = data['names']
    crime = [city[3] for city in data['ratings']]

    mean, median, mode = mean_median_mode(crime)
    print("(arithmetic) mean:", mean)
    print("median:", median)
    print("mode:", mode.mode)
    variance, stddev = variance_stddev(crime)
    print("variance:", variance)
    print("standard deviation:", stddev)

    plt.title("Crime rate in US cities")
    plt.xlabel("city")
    plt.xticks(rotation=45)
    plt.ylabel("crime rate")
    plt.bar(names, crime, width=1.0, color='c')
    plt.tight_layout()
    plt.show()

# demo
def demo_generate_random_values():
    rng = np.random.default_rng()
    size = 10000

    # example of random number generation using a discrete distribution
    n = 1000
    p = .5
    values = rng.binomial(n, p, size)
    mean, median, mode = mean_median_mode(values)
    print("(arithmetic) mean:", mean)
    print("median:", median)
    print("mode:", mode.mode)
    variance, stddev = variance_stddev(values)
    print("variance:", variance)
    print("standard deviation:", stddev)

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
    mean, median, mode = mean_median_mode(values)
    print("(arithmetic) mean:", mean)
    print("median:", median)
    print("mode:", mode.mode)
    variance, stddev = variance_stddev(values)
    print("variance:", variance)
    print("standard deviation:", stddev)

    bins = np.arange(min(values), max(values))
    plt.title("Normal distribution")
    plt.xlabel("number")
    plt.ylabel("probability")
    plt.hist(values, bins, color='r')
    plt.show()
