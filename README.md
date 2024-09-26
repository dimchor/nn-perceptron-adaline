# Perceptron and Adaline neurons
This is a university assignment for the Neural Networks course. 

## Neurons

The Iris dataset is used in this project. The dataset is split into two smaller ones, specifically: the train set and the test set. The train set is used to train the neuron and adjust its weights appropriately. Then, the test set is used to verify that the weights have been set correctly.

### Perceptron

`perceptron.py` contains the Perceptron class. The step function is used as the activator function. The step function decides the appropriate category for each item.

### Adaline

`adaline.py` contains the Adaline class. The Adaline class works in a similar fashion as the Perceptron class. Their main differences lie in the way the training is done. Adaline uses the linear function instead of the step function as the activator function and the training stops once the mean square error falls below the minimum mean square error, which is set beforehand. 

### Error rate
The following figure shows the falling error rate of the aforementioned neurons on the same dataset:
![fig](https://github.com/user-attachments/assets/0f818514-e023-4cbd-a48b-5290b70fd039)

_The blue and orange lines represent the error rate of the Perceptron and Adaline neurons respectively_

### Learning rate

Choosing the proper learning rate (shown as _beta_ in source code) can be quite challenging, as there's no clear-cut answer. Setting the learning rate too low may result in significant performance degradation, due to the weight adjustments being mininal or almost negligible. On the other hand, setting the learning rate too high may result in failure to adjust the weights properly, leading to a high error rate. This was particularly the case when working with Adaline (_considering the dataset used and the implementation_). The chosen learning rate seems to be sufficient in delivering quality results in a reasonable amount of time.

## Statistics

`stats.py` contains some demo functions that visualise datasets, specifically: two of which contain real world data from USA cities and hospitals and the other two merely contain random values.
