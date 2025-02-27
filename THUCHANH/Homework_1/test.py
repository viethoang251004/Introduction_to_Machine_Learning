import numpy as np
import matplotlib.pyplot as plt

def f(x, w):
    return w[0] + w[1] * x

def loss(x, y, w):
    d = 0
    for i in range(len(x)):
        d += (y[i] - (w[0] + w[1] * x[i]))**2
    return d / (2 * len(x))

def derivative(x, y, w):
    d0 = 0
    d1 = 0
    for i in range(len(x)):
        d1 += x[i] * (f(x[i], w) - y[i])
        d0 += f(x[i], w) - y[i]
    return d0 / len(x), d1 / len(x)

x = np.linspace(start=1, stop=10, num=50)
y = 2 * x + np.random.normal(0, 1, 50)  # Example linear data with some noise

epoch = 10
learning_rate = 0.01
w = [1, 1]  # Initial guess for the model parameters
los_old = float('inf')  # Initialize as infinity

for i in range(epoch):
    # Plot the data points
    plt.plot(x, y, 'ro')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot the current model
    x0 = np.linspace(start=1, stop=10, num=50)
    y0 = w[0] + w[1] * x0
    plt.plot(x0, y0)