# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 10:15:58 2019

@author: bimta
"""
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

X, y, coef = make_regression(n_samples=100, n_features=1, noise=10, coef = True)
#Include bias
offset = np.random.randint(-50, 50)
y += offset
y = y.reshape(len(y),1)
# plot regression dataset
plt.scatter(X,y)



class NeuralNetwork:
    def __init__(self, x, y, l=0.1):
        self.input  = x
        self.weight = np.random.rand() 
        self.bias   = np.random.rand()                 
        self.y      = y
        self.output = np.zeros(y.shape)
        self.learning_rate = l
    
    def feedforward(self):
        #Equal to predicting
        self.output = np.dot(self.input.T, self.weight) + self.bias
        
        # Loss function
        self.J = (np.sum((self.y - self.output.T) ** 2)) / len(self.input)
    
    def backprop(self):
        # Then we want the partial derivative regarding weight and bias of this loss 
        weight_deriv = (np.sum(-2 * self.input * (self.y - self.output.T))) / len(self.input) 
        bias_deriv = (np.sum(-2 * (self.y - self.output.T))) / len(self.input)
        
        # update the weights with the derivative (slope) of the loss function
        self.weight -= weight_deriv * self.learning_rate
        self.bias -= bias_deriv * self.learning_rate


nn = NeuralNetwork(X,y)
print(f'Weight: {nn.weight} Bias: {nn.bias}')

t_weights = [nn.weight]
t_biases = [nn.bias]
t_losses = []
for i in range(200):    
    nn.feedforward()
    nn.backprop()
    print(f'Weight: {nn.weight} Bias: {nn.bias} Cost: {nn.J}')
    t_weights.append(nn.weight)
    t_biases.append(nn.bias)
    t_losses.append(nn.J)


y_pred = nn.output.T
t_losses.append(np.sum((y - y_pred)**2) / len(y))

plt.scatter(X,y)
plt.scatter(X,y_pred)
plt.show()


###############################################################################
# Plot Loss surface:

def loss_plot():
    weights = np.linspace(coef - 1.5 * coef, coef + 1.5 * coef, 100)
    biases = np.linspace(offset - 1.5 * offset, offset + 1.5 * offset , 100)
    ww, bb = np.meshgrid(weights, biases)
    
    losses = []
    for bias in biases:
        for weight in weights:
            pred = np.dot(X.T, weight).T + bias
            loss = (np.sum((y - pred) ** 2)) / len(X)
            losses.append(loss)
    
    losses = np.array(losses).reshape(100, 100)
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(ww, bb, losses, linewidth=0, antialiased=False, cmap = cm.coolwarm, alpha = 0.5)
    ax.scatter(t_weights, t_biases, t_losses, c = 'r')
    plt.show()
    return

loss_plot()