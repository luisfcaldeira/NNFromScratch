import numpy as np
import matplotlib.pyplot as plt


from matplotlib import cm 
from matplotlib.colors import colorConverter, ListedColormap
import seaborn as sns  

# Set seaborn plotting style
sns.set_style('darkgrid')

# Set the seed for reproducibility
np.random.seed(seed=1)

#make data
from sklearn.datasets import make_classification

#make data
X,y = make_classification(n_samples=100, n_features=2, n_informative=2,n_redundant=0)
y = y.reshape(y.shape[0],1)
X.shape, y.shape

plt.figure(figsize=(6, 4))
plt.plot(X[np.where(y==1)[0],0], X[np.where(y==1)[0],1], 'ro', label='class: 1')
plt.plot(X[np.where(y==0)[0],0], X[np.where(y==0)[0],1], 'bo', label='class: 0')
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.axis([-4, 4, -4, 4])
plt.title('Positive vs. Negative class in the input space')
# plt.show()

# Randomly inizialise the weights for the output layer
W = np.random.rand(1,X.shape[1]) 
W.shape


#define logistic function as our activation 
def sigmoid(z):
    return 1. / (1 + np.exp(-z))

output_layer = np.dot(X,W.T)
output_activation = sigmoid(output_layer)

nb_of_xs = X.shape[0]
xsa = np.linspace(-4, 4, num=nb_of_xs)
xsb = np.linspace(-4, 4, num=nb_of_xs)

xx, yy = np.meshgrid(xsa, xsb)

prediction_matrix = np.zeros((nb_of_xs,nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        z = np.dot(np.asmatrix([xx[i,j], yy[i,j]]),W.T)
        prediction_matrix[i,j] = sigmoid(z)

cmap = ListedColormap([
        colorConverter.to_rgba('b', alpha=0.3),
        colorConverter.to_rgba('r', alpha=0.3)])

plt.figure(figsize=(6, 4))

plt.plot(X[np.where(y==1)[0],0], X[np.where(y==1)[0],1], 'ro', label='class: 1')
plt.plot(X[np.where(y==0)[0],0], X[np.where(y==0)[0],1], 'bo', label='class: 0')
plt.contourf(xx, yy, prediction_matrix, 1, cmap=cmap)
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.axis([-4, 4, -4, 4])
plt.title('Positive vs. Negative class in the feature space (Random weights)')
# plt.show()

def cross_entropy(y_hat, y):
    return - np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

def gradient(y_hat,y,X):
    return np.dot((y_hat - y).T,X)

#define parameters for gradient descent
step = 0.05 
n_iter = 15 

losses = []
W_history = np.zeros((n_iter,W.shape[1]))

#iterate
for n in range(n_iter):
    output_layer = np.dot(X,W.T)
    output_activation = sigmoid(output_layer)
    loss = cross_entropy(output_activation, y)
    dW = gradient(output_activation,y,X)
    W -= step * dW
    losses.append(loss)
    W_history[n,:] = W[:]

plt.plot(losses)
plt.xlabel('Number of iterations')
plt.ylabel('Loss')

#calculate loss for all possible values in the parameter space
loss_matrix = np.zeros((W_history.shape[0],W_history.shape[0]))
for i in range(W_history.shape[0]):
    for j in range(W_history.shape[0]):
        W_tmp = np.array([W_history[i],W_history[j]]) 
        loss_matrix[i,j] = cross_entropy(sigmoid(np.dot(X,W_tmp.T)), y)

plt.figure(figsize=(8, 6))
plt.contourf(W_history[:,0], W_history[:,1], loss_matrix, 10, alpha=0.9, cmap=cm.magma)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Loss')

# Plot the updates
for i in range(1,W_history.shape[0]): 
    w1 = W_history[i-1,:]
    w2 = W_history[i,:]
    # Plot the weight-loss values that represents the update
    plt.plot(w1[0], w1[1], 'wo')  # Plot the weight-loss value
    plt.plot([w1[0], w2[0]], [w1[1], w2[1]], 'w-')

# Show figure
plt.xlabel('$w_1$', fontsize=12)
plt.ylabel('$w_2$', fontsize=12)
plt.title('Gradient descent updates on loss surface')

# plt.show()

nb_of_xs = X.shape[0]
xsa = np.linspace(-4, 4, num=nb_of_xs)
xsb = np.linspace(-4, 4, num=nb_of_xs)

xx, yy = np.meshgrid(xsa, xsb)

prediction_matrix = np.zeros((nb_of_xs,nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        prediction_matrix[i,j] = sigmoid(np.asmatrix([xx[i,j], yy[i,j]])).item(0)

cmap = ListedColormap([
        colorConverter.to_rgba('b', alpha=0.3),
        colorConverter.to_rgba('r', alpha=0.3)])

plt.figure(figsize=(6, 4))
plt.plot(X[np.where(y==1)[0],0], X[np.where(y==1)[0],1], 'ro', label='class: 1')
plt.plot(X[np.where(y==0)[0],0], X[np.where(y==0)[0],1], 'bo', label='class: 0')
plt.contourf(xx, yy, prediction_matrix, 1, cmap=cmap)
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.axis([-4, 4, -4, 4])
plt.title('Positive vs. Negative class in the feature space')
plt.show()

