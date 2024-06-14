import numpy as np
import matplotlib.pyplot as plt


# Objective function: Least squares
def least_squares(x, b, y):
    return np.sum((np.dot(x, b) - y) ** 2) / (2 * len(y))


# Gradient of Least squares objective function
def least_squares_gradient(x: np.array, b, y: np.array):
    return (2 * np.dot(np.dot(x.T, x), b)) - (2 * np.dot(x.T, y))


# calculate the loss of Least squares
def least_squares_loss(x, b, y):
    # Calculate the least squares loss
    l = len(y)
    loss = (1 / l) * np.sum((np.dot(x, b) - y) ** 2)
    return loss


# Stochastic Gradient Descent (SGD) for Least squares
def stochastic_gradient_descent(x, y, grad_func, max_iter=100,
                                learning_rate=0.1, batch_size=10):
    m, n = x.shape
    d = y.shape[1]
    w = np.zeros([n, d])
    losses = []

    for epoch in range(1, max_iter + 1):

        indices = np.random.permutation(m)

        if epoch % 25 == 0:
            learning_rate = learning_rate * 0.5

        for k in range(m // batch_size):
            curr_indices = indices[k * batch_size: (k + 1) * batch_size]
            xk = np.array([x[i] for i in curr_indices])
            yk = np.array([y[i] for i in curr_indices])

            grad = (1 / batch_size) * (grad_func(xk, w, yk))
            w = w - (learning_rate * grad)

        loss = least_squares_loss(x, w, y)
        losses.append(loss)
    return losses


# demonstrate and plot the SGD for LS
def ls_demonstration():
    # Generate synthetic data for demonstration of
    X = np.array(np.random.rand(500, 5))
    W = np.array(np.random.rand(5))
    Y = np.array(X * W + 0.05 * np.random.randn(500, 1))

    losses = stochastic_gradient_descent(X, Y, least_squares_gradient)

    plt.figure()
    plt.title('stochastic gradient descent for least squares')
    plt.plot(losses)
    plt.legend(["sgd"])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
