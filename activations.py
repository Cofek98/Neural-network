import numpy as np


# shapes are:
# x = features * samples
# c = labels * samples
# w = features * labels


# calculates the softmax
def softmax(x, w):
    a = x.T @ w
    dividend = np.exp(a - np.max(a, axis=1, keepdims=True))
    divisor = np.sum(dividend, axis=1, keepdims=True)
    return dividend / divisor


# calculates the loss of W
def cross_entropy_loss(x, w, c):
    m = x.shape[1]
    loss = 0
    sm = softmax(x, w)
    for k in range(w.shape[1]):
        loss += np.dot(c[k].T, np.log(sm.T[k]))
    return (-1 / m) * loss


# calculates the next layer (hidden layer)
def hidden_layer_forward_pass(w, x):
    z = w.T @ x
    a = np.tanh(z)
    return a


# calculates the next layer (hidden layer) for residual network
def hidden_layer_forward_pass_residual(w1, w2, x):
    a1 = hidden_layer_forward_pass(w1, x)
    a2 = x + w2.T @ a1
    return a2


# calculates the output layer
def output_layer_forward_pass(w, x):
    a = softmax(x, w).T
    return a
