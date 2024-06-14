import numpy as np
import activations


# tanh'(x)
def activation_derivative(x):
    return 1 - np.tanh(x) ** 2


# jacobian of hidden layer w.r.t w (contains b)
def jac_w_transpose(w, x, v):
    act_der = activation_derivative(np.dot(w.T, x))
    return np.dot(np.multiply(act_der, v), x.T).T


# jacobian of hidden layer w.r.t w1 (contains b) for residual network
def jac_w1_transpose_residual(w1, w2, x, v):
    act_der = activation_derivative(np.dot(w1.T, x))
    ans = np.dot(np.multiply(act_der, w2 @ v), x.T)
    return ans.T


# jacobian of hidden layer w.r.t w2 (contains b) for residual network
def jac_w2_transpose_residual(w1, x, v):
    ans = np.dot(v, activations.hidden_layer_forward_pass(w1, x).T)
    return ans.T


# jacobian of hidden layer w.r.t x
def jac_x_transpose(w, x, v):
    act_der = activation_derivative(np.dot(w.T, x))
    ans = np.dot(w, np.multiply(act_der, v))
    return ans


# jacobian of hidden layer w.r.t x for residual network
def jac_x_transpose_residual(w1, w2, x, v):
    act_der = activation_derivative(np.dot(w1.T, x))
    tmp = np.dot(w1, np.multiply(act_der, w2 @ v))
    return v + tmp


# calculates the gradient of the loss function w.r.t w
def calc_grad_w(x, w, c):
    m = x.shape[1]
    sm = activations.softmax(x, w)
    tmp = np.dot(x, (sm.T - c).T)
    return (1 / m) * tmp


# calculates the gradient of the loss function w.r.t x
def calc_grad_x(x, w, c):
    m = x.shape[1]
    dividend = np.exp(np.dot(w.T, x))
    divisor = np.zeros(np.array(dividend).shape)
    for j in range(w.shape[1]):
        wj = w[:, j]
        divisor = divisor + np.exp(np.dot(wj.T, x))
    tmp = np.divide(dividend, divisor)
    ans = (1 / m) * np.dot(w, tmp - c)
    return ans
