import numpy as np
import gradient
import activations


# shapes are:
# x = features * samples
# c = labels * samples
# w = features * labels


# predicts the label of x using w
def predict(x, w):
    num_of_hidden_layers = len(w) - 1
    curr_layer = x
    for i in range(num_of_hidden_layers):
        curr_layer = activations.hidden_layer_forward_pass(w[i], curr_layer)
    curr_layer = activations.output_layer_forward_pass(w[num_of_hidden_layers],
                                                       curr_layer)

    return np.argmax(curr_layer)


# preform the back propagation
def backward_pass(x, w, c):
    a_values = forward_pass(x, w)
    grad = [None] * len(w)
    grad_last = gradient.calc_grad_w(a_values[-1], w[-1], c)
    grad[-1] = grad_last
    v = gradient.calc_grad_x(a_values[-1], w[-1], c)  # first v

    # Propagate the gradient backwards through the network
    for l in range(len(grad) - 2, -1, -1):
        curr_delta = gradient.jac_w_transpose(w[l], a_values[l], v)
        grad[l] = curr_delta
        v = gradient.jac_x_transpose(w[l], a_values[l], v)

    return grad


# train the network
def forward_pass(x, w):
    num_of_hidden_layers = len(w) - 1
    a_values = [x]
    for i in range(num_of_hidden_layers):
        a = activations.hidden_layer_forward_pass(np.array(w[i]), a_values[-1])
        a_values.append(a)

    return a_values
