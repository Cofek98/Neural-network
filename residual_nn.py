import numpy as np
import gradient
import activations


# shapes are:
# x = features * samples
# c = labels * samples
# w = features * labels


# predicts the label of x using w
def predict(x, w1, w2):
    num_of_hidden_layers = len(w1) - 1
    # first layer is classic
    curr_layer = activations.hidden_layer_forward_pass(w1[0], x)
    for i in range(1, num_of_hidden_layers):
        curr_layer = activations.hidden_layer_forward_pass_residual(w1[i],
                                                                    w2[i],
                                                                    curr_layer)
    curr_layer = activations.output_layer_forward_pass(
            w1[num_of_hidden_layers], curr_layer)
    predicted = [np.argmax(curr_layer[:, s]) for s in
                 range(curr_layer.shape[1])]
    return predicted


# preform the back propagation
def backward_pass(x, w1, w2, c):
    a_values = forward_pass(x, w1, w2)
    grad1 = [None] * len(w1)
    grad2 = [None] * len(w2)

    grad_last1 = gradient.calc_grad_w(a_values[-1], w1[-1], c)
    grad_last2 = np.zeros_like(grad_last1)
    grad1[-1] = grad_last1
    grad2[-1] = grad_last2

    v = gradient.calc_grad_x(a_values[-1], w1[-1], c)

    # Propagate the gradient backwards through the network
    for l in range(len(grad1) - 2, 0, -1):
        curr_delta1 = gradient.jac_w1_transpose_residual(w1[l], w2[l],
                                                         a_values[l], v)
        curr_delta2 = gradient.jac_w2_transpose_residual(w1[l], a_values[l],
                                                         v)
        grad1[l] = curr_delta1
        grad2[l] = curr_delta2
        v = gradient.jac_x_transpose_residual(w1[l], w2[l], a_values[l], v)
    first_grad = gradient.jac_w_transpose(w1[0], a_values[0], v)
    grad1[0] = first_grad
    grad2[0] = np.zeros_like(first_grad)

    return [grad1, grad2]


# train the network
def forward_pass(x, w1, w2):
    num_of_hidden_layers = len(w1) - 1
    a_values = [x]
    a = activations.hidden_layer_forward_pass(w1[0], x)
    a_values.append(a)
    for i in range(1, num_of_hidden_layers):
        a = activations.hidden_layer_forward_pass_residual(w1[i], w2[i],
                                                           a_values[-1])
        a_values.append(a)

    return a_values
