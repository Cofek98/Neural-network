import numpy as np
import matplotlib.pyplot as plt
import activations
import gradient as g
import classic_nn
import residual_nn


# preform the gradient test for the loss function w.r.t. w
def loss_grad_w_test(x, c):
    w = np.array(np.random.randn(3, 5))
    d = np.random.randn(3, 5)
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    f0 = activations.cross_entropy_loss(x, w, c)
    g0 = g.calc_grad_w(x, w, c)

    dot_product = np.dot(g0.flatten(), d.flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        fk = activations.cross_entropy_loss(x, w + (epsk * d), c)
        f1 = f0 + (epsk * dot_product)
        y0[k - 1] = abs(fk - f0)
        y1[k - 1] = abs(fk - f1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Grad test of loss w.r.t. w in semilogarithmic plot")


# preform the gradient test for the loss function w.r.t. x
def loss_grad_x_test():
    x = np.array(np.random.randn(3, 20))
    c = np.zeros((5, 20))
    for col in range(20):
        row_index = np.random.randint(5)
        c[row_index, col] = 1
    w = np.array(np.random.randn(3, 5))
    d = np.random.randn(3, 20)
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    f0 = activations.cross_entropy_loss(x, w, c)
    g0 = g.calc_grad_x(x, w, c)

    dot_product = np.dot(g0.flatten(), d.flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        fk = activations.cross_entropy_loss(x + (epsk * d), w, c)
        f1 = f0 + (epsk * dot_product)
        y0[k - 1] = abs(fk - f0)
        y1[k - 1] = abs(fk - f1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Grad test of loss w.r.t. x in semilogarithmic plot")


# preform the direct jacobian transpose test w.r.t. w
def act_jac_w_transpose_test(x):
    if x.ndim != 1:
        m = x.shape[1]
    else:
        m = 1
    w = np.array(np.random.randn(3, 5))
    d = np.random.randn(3, 5)
    u = np.array(np.random.randn(5, m))
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    f0 = activations.hidden_layer_forward_pass(w, x)
    g0 = np.dot(f0.flatten(), u.flatten())
    grad_g0 = g.jac_w_transpose(w, x, u)
    dot_product = np.dot(grad_g0.flatten(), d.flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        fk = activations.hidden_layer_forward_pass(w + (epsk * d), x)
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * dot_product)
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Jacobian test of tanh w.r.t w in semilogarithmic plot")


# preform the direct jacobian transpose test w.r.t. w for residual network
def act_jac_w_transpose_test_residual():
    x = np.array(np.random.randn(5, 20))
    c = np.zeros((5, 20))
    for col in range(20):
        row_index = np.random.randint(5)
        c[row_index, col] = 1
    w1 = np.array(np.random.randn(5, 5))
    w2 = np.array(np.random.randn(5, 5))
    d = np.random.randn(2, 5, 5)
    u = np.array(np.random.randn(5, 20))
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    f0 = (activations.hidden_layer_forward_pass_residual(w1, w2, x))
    g0 = np.dot(f0.flatten(), u.flatten())
    grad_g0_w1 = g.jac_w1_transpose_residual(w1, w2, x, u)
    grad_g0_w2 = g.jac_w2_transpose_residual(w1, x, u)
    grad_g0 = np.array([grad_g0_w1, grad_g0_w2])
    dot_product = np.dot(grad_g0.flatten(), d.flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        fk = activations.hidden_layer_forward_pass_residual(w1 + (epsk * d[0]),
                                                            w2 + (epsk * d[1]),
                                                            x)
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * dot_product)
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Jacobian test of tanh w.r.t w for residual network in semilogarithmic plot")


# preform the direct jacobian transpose test w.r.t. x
def act_jac_x_transpose_test():
    x = np.array(np.random.randn(3, 20))
    w = np.array(np.random.randn(3, 5))
    d = np.random.randn(3, 20)
    u = np.array(np.random.randn(5, 20))
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    f0 = activations.hidden_layer_forward_pass(w, x)
    g0 = np.dot(f0.flatten(), u.flatten())
    grad_g0 = g.jac_x_transpose(w, x, u)
    dot_product = np.dot(grad_g0.flatten(), d.flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        fk = activations.hidden_layer_forward_pass(w, x + (epsk * d))
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * dot_product)
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Jacobian test of tanh w.r.t x in semilogarithmic plot")


# preform the direct jacobian transpose test w.r.t. x for residual network
def act_jac_x_transpose_test_residual():
    x = np.array(np.random.randn(5, 20))
    w1 = np.array(np.random.randn(5, 5))
    w2 = np.array(np.random.randn(5, 5))
    d = np.random.randn(5, 20)
    u = np.array(np.random.randn(5, 20))
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    f0 = activations.hidden_layer_forward_pass_residual(w1, w2, x)
    g0 = np.dot(f0.flatten(), u.flatten())
    grad_g0 = g.jac_x_transpose_residual(w1, w2, x, u)
    dot_product = np.dot(grad_g0.flatten(), d.flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        fk = activations.hidden_layer_forward_pass_residual(w1, w2,
                                                            x + (epsk * d))
        gk = np.dot(fk.flatten(), u.flatten())
        g1 = g0 + (epsk * dot_product)
        y0[k - 1] = abs(gk - g0)
        y1[k - 1] = abs(gk - g1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Jacobian test of tanh w.r.t x for residual network in semilogarithmic plot")


# generate random weight matrices
def create_random_w(is_classic, num_of_layers, d1, d2, d3, d4):
    first_w = np.random.randn(d1, d2) * 0.1
    w = [first_w]
    for _ in range(1, num_of_layers - 1):
        w.append(np.random.randn(d2, d3) * 0.1)
    w.append(np.random.randn(d2, d4) * 0.1)
    if is_classic:
        return w

    else:
        w2 = [first_w]
        for _ in range(1, num_of_layers - 1):
            w2.append(np.random.randn(d3, d2) * 0.1)
        w2.append(np.random.randn(d2, d4))
        return [w, w2]


# preform the gradient test for the whole classic NN
def whole_net_grad_test(num_of_layers):
    x = np.array(np.random.randn(3, 20))
    w = create_random_w(True, num_of_layers, 3, 4, 4, 5)
    d = create_random_w(True, num_of_layers, 3, 4, 4, 5)
    c = np.zeros((5, 20))
    for col in range(20):
        row_index = np.random.randint(5)
        c[row_index, col] = 1
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    a_values = classic_nn.forward_pass(x, w)
    f0 = activations.cross_entropy_loss(a_values[-1], w[-1], c)
    g0 = []
    grad = classic_nn.backward_pass(x, w, c)
    for i in range(len(w)):
        g0.append(grad[i])

    dot_product = 0
    for i in range(len(g0)):
        dot_product += np.dot(g0[i].flatten(), d[i].flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        w_k = []
        for i in range(len(d)):
            w_k.append(w[i] + (epsk * d[i]))

        a_values_k = classic_nn.forward_pass(x, w_k)
        fk = activations.cross_entropy_loss(a_values_k[-1], w_k[-1], c)
        f1 = f0 + (epsk * dot_product)
        y0[k - 1] = abs(fk - f0)
        y1[k - 1] = abs(fk - f1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Grad test of classic NN in semilogarithmic plot")


# preform the gradient test for the whole residual NN
def whole_net_grad_test_residual(num_of_layers):
    x = np.array(np.random.randn(3, 20))
    w = create_random_w(False, num_of_layers, 3, 4, 10, 5)
    w1 = w[0]
    w2 = w[1]
    d = create_random_w(False, num_of_layers, 3, 4, 10, 5)
    d1 = d[0]
    d2 = d[1]

    c = np.zeros((5, 20))
    for col in range(20):
        row_index = np.random.randint(5)
        c[row_index, col] = 1
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    a_values = residual_nn.forward_pass(x, w1, w2)
    f0 = activations.cross_entropy_loss(a_values[-1], w1[-1], c)
    g0 = [[], []]
    grad = residual_nn.backward_pass(x, w1, w2, c)
    for i in range(len(w1)):
        g0[0].append(grad[0][i])
        g0[1].append(grad[1][i])

    dot_product = 0
    for i in range(len(g0[0])):
        dot_product += np.dot(g0[0][i].flatten(), d1[i].flatten()) + np.dot(
                g0[1][i].flatten(), d2[i].flatten())

    print("k\terror order 1 \t\t\t error order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        w_k = [[], []]
        for i in range(len(d1)):
            w_k[0].append(w1[i] + (epsk * d1[i]))
            w_k[1].append(w2[i] + (epsk * d2[i]))

        a_values_k = residual_nn.forward_pass(x, w_k[0], w_k[1])
        fk = activations.cross_entropy_loss(a_values_k[-1], w_k[0][-1], c)
        f1 = f0 + (epsk * dot_product)
        y0[k - 1] = abs(fk - f0)
        y1[k - 1] = abs(fk - f1)
        print(k, "\t", y0[k - 1], "\t\t", y1[k - 1])

    plot_errors(y0, y1,
                "Successful Grad test of classic NN in semilogarithmic plot")


# plot the errors of y0, y1
def plot_errors(y0, y1, title):
    plt.semilogy(np.arange(1, 9), y0)
    plt.semilogy(np.arange(1, 9), y1)
    plt.legend(("Zero order approx", "First order approx"))
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()
