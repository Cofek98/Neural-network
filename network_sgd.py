import numpy as np
import activations
import matplotlib.pyplot as plt
import residual_nn


# preform SGD for the network
def stochastic_gradient_descent(num_of_layers, xt, yt, xv, yv, max_iter=100,
                                learning_rate=0.1, batch_size=12500, d1w=10,
                                d2w=4):
    f, s = xt.shape
    l = yt.shape[0]
    # first layer is "classic" - changes dimensions
    first_w = np.random.randn(f, d1w) / d1w
    w1 = [first_w]
    w2 = [first_w]
    for _ in range(1, num_of_layers - 1):
        # hidden layers are residual
        w1.append(np.random.randn(d1w, d2w) / (max(d1w, d2w)))
        w2.append(np.random.randn(d2w, d1w) / (max(d1w, d2w)))
    # last layer is "classic" - changes dimensions
    w1.append(np.random.randn(d1w, l) / d1w)
    w2.append(np.random.randn(d1w, l) / d1w)

    losses = []
    t_success = []
    v_success = []

    for epoch in range(1, max_iter + 1):

        indices = np.random.permutation(s)

        if epoch % 25 == 0:
            learning_rate = learning_rate * 0.5

        for k in range(s // batch_size):
            curr_indices = indices[k * batch_size: (k + 1) * batch_size]
            xk = np.array([xt.T[i] for i in curr_indices])
            yk = np.array([yt.T[i] for i in curr_indices])

            grad = residual_nn.backward_pass(xk.T, w1, w2, yk.T)

            for i in range(len(w1)):
                w1[i] = w1[i] - learning_rate * grad[0][i]
                w2[i] = w2[i] - learning_rate * grad[1][i]

        t_success.append(calc_success(xt, w1, w2, yt))
        v_success.append(calc_success(xv, w1, w2, yv))
        last_x = residual_nn.forward_pass(xt, w1, w2)[-1]
        loss = activations.cross_entropy_loss(last_x, w1[-1], yt)
        losses.append(loss)

    return losses, t_success, v_success


# demonstrate and plot SGD for the network
def net_demonstration(learning_rate, epochs, mini_batch_size, num_of_layers,
                      Xt, Ct, Xv, Cv, d1w=10, d2w=4):
    sgd_norms, t_suc, v_suc = stochastic_gradient_descent(num_of_layers, Xt,
                                                          Ct, Xv, Cv, epochs,
                                                          learning_rate,
                                                          mini_batch_size, d1w,
                                                          d2w)

    plt.figure('stochastic gradient for softmax')

    plt.subplot(1, 2, 1)
    plt.semilogy(sgd_norms)
    plt.legend(["loss"])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(t_suc)
    plt.plot(v_suc)
    plt.legend(
            ['success of training samples', 'success of validation samples'])
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.title('succession over epochs')

    plt.suptitle(
            'Learning rate = ' + str(
                    learning_rate) + ', Mini batch size = ' + str(
                    mini_batch_size) + ', Layers = ' + str(num_of_layers))
    plt.show()


# calculate the success rate of prediction for w1 w2
def calc_success(x, w1, w2, c):
    xt = x.T
    ct = c.T
    indices = np.random.permutation(xt.shape[0])[:100]
    sub_x = (np.array([xt[i] for i in indices])).T
    sub_c = (np.array([ct[i] for i in indices])).T
    predict_labels = residual_nn.predict(sub_x, w1, w2)
    actual_labels = np.argmax(sub_c, axis=0)
    success = np.sum(predict_labels == actual_labels)

    return success / 100
