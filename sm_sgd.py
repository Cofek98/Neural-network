import numpy as np
import activations
import matplotlib.pyplot as plt
import gradient


# Stochastic Gradient Descent (SGD) for loss
def stochastic_gradient_descent(xt, yt, xv, yv, grad_func, max_iter=100,
                                learning_rate=0.1, batch_size=1000):
    f, s = xt.shape
    l = yt.shape[0]
    w = np.random.randn(f, l)

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

            grad = grad_func(xk.T, w, yk.T)
            w = w - (learning_rate * grad)

        t_success.append(calc_success(xt, w, yt))
        v_success.append(calc_success(xv, w, yv))
        loss = activations.cross_entropy_loss(xt, w, yt)
        losses.append(loss)

    return losses, t_success, v_success


# demonstrate and plot the SGD for LS
def sm_demonstration(Xt, Ct, Xv, Cv):
    learning_rate = 0.1
    epochs = 100
    mini_batch_size = 25000
    losses, t_suc, v_suc = stochastic_gradient_descent(Xt, Ct, Xv, Cv,
                                                       gradient.calc_grad_w,
                                                       epochs, learning_rate,
                                                       mini_batch_size)

    plt.figure('stochastic gradient for softmax')

    plt.subplot(1, 2, 1)
    plt.plot(losses)
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
                    mini_batch_size))
    plt.show()


# calculate the success rate of prediction for w
def calc_success(x, w, c):
    xt = x.T
    ct = c.T
    indices = np.random.permutation(xt.shape[0])[:100]
    sub_x = (np.array([xt[i] for i in indices])).T
    sub_c = (np.array([ct[i] for i in indices])).T

    # y_predict = np.zeros(sub_c.shape)
    # for k in range(w.shape[1]):
    #     y_predict[k] = activations.softmax(sub_x, w, k)
    y_predict = activations.softmax(sub_x, w)
    predict_labels = np.argmax(y_predict, axis=0)
    actual_labels = np.argmax(sub_c, axis=0)
    success = np.sum(predict_labels == actual_labels)

    return success / 100
