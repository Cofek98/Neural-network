import numpy as np
import matplotlib.pyplot as plt
import ls_sgd
import classic_nn
import gradient as grad
import scipy.io as sc
import sm_sgd
import tests
import network_sgd

# load the Peaks data
mat_data = sc.loadmat('PeaksData.mat')
Xt = np.array(mat_data['Yt'])
Ct = np.array(mat_data['Ct'])
Xv = np.array(mat_data['Yv'])
Cv = np.array(mat_data['Cv'])

# adding bias
ones_row_t = np.ones((1, Xt.shape[1]))
Xt = np.vstack((ones_row_t, Xt))
ones_row_v = np.ones((1, Xv.shape[1]))
Xv = np.vstack((ones_row_v, Xv))

# load the GMM data
mat_data = sc.loadmat('GMMData.mat')
X1t = np.array(mat_data['Yt'])
C1t = np.array(mat_data['Ct'])
X1v = np.array(mat_data['Yv'])
C1v = np.array(mat_data['Cv'])

# adding bias
ones_row_t = np.ones((1, X1t.shape[1]))
X1t = np.vstack((ones_row_t, X1t))
ones_row_v = np.ones((1, X1v.shape[1]))
X1v = np.vstack((ones_row_v, X1v))


# part 1 q 1
# preform the gradient test for data
# tests.loss_grad_w_test(Xt, Ct)

# part 1 q 2
# demonstrates and verify the SGD for least squares
# ls_sgd.ls_demonstration()

# part 1 q 3
# demonstrates and verify the SGD for cross entropy loss- best option
# sm_sgd.sm_demonstration(Xt, Ct, Xv, Cv)

# part 2 q 1
# the gradient and jacobian tests for the classic NN
# tests.loss_grad_x_test()
# tests.act_jac_w_transpose_test(Xt[:, 0:2])
# tests.act_jac_x_transpose_test()

# part 2 q 2
# the gradient and jacobian tests for the residual NN
# tests.act_jac_w_transpose_test_residual()
# tests.act_jac_x_transpose_test_residual()

# part 2 q 3
# the gradient tests for the whole networks
# tests.whole_net_grad_test(4)
# tests.whole_net_grad_test_residual(4)

# part 2 q 4
# Preform SGD on the whole network for Peaks data:
# network_sgd.net_demonstration(0.5, 200, 500, 4, Xt, Ct, Xv, Cv)
# Preform SGD on the whole network for GMM data:
# network_sgd.net_demonstration(1, 200, 500, 4, X1t, C1t, X1v, C1v)

# part 2 q 5
# Peaks data:
# network_sgd.net_demonstration(0.5, 150, 500, 4, Xt, Ct, Xv, Cv, 32, 2)
# GMM data:
# network_sgd.net_demonstration(1, 150, 500, 4, X1t, C1t, X1v, C1v, 32, 2)
