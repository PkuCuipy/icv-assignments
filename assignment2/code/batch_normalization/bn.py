import numpy as np
import cv2

# eps may help you to deal with numerical problem
eps = 1e-5


def bn_forward_test(x: np.ndarray, gamma, beta, mean, var):
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    return out


def bn_forward_train(x: np.ndarray, gamma, beta):
    sample_mean = x.mean(axis=0, keepdims=True)     # x: (batch_size, n_channel)
    sample_var = x.var(axis=0, keepdims=True)       # -> (1, n_channel)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta

    # save intermediate variables for computing the gradient when backward
    cache = (gamma, x, sample_mean, sample_var, x_hat)
    return out, cache


def bn_backward(dout, cache):

    # 简记
    (gamma, x, mu, sigma2, x_hat) = cache
    sigma2 = sigma2 + eps   # σ² + ε
    N = x.shape[0]          # batch size

    # ∂L/∂γ 和 ∂L/∂β
    dgamma = (dout * x_hat).sum(axis=0)
    dbeta = dout.sum(axis=0)

    # dx 根据计算图拆为三部分之和
    grad_L_xhat = dout * gamma
    grad_L_sigma2 = -0.5 * (grad_L_xhat * (x - mu)).sum(axis=0) * sigma2**(-1.5)
    grad_L_mu = -(grad_L_xhat / np.sqrt(sigma2)).sum(axis=0)     # 这里舍去了带 Σ(xi-μ) 的项, 因为其恒为 0
    dx = grad_L_xhat / np.sqrt(sigma2) + 2 * grad_L_sigma2 * (x - mu) / N + grad_L_mu / N

    return dx, dgamma, dbeta


# This function may help you to check your code
def print_info(x):
    print('mean:', np.mean(x, axis=0))
    print('var:', np.var(x, axis=0))
    print('------------------')
    return


if __name__ == "__main__":

    # input data
    train_data = np.zeros((9, 784))
    for i in range(9):
        train_data[i, :] = cv2.imread("mnist_subset/" + str(i) + ".png", cv2.IMREAD_GRAYSCALE).reshape(-1) / 255.
    gt_y = np.zeros((9, 1))
    gt_y[0] = 1

    val_data = np.zeros((1, 784))
    val_data[0, :] = cv2.imread("mnist_subset/9.png", cv2.IMREAD_GRAYSCALE).reshape(-1) / 255.
    val_gt = np.zeros((1, 1))

    np.random.seed(14)

    # Intialize MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784, 16)
    MLP_layer_2 = np.random.randn(16, 1)

    # Initialize gamma and beta
    gamma = np.random.randn(16)
    beta = np.random.randn(16)

    lr = 1e-1
    loss_list = []

    #>>> compute mean and var for testing >>>
    running_mean = np.zeros(16)
    running_var = np.zeros(16)
    rho = 0.9
    #<<< compute mean and var for testing <<<

    # training 
    for i in range(50):
        # Forward
        output_layer_1 = train_data.dot(MLP_layer_1)
        output_layer_1_bn, cache = bn_forward_train(output_layer_1, gamma, beta)
        output_layer_1_act = 1 / (1 + np.exp(-output_layer_1_bn))  # sigmoid activation function
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
        pred_y = 1 / (1 + np.exp(-output_layer_2))  # sigmoid activation function

        # compute loss
        loss = -(gt_y * np.log(pred_y) + (1 - gt_y) * np.log(1 - pred_y)).sum()
        print("iteration: %d, loss: %f" % (i + 1, loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)
        grad_pred_y = -(gt_y / pred_y) + (1 - gt_y) / (1 - pred_y)
        grad_activation_func = grad_pred_y * pred_y * (1 - pred_y)
        grad_layer_2 = output_layer_1_act.T.dot(grad_activation_func)
        grad_output_layer_1_act = grad_activation_func.dot(MLP_layer_2.T)
        grad_output_layer_1_bn = grad_output_layer_1_act * (1 - output_layer_1_act) * output_layer_1_act
        grad_output_layer_1, grad_gamma, grad_beta = bn_backward(grad_output_layer_1_bn, cache)
        grad_layer_1 = train_data.T.dot(grad_output_layer_1)

        # update parameters
        gamma -= lr * grad_gamma
        beta -= lr * grad_beta
        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2
        running_mean = rho * running_mean + (1 - rho) * cache[2]    # <<< streaming algo
        running_var = rho * running_var + (1 - rho) * cache[3]      # <<< streaming algo

    # validate
    output_layer_1 = val_data.dot(MLP_layer_1)
    output_layer_1_bn = bn_forward_test(output_layer_1, gamma, beta, running_mean, running_var)
    output_layer_1_act = 1 / (1 + np.exp(-output_layer_1_bn))  # sigmoid activation function
    output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
    pred_y = 1 / (1 + np.exp(-output_layer_2))  # sigmoid activation function
    loss = -(val_gt * np.log(pred_y) + (1 - val_gt) * np.log(1 - pred_y)).sum()
    print("validation loss: %f" % (loss))
    loss_list.append(loss)

    np.savetxt("../results/bn_loss.txt", np.array(loss_list))
