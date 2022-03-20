import numpy as np
from utils import read_img

if __name__ == "__main__":

    ### Input
    input_vector = a0 = np.zeros((10, 784))
    for i in range(10):
        input_vector[i, :] = read_img("mnist_subset/" + str(i) + ".png").reshape(-1) / 255.
    tag = np.zeros((10, 1))
    tag[0] = 1

    np.random.seed(14)

    ### Intialization MLP [ 784 -> 16 -> 1 ]
    W1 = np.random.randn(784, 16)
    W2 = np.random.randn(16, 1)
    lr = 1e-1
    loss_list = []

    ### Tool Functions
    def sigmoid(x):
        """ Sigmoid Function """
        return 1 / (1 + np.exp(-x))

    def CELoss(x, y):
        """ Cross-Entroy Loss """
        return -(x * np.log(y) + (1 - x) * np.log(1 - y)).sum()

    ### Train
    for i in range(50):

        # Forward & 计算 Local Gradient
        z1 = np.dot(a0, W1);        z1_deri_a0 = W1.T;              z1_deri_W1 = a0.T
        a1 = sigmoid(z1);           a1_deri_z1 = a1 * (1 - a1)      # 是 a(1-a) 而非 z(1-z).. debug 1e4 years
        z2 = np.dot(a1, W2);        z2_deri_a1 = W2.T;              z2_deri_W2 = a1.T
        a2 = sigmoid(z2);           a2_deri_z2 = a2 * (1 - a2)
        loss = CELoss(tag, a2);     loss_deri_a2 = (a2 - tag) / (a2 * (1 - a2))
        print("iteration: %d, loss: %f" % (i + 1, loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 and layer2
        loss_deri_z2 = loss_deri_a2 * a2_deri_z2    # ∂L/∂z2 = ∂L/∂a2 • ∂a2/∂z2
        loss_deri_W2 = z2_deri_W2 @ loss_deri_z2    # ∂L/∂W2 = ∂L/∂z2 • ∂z2/∂W2
        loss_deri_a1 = loss_deri_z2 @ z2_deri_a1    # ∂L/∂a1 = ∂L/∂z2 • ∂z2/∂a1
        loss_deri_z1 = loss_deri_a1 * a1_deri_z1    # ∂L/∂z1 = ∂L/∂a1 • ∂a1/∂z1
        loss_deri_W1 = z1_deri_W1 @ loss_deri_z1    # ∂L/∂W1 = ∂L/∂z1 • ∂z1/∂W1

        grad_layer_1 = loss_deri_W1
        grad_layer_2 = loss_deri_W2

        # 沿梯度反向走一小步
        W1 -= lr * grad_layer_1
        W2 -= lr * grad_layer_2


    np.savetxt("result/HM1_BP.txt", np.array(loss_list))






