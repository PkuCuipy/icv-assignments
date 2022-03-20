import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img
from PIL import Image
import matplotlib.pyplot as plt

"""
⚠️ 我对 x 和 y 的理解是
      • — — —> y
      |
      |
      v
      x
但原始代码中 Sobel_x 和 Sobel_y 的构造意味着它和我的理解是反的.
很晚我才意识到这一点, 补救方法是: 在这个函数中, 交换形式参数的顺序.
即: 在我的代码命名中, x 和 y 永远都是按我的理解来的, 这个函数则帮助完成了二者的兼容对接.
"""
def compute_gradient_magnitude_direction(y_grad, x_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)         # 强度矩阵 M
    direction_grad = np.arctan(y_grad / (x_grad + 1e-20))   # 方向矩阵 D
    direction_grad[x_grad < 0] += np.pi

    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """

    # 并行化的双线性插值, img 是二维矩阵, 列表 x 和列表 y 的对应元素拼在一起表示某个坐标.
    def bilinear_interpolate(img: np.ndarray, x: np.ndarray, y: np.ndarray):

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, img.shape[0] - 1)
        x1 = np.clip(x1, 0, img.shape[0] - 1)
        y0 = np.clip(y0, 0, img.shape[1] - 1)
        y1 = np.clip(y1, 0, img.shape[1] - 1)

        Ia = img[x0, y0]
        Ib = img[x1, y0]
        Ic = img[x0, y1]
        Id = img[x1, y1]

        wa = (y1 - y) * (x1 - x)
        wb = (y1 - y) * (x - x0)
        wc = (y - y0) * (x1 - x)
        wd = (y - y0) * (x - x0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    # 计算每个点的梯度向量, 得到两个 (N, M) 矩阵, 分别表示 x 方向和 y 方向的梯度
    grad_x = magnitude_grad * np.cos(grad_dir)
    grad_y = magnitude_grad * np.sin(grad_dir)

    # 对于每个格点位置, 计算当前位置的梯度强度 A, 计算沿梯度正方向走一步的梯度强度值 B, 计算沿梯度反方向走一步的梯度强度值 C.
    A = grad_mag

    yB, xB = np.meshgrid(np.arange(grad_mag.shape[1]), np.arange(grad_mag.shape[0]))
    xB, yB = (xB + grad_x).ravel(), (yB + grad_y).ravel()
    B = bilinear_interpolate(grad_mag, xB, yB).reshape(A.shape)

    # # DEBUG: 打印梯度图 (quiver 图)
    # Y, X = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))
    # plt.quiver(Y, X, grad_y, grad_x, angles='xy', scale_units='xy', scale=1)

    yC, xC = np.meshgrid(np.arange(grad_mag.shape[1]), np.arange(grad_mag.shape[0]))
    xC, yC = (xC - grad_x).ravel(), (yC - grad_y).ravel()
    C = bilinear_interpolate(grad_mag, xC, yC).reshape(A.shape)

    # 如果发现 A > B 且 A > C, 则说明 A 是一个 Maximal, 否则就是 Non-Maximal, 需要被 Suppress.
    NMS_output = grad_mag * ((A>B) & (A>C))

    # # debug: before NMS <--> after NMS
    # (lambda im: Image.fromarray((im / im.max() * 255).astype("u1")).show())(grad_mag)
    # (lambda im: Image.fromarray((im / im.max() * 255).astype("u1")).show())(NMS_output)

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """

    # 设置两个阈值 H 和 L, 认为高于 H 的一定是边缘, 认为低于 L 的一定不是边缘, 认为介于二者之间的仍有待判断
    high_threshold = 0.3
    low_threshold = 0.1

    strong_mask = img > high_threshold
    not_mask = img < low_threshold
    still_unsure = (low_threshold <= img) & (img <= high_threshold)

    # 如果一个 unsure pixel 的周围有 strong pixel, 则认为它也是 strong 的
    for i in range(strong_mask.shape[0]):
        for j in range(strong_mask.shape[1]):
            if still_unsure[i, j] and strong_mask[i-1:i+2, j-1:j+2].any():
                still_unsure[i, j] = False
                strong_mask[i, j] = True

    img[strong_mask] = 255     # 强边缘
    img[not_mask] = 0        # 不是边缘
    img[still_unsure] = 0   # 也不是边缘

    output = img.copy()
    return output


if __name__ == "__main__":
    # Load the input images
    input_img = read_img("Lenna.png") / 255

    # Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    # Compute the magnitude and the direction of gradient
    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    # NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    # Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    # Save image
    write_img("result/HM1_Canny_result.png", output_img * 255)
