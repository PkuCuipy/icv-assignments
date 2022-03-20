import numpy as np
from utils import read_img, write_img
import typing


def padding(img: np.ndarray, pad: int, type: str):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            pad: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    assert len(img.shape) == 2
    assert img.dtype == float

    imH, imW = img.shape
    newH, newW = (imH + 2 * pad, imW + 2 * pad)

    padding_img = np.zeros(shape=(newH, newW), dtype=float)
    padding_img[pad: imH + pad, pad: imW + pad] = img

    if type == "zeroPadding":
        return padding_img

    elif type == "replicatePadding":
        padding_img[:, 0: pad] = padding_img[:, pad: pad + 1]                               # 补左侧
        padding_img[:, imW + pad: imW + pad * 2] = padding_img[:, imW + pad - 1: imW + pad] # 补右侧
        padding_img[0: pad, :] = padding_img[pad: pad + 1, :]                               # 补上面
        padding_img[imH + pad: imH + pad * 2, :] = padding_img[imH + pad - 1: imH + pad, :] # 补下面
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # zero padding
    padding_img = padding(img, 1, "zeroPadding")

    # build the Toeplitz matrix
    T = np.zeros(shape=[1,64])
    T[0, 0:3] = kernel[0, :]; T[0, 8:11] = kernel[1, :]; T[0, 16:19] = kernel[2, :] # 构建第 1 行
    T = np.concatenate([T, np.roll(T, 1, axis=1)], axis=0)                          # 构建前 2 行
    T = np.concatenate([T, np.roll(T, 2, axis=1), np.roll(T, 4, axis=1)], axis=0)   # 构建前 6 行
    T = np.concatenate([T, np.roll(T, 8, axis=1)], axis=0)                          # 构建前 12 行
    T = np.concatenate([T, np.roll(T, 16, axis=1), np.roll(T, 32, axis=1)], axis=0) # 构建全 36 行

    # compute convolution
    output = (T @ padding_img.ravel()).reshape(6, 6)

    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """

    # # zero padding (0-padding 后的结果应和上一题一致, 经测试的确一致)
    # img = padding(img, 1, "zeroPadding")

    kerH, kerW = kernel.shape                       # 卷积核尺寸
    imgH, imgW = img.shape                          # 原始图片尺寸
    outH, outW = (imgH - kerH + 1, imgW - kerW + 1) # 输出矩阵的尺寸

    # 先切成 outH 个 kerH × imgW 子图, 并旋转成 imgW × kerH 方便下一次切割
    index1 = np.arange(kerH).reshape(1,-1) + np.arange(outH).reshape(-1,1)  # index1 = [[0,1,2],
                                                                            #           [1,2,3],
                                                                            #           [2,3,4], ... ]
    temp1 = img[index1]                     # -> (outH, kerH, imgW)
    temp1 = np.transpose(temp1, (0,2,1))    # -> (outH, imgW, kerH)

    # 将每个 imgW × kerH 子图都再切成 outW 个 kerW × kerH 子图, 并旋转成 kerH × kerW 方便之后和 kernel 逐元素相乘.
    index2 = np.arange(kerW).reshape(1,-1) + np.arange(outW).reshape(-1,1)  # index2 = [[0,1,2],
                                                                            #           [1,2,3],
                                                                            #           [2,3,4], ... ]
    temp2 = temp1[:, index2]                # -> (outH, outW, kerW, kerH)
    temp2 = np.transpose(temp2, (0,1,3,2))  # -> (outH, outW, kerH, kerW)

    # 现在 temp2 可以理解为是一个 (outH, outW) 二维矩阵, 其每个元素是一个 (kerH, kerW) 矩阵.
    # 这样直接逐元素和 kernel 相乘, 再相加即可
    output = np.sum(temp2 * kernel, axis=(2,3))

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

# # 更广泛的 Gaussian_filter. JUST FOR TEST!!
# def Gaussian_filter(img, sigma=5, ksize=9):
#     assert ksize % 2 == 1
#     center = (ksize - 1) // 2
#     kernel = np.zeros((ksize, ksize))
#     for i in range(kernel.shape[0]):
#         for j in range(kernel.shape[1]):
#             kernel[i, j] = np.exp(-((i-center)**2 + (j-center)**2) / (2 * sigma**2))
#     gaussian_kernel = kernel / kernel.sum()
#     padding_img = padding(img, center, "replicatePadding")
#     output = convolve(padding_img, gaussian_kernel)
#     return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":

    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png") / 255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x * 255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y * 255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur * 255)
