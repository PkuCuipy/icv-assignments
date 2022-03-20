import numpy as np
from utils import read_img, draw_corner
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from PIL import Image

def corner_response_function(I_xx, I_yy, I_xy, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            I_xx: array(float)
            I_yy: array(float) 
            I_yy: array(float) 
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # for detials of corner_response_function, please refer to the slides.

    Theta = I_xx * I_yy - I_xy**2 - alpha * (I_xx + I_yy)**2

    index_of_rows, index_of_cols = np.where(Theta > threshold)
    theta = Theta[index_of_rows, index_of_cols]
    corner_list = list(zip(index_of_rows, index_of_cols, theta))

    return corner_list  # the corners in corner_list: a tuple of (index of rows, index of cols, theta)


if __name__ == "__main__":

    # Load the input images
    input_img = read_img("hand_writting.png") / 255.

    I_x = Sobel_filter_x(input_img)
    I_y = Sobel_filter_y(input_img)

    I_xx = I_x ** 2
    I_yy = I_y ** 2
    I_xy = I_x * I_y

    I_xx = Gaussian_filter(I_xx)
    I_yy = Gaussian_filter(I_yy)
    I_xy = Gaussian_filter(I_xy)

    # you can adjust the parameters to fit your own implementation
    window_size = 5
    alpha = 0.04
    threshold = 0.01

    corner_list = corner_response_function(I_xx, I_yy, I_xy, window_size, alpha, threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key=lambda x: x[2], reverse=True)
    NML_selected = []
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted:
        for j in NML_selected:
            if (abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis):
                break
        else:
            NML_selected.append(i[:-1])

    # save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
