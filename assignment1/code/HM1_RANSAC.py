import numpy as np
from utils import draw_save_plane_with_points

if __name__ == "__main__":
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")

    # RANSAC
    # we recommend you to formulate the plane function as:  A*x+B*y+C*z+D=0

    # more than 99.9% probability at least one hypothesis does not contain any outliers
    sample_time = 12    # 大于 ln(0.001) / ln(1-(100*99*98)/(130*129*128)) 即可
    distance_threshold = 0.05

    # sample points group: 三个点为一组, 抽 12 组.
    # 理论上, 每一组的三个点是不能重复的, 而组之间则应该是允许重复的.
    # 但我不知道该怎么不用for循环实现这种抽法.. 所以干脆就强行让组之间也不允许重复!
    # 这也没啥问题, 因为:
    # 首先, 从 130 个点中抽 36 个互异的点是可以做到的;
    # 此外, 如果这里是互异的, 那么 12 次是能保证概率 > 99.9% 的.
    #   因为可以试想, 在这种情况下如果某组失败, 就说明这组的三个点中至少有一个点是 outlier, 而好消息是这个点之后不会再被抽到! 这样就大大提高了*后续组*中所有点都是 inlier 的概率! (概率提高的原因: 1/3 > 30/130)
    np.random.seed(1900017785)
    groups = noise_points[
        np.random.choice(range(noise_points.shape[0]), sample_time * 3, replace=False)
    ].reshape(-1,3,3)

    # estimate the plane with sampled points group
    # 这里假设选用 Ax + By + Cz + 1 = 0, 因此每个平面用 (A, B, C) 表示
    ABCs = np.linalg.inv(groups) @ np.array([-1, -1, -1]).reshape(3, 1)

    # evaluate inliers (with point-to-plane distance < distance_threshold)
    # 点到平面距离公式: d = |Ax + By + Cz + D| / √(A²+B²+C²)
    ABCs = ABCs.reshape(-1, 1, 3)
    points = noise_points.reshape(1, -1, 3)

    distances = np.abs(np.sum(ABCs * points, axis=2) + 1) / np.sqrt(np.sum(ABCs**2, axis=2)) # shape=(12, 130), i.e. (#sample, #points)
    n_inliers = np.sum(distances < distance_threshold, axis=0)
    best_ABC_index = np.argmax(n_inliers)
    best_ABC = ABCs[best_ABC_index].reshape(3)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    inliers = noise_points[distances[best_ABC_index] < distance_threshold]
    lse = -np.sum(np.linalg.inv(inliers.T @ inliers) @ inliers.T, axis=1)   # 最外层的 -np.sum() 其实是 @ [-1,...,-1]' 的等价写法

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0
    pf = [*lse, 1]  # D = 1
    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", np.array(pf))
