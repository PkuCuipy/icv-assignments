import numpy as np
import trimesh
import tqdm

def debug_draw_3D_scatter(points_3d: np.ndarray):
    """ 绘制三维散点图 """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    plt.show()

def uniform_sampling_from_mesh(vertices, faces, n_sample):
    # -------- TODO -----------
    # 1. compute area of each triangles
    # 2. compute probability of each triangles from areas
    # 3. sample N faces according to the probability
    # 4. for each face, sample 1 point
    # ** Note that FOR-LOOP is NOT allowed! **
    # -------- TODO -----------

    triangles = vertices[faces]  # -> (n_triangles, 3, 3)

    # 1. 使用海伦公式计算每个三角形的面积 (Ref: https://zh.m.wikipedia.org/zh-hk/海伦公式)
    a = np.sqrt(np.sum(np.square(triangles[:, 0] - triangles[:, 1]), axis=1))
    b = np.sqrt(np.sum(np.square(triangles[:, 1] - triangles[:, 2]), axis=1))
    c = np.sqrt(np.sum(np.square(triangles[:, 2] - triangles[:, 0]), axis=1))
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    # 2. 计算每个三角形的面积占总表面积的比例
    prob = area / area.sum()

    # 3. 根据占比抽 n_sample 个三角形
    rand_indices = np.random.choice(range(triangles.shape[0]), size=n_sample, p=prob)
    rand_triangles = triangles[rand_indices]

    # 4. 在每个三角形上抽一个点 (Ref: https://blogs.sas.com/content/iml/2020/10/19/random-points-in-triangle.html)
    P1 = rand_triangles[:, 0]
    P2 = rand_triangles[:, 1]
    P3 = rand_triangles[:, 2]
    a = P2 - P1
    b = P3 - P1
    u1 = np.random.rand(rand_triangles.shape[0])
    u2 = np.random.rand(rand_triangles.shape[0])
    temp_indices = u1 + u2 > 1
    u1[temp_indices] = 1 - u1[temp_indices]
    u2[temp_indices] = 1 - u2[temp_indices]
    w = u1[:, np.newaxis] * a + u2[:, np.newaxis] * b
    uniform_pc = w + P1

    return area, prob, uniform_pc

def farthest_point_sampling(point_cloud, n_sample):
    # -------- TODO -----------
    # FOR LOOP is allowed here.
    # -------- TODO -----------

    remains = np.copy(point_cloud)      # 最后 i 个是无效的
    results = np.zeros((n_sample, 3))   # 前 i 个是有效的

    # 不妨以最后一个点作为初始点
    results[0] = remains[-1]

    for i in tqdm.trange(1, n_sample):
        U = remains[:-i]    # 尚未被加入 results 的点
        S = results[:i]     # 已决定加入 results 的点
        # 从 S 中选择一个点 (一行) s, 满足 s 到 U 的 [最小距离] 最大
        U = U.reshape(U.shape[0], 1, 3)
        S = S.reshape(1, S.shape[0], 3)
        dist = np.sum(np.square(S - U), axis=-1)
        min_dist = np.min(dist, axis=-1)    # [最小距离]
        max_idx = np.argmax(min_dist)       # 找出 S 中与 U 的 [最小距离] 最大的点 s
        # 下面将 remains[max_idx] 加入到 results 中
        results[i] = remains[max_idx]       # results.append(s)
        remains[max_idx] = remains[-i - 1]  # remains.remove(max_idx)

    return results

def chamfer_distance(S1: np.ndarray, S2: np.ndarray):
    S1 = S1.reshape((S1.shape[0], 1, 3))
    S2 = S2.reshape((1, S2.shape[0], 3))
    dist_mat = np.sqrt(np.sum((S1 - S2) ** 2, axis=-1))
    chf_dist = np.mean(np.min(dist_mat, axis=0)) + np.mean(np.min(dist_mat, axis=1))
    return chf_dist

# ================================================================================#
# task 1: uniform sampling

mesh = trimesh.load('./spot.obj')
sample_num = 512
area, prob, uniform_pc = uniform_sampling_from_mesh(mesh.vertices, mesh.faces, sample_num)
vertices = mesh.vertices
faces = mesh.faces

# Visualization. For you to check your code
np.savetxt('uniform_sampling_vis.txt', uniform_pc)
debug_draw_3D_scatter(uniform_pc)

# For submission
save_dict = {'area': area, 'prob': prob, 'pc': uniform_pc}
np.save('../results/uniform_sampling_results', save_dict)

# ================================================================================#
# task 2: FPS

init_sample_num = 2000
final_sample_num = 512
_, _, tmp_pc = uniform_sampling_from_mesh(mesh.vertices, mesh.faces, init_sample_num)
fps_pc = farthest_point_sampling(tmp_pc, final_sample_num)
# debug_draw_3D_scatter(fps_pc)

# Visualization. For you to check your code
np.savetxt('fps_vis.txt', fps_pc)

# For submission
np.save('../results/fps_results', fps_pc)


# # ================================================================================#
# task 3: metrics

from earthmover.earthmover import earthmover_distance   # EMD may be very slow (1~2mins)
# -----------TODO---------------
# compute chamfer distance and EMD for two point clouds sampled by uniform sampling and FPS.
# sample and compute CD and EMD again. repeat for five times.
# save the mean and var.
# -----------TODO---------------

CDs = []
EMDs = []

for _ in range(5):
    # 分别用 uniform 方法和 fps 方法采样
    uniform_pc = uniform_sampling_from_mesh(mesh.vertices, mesh.faces, sample_num)[2]
    tmp_pc = uniform_sampling_from_mesh(mesh.vertices, mesh.faces, init_sample_num)[2]
    fps_pc = farthest_point_sampling(tmp_pc, final_sample_num)
    # 计算两个采样之间的距离
    to_list_of_tuple = lambda matrix: [tuple(row) for row in matrix]
    CDs.append(chamfer_distance(uniform_pc, fps_pc))
    EMDs.append(earthmover_distance(to_list_of_tuple(uniform_pc), to_list_of_tuple(fps_pc)))

CD_mean = np.mean(CDs)
CD_var = np.var(CDs)
EMD_mean = np.mean(EMDs)
EMD_var = np.var(EMDs)

print({'CD_mean': CD_mean, 'CD_var': CD_var, 'EMD_mean': EMD_mean, 'EMD_var': EMD_var})
print(f"[σ/μ](CD) = {np.sqrt(CD_var) / CD_mean}")
print(f"[σ/μ](EMD) = {np.sqrt(EMD_var) / EMD_mean}")
print(f"会发现 [σ/μ](EMD) > [σ/μ](CD), 因此 EMD 的确对 sampling 更 sensitive.")

# For submission
np.save('../results/metrics', {'CD_mean': CD_mean, 'CD_var': CD_var, 'EMD_mean': EMD_mean, 'EMD_var': EMD_var})
