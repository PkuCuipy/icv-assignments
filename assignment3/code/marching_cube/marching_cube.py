import numpy as np
from lookup_table import CaseNum2EdgeOffset, getCaseNum
import trimesh
import os
import time
from tqdm import trange


def marching_cube(isovalue, grid):

    # grid[x][y][z] < threshold 意味着 (x, y, z) 是物体内部的点
    grid = np.array(grid)

    t1 = time.time()
    # -------------------TODO------------------
    # - compute vertices and faces.
    # - return:
    #       vertices: [N, 3]
    #       faces: [M, 3], e.g. np.array([[0,1,2]]) means a triangle composed of vertices[0], vertices[1] and vertices[2]
    # - for-loop is allowed to reduce difficulty
    # -------------------TODO------------------
    triangles = []  # List[List[Tuple[int; 3]; 3]; N]
    for x in trange(grid.shape[0] - 1):
        for y in range(grid.shape[1] - 1):
            for z in range(grid.shape[2] - 1):
                xyz = np.array([x, y, z])
                case_nums = getCaseNum(x, y, z, isovalue, grid)
                edges = case_nums[case_nums != -1].reshape(-1, 3)   # -> (N, 3), 其中 N 是三角形数目, 3 是三角形的三条边的编号 (0..12)
                edges = CaseNum2EdgeOffset[edges]                   # -> (N, 3, 6), 其中 N 是三角形数目, 3 是三角形的三条边, 6 是每条边的两个顶点的 (x,y,z, x,y,z)
                for triangle in edges:
                    three_points = []
                    for edge in triangle:
                        P1 = xyz + edge[:3]
                        P2 = xyz + edge[3:]
                        V1 = grid[tuple(P1)]
                        V2 = grid[tuple(P2)]
                        P = P1 + ((isovalue - V1) / (V2 - V1)) * (P2 - P1)
                        three_points.append(tuple(P))
                    triangles.append(three_points)

    vertex_array = list(set(tup for tri in triangles for tup in tri))
    vtx_to_idx = dict(zip(vertex_array, range(len(vertex_array))))
    face_array = [tuple(vtx_to_idx[vtx] for vtx in tri) for tri in triangles]
    t2 = time.time()
    print("\nTime taken by algorithm\n" + '-' * 40 + "\n{} s".format(t2 - t1))

    return np.array(vertex_array), np.array(face_array)


# reconstruct these two animals
shape_name_lst = ['spot', 'bob']
for shape_name in shape_name_lst:
    data = np.load(os.path.join('data', shape_name + '_cell.npy'))
    verts, faces = marching_cube(0, data)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh_txt = trimesh.exchange.obj.export_obj(mesh)
    with open(os.path.join('../results', shape_name + '.obj'), "w") as fp:
        fp.write(mesh_txt)