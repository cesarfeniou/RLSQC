import numpy as np
from scipy.spatial import Voronoi

def signed_area_triangle(p1, p2, q):
    v1 = np.subtract(p1,q)
    v2 = np.subtract(p2,q)
    return 0.5 * np.linalg.norm(np.cross(v1, v2))

def mat_el(area,u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    u = np.array(u)
    v = np.array(v)
    return area*np.dot(u, v) / (norm_u * norm_v)

def voronoi_L(points):
    n = len(points)
    vor = Voronoi(points)
    L = np.zeros((n, n))
    v = np.zeros(n)
    for ridge, verts in zip(vor.ridge_points, vor.ridge_vertices):
        i, j = ridge
        polygon = vor.vertices[verts]
        q = np.add(points[i],points[j]) / 2
        area = 0.0
        M = len(polygon)
        for k in range(M):
            p1 = polygon[k]
            p2 = polygon[(k + 1) % M]
            area += signed_area_triangle(p1, p2, q)
        hij = np.linalg.norm(np.subtract(points[j],points[i]))
        L[i, j] = area / hij
        L[j, i] = area / hij
        L[i, i] -= area / hij
        L[j, j] -= area / hij
        v[i] += hij*area
        v[j] += hij*area
    return np.diag(6/v) @ L

def voronoi_L_sym(points):
    n = len(points)
    vor = Voronoi(points)
    L = np.zeros((n, n))
    v = np.zeros(n)
    for ridge, verts in zip(vor.ridge_points, vor.ridge_vertices):
        i, j = ridge 
        polygon = vor.vertices[verts]
        q = np.add(points[i],points[j]) / 2
        area = 0.0
        M = len(polygon)
        for k in range(M):
            p1 = polygon[k]
            p2 = polygon[(k + 1) % M]
            area += signed_area_triangle(p1, p2, q)
        hij = np.linalg.norm(np.subtract(points[j],points[i]))
        L[i, j] = area / hij
        L[j, i] = area / hij
        L[i, i] -= area / hij
        L[j, j] -= area / hij
        v[i] += hij*area
        v[j] += hij*area
    return np.diag(np.sqrt(v)) @ (np.diag(6/v) @ L) @ np.diag(1/np.sqrt(v))

def voronoi_D(points,alpha):
    n = len(points)
    vor = Voronoi(points)
    L = np.zeros((n, n))
    v = np.zeros(n)
    for ridge, verts in zip(vor.ridge_points, vor.ridge_vertices):
        i, j = ridge 
        polygon = vor.vertices[verts]
        q = np.add(points[i],points[j]) / 2
        area = 0.0
        M = len(polygon)
        for k in range(M):
            p1 = polygon[k]
            p2 = polygon[(k + 1) % M]
            area += signed_area_triangle(p1, p2, q)
        hij = np.linalg.norm(np.subtract(points[j],points[i]))
        L[i, j] = (area / hij)*np.dot(np.subtract(points[j],points[i]),np.subtract(points[i],alpha)/np.linalg.norm(np.subtract(points[i],alpha)))
        L[j, i] = (area / hij)*np.dot(np.subtract(points[i],points[j]),np.subtract(points[j],alpha)/np.linalg.norm(np.subtract(points[j],alpha)))
        v[i] += hij*area
        v[j] += hij*area
    return np.diag(3/v) @ L

from scipy import sparse as sparse

def voronoi_Dee(points):
    N = len(points)
    vor = Voronoi(points)
    rows_L1, cols_L1, data_L1 = [], [], []
    rows_L2, cols_L2, data_L2 = [], [], []
    v1 = np.zeros(N**2)
    v2 = np.zeros(N**2)
    for ridge1, verts1 in zip(vor.ridge_points, vor.ridge_vertices):
        m, n = ridge1
        polygon1 = vor.vertices[verts1]
        rho1 = (points[m] + points[n]) / 2
        area1 = 0.0
        M1 = len(polygon1)
        for i1 in range(M1):
            pol1 = polygon1[i1]
            poll1 = polygon1[(i1 + 1) % M1]
            area1 += signed_area_triangle(pol1, poll1, rho1)
        h1 = np.linalg.norm(points[n] - points[m])
        for ridge2, verts2 in zip(vor.ridge_points, vor.ridge_vertices):
            p, q = ridge2
            polygon2 = vor.vertices[verts2]
            rho2 = (points[p] + points[q]) / 2
            area2 = 0.0
            M2 = len(polygon2)
            for i2 in range(M2):
                pol2 = polygon2[i2]
                poll2 = polygon2[(i2 + 1) % M2]
                area2 += signed_area_triangle(pol2, poll2, rho2)
            h2 = np.linalg.norm(points[q] - points[p])
            idx1 = m*N + p
            idx2 = n*N + q
            val1_1 = mat_el(area1, points[n] - points[m], points[m] - points[p])
            val1_2 = mat_el(area1, points[m] - points[n], points[n] - points[q])
            rows_L1.extend([idx1, idx2])
            cols_L1.extend([idx2, idx1])
            data_L1.extend([val1_1, val1_2])
            v1[idx1] += h1 * area1
            v1[idx2] += h1 * area1
            val2_1 = mat_el(area2, points[q] - points[p], points[m] - points[p])
            val2_2 = mat_el(area2, points[p] - points[q], points[n] - points[q])
            rows_L2.extend([idx1, idx2])
            cols_L2.extend([idx2, idx1])
            data_L2.extend([val2_1, val2_2])
            v2[idx1] += h2 * area2
            v2[idx2] += h2 * area2
    L1 = sparse.csr_matrix((data_L1, (rows_L1, cols_L1)), shape=(N**2, N**2))
    L2 = sparse.csr_matrix((data_L2, (rows_L2, cols_L2)), shape=(N**2, N**2))
    Dee = sparse.diags(3 / v1) @ L1 - sparse.diags(3 / v2) @ L2
    Dee.data[np.isinf(Dee.data)] = 0
    Dee.data[np.isnan(Dee.data)] = 0
    return Dee

Nr = 5
Nang = 5
alpha = [(0,0,0)]  
from grid_generator import GridSetup #type:ignore
grid = GridSetup(Nr, Nang, alpha)
#grid.plot_grid()

import time
start = time.time()
print(np.isinf(voronoi_Dee(grid.positions).data).any())
end=time.time()
print(end-start)
