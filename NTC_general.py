import numpy as np
from scipy import sparse as sparse
from scipy import special as sp
from scipy.spatial import distance
from matplotlib import pyplot as plt
from voronoi import voronoi_L_sym  # type: ignore
from grid_generator import GridSetup  # type: ignore

def diag_op(f,Z,grid,k):
    d = np.linalg.norm(grid.positions - np.array(grid.alpha[k]), axis=1)
    return sparse.diags(f(Z,d))

def Vne(Z,grid):
    transc = lambda Z,d: Z/d
    Vtrans = 0
    for k in range(len(grid.alpha)):
        Vtrans += diag_op(transc,Z,grid,k)
    return Vtrans

def r12(grid):
    return distance.cdist(grid.positions,grid.positions,'euclidean').flatten()

def laplacian(Z,grid):
    return voronoi_L_sym(grid.positions)
        
def ham1e(Z,grid):
    return -0.5*laplacian(Z,grid) - Vne(Z, grid)

def erf_over_x(x):
    x = np.asarray(x)
    result = np.empty_like(x, dtype=np.float64)
    nonzero = x != 0
    result[nonzero] = sp.erf(x[nonzero]) / x[nonzero]
    result[~nonzero] = 2 / np.sqrt(np.pi)
    return result

def Vee(grid):
    d = r12(grid)
    return sparse.diags(1/d)

def hamiltonian(Z,grid, N):
    H = ham1e(Z,grid)
    if N == 1:
        return sparse.csr_matrix(H)
    else:
        Ht = sparse.csr_matrix(((H.shape[0])**N, (H.shape[0])**N))
        for i in range(N):
            ops = [sparse.identity(H.shape[0])] * N
            ops[i] = H
            H_local = ops[0]
            for op in ops[1:]:
                H_local = sparse.kron(H_local, op)
            Ht += H_local
        return sparse.csr_matrix(Ht) + Vee(grid)
                         
#_________________________________________________Parameters and grid_________________________________
Z=1    #CHANGE ATOMIC NUMBER

Ne = 1   #CHANGE NUMBER OF ELECTRONS

Nr = 32
Nang = 11

# CHANGE NUMBER AND POSITIONS OF NUCLEI
alpha = [(0,0,0)]  #H
#alpha = [(0,0,0), (1.27,0,0)]  #H2+

grid = GridSetup(Nr, Nang, alpha)
grid.plot_grid()

#_____________________________________________________Resolution and plot________________________________________

H = hamiltonian(Z,grid,Ne)

eigenvalues, eigenvectors = np.linalg.eig(H.toarray())

idx = np.argsort(eigenvalues.real)[:5]  
eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
print(eigenvalues[0])

X = grid.x
Y = grid.y
Z = grid.z
positions = np.stack([X, Y, Z], axis=-1)
R = np.sqrt(X**2 + Y**2 + Z**2)
plt.plot(X, eigenvectors[:,0].real, linewidth=0, marker='o', markersize=1)
plt.show()