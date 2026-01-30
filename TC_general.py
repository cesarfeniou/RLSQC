import numpy as np
from scipy import sparse as sparse
from scipy import special as sp
from scipy.spatial import distance
from matplotlib import pyplot as plt
from voronoi import voronoi_L, voronoi_D, voronoi_Dee  # type: ignore
from grid_generator import GridSetup  # type: ignore

def dg1(Z,mu,r):
    return -(Z-sp.erf(mu*r))

def dg2(mu,r):
    return 2*(mu/np.sqrt(np.pi))*np.exp(-(mu*r)**2)

def diag_op(f,Z,mu,grid,k):
    d = np.linalg.norm(grid.positions - np.array(grid.alpha[k]), axis=1)
    return sparse.diags(f(Z,mu,d))

def Vne(Z,mu,grid):
    transc = lambda Z,m,d: sp.erf(m*d)/d+ dg2(m,d)/2 + (dg1(Z,m,d)**2)/2
    Vtrans = 0
    for k in range(len(grid.alpha)):
        Vtrans += diag_op(transc,Z,mu,grid,k)
    return Vtrans

def r12(grid):
    return distance.cdist(grid.positions,grid.positions,'euclidean').flatten()

def laplacian(Z, mu, grid):
    nherm = 0
    for k in range(len(alpha)):
        nherm += 2*diag_op(dg1,Z,mu,grid,k)@voronoi_D(grid.positions,alpha[k]) 
    return voronoi_L(grid.positions) + nherm
        
def ham1e(Z,mu,grid):
    return -0.5*laplacian(Z, mu, grid) - Vne(Z, mu, grid)

def erf_over_x(x):
    x = np.asarray(x)
    result = np.empty_like(x, dtype=np.float64)
    nonzero = x != 0
    result[nonzero] = sp.erf(x[nonzero]) / x[nonzero]
    result[~nonzero] = 2 / np.sqrt(np.pi)
    return result

def Vee(mu,grid):
    d = r12(grid)
    return sparse.diags(mu*erf_over_x(mu*d)+ dg2(mu,d)/2 + (dg1(1,mu,d)**2)/2)

def nhermee(mu,grid):
    d = r12(grid)
    return -sparse.diags(dg1(1,mu,d))@voronoi_Dee(grid.positions)

def hamiltonian(Z,m,grid, N):
    H = ham1e(Z,m,grid)
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
        return sparse.csr_matrix(Ht) + Vee(m,grid) + nhermee(mu,grid)
                         
#_________________________________________________Parameters and grid_________________________________
Z=1    #CHANGE ATOMIC NUMBER

Ne = 1   #CHANGE NUMBER OF ELECTRONS

mu = 1

Nr = 32
Nang = 11

# CHANGE NUMBER AND POSITIONS OF NUCLEI
alpha = [(0,0,0)]  #H
#alpha = [(0,0,0), (1.27,0,0)]  #H2+

grid = GridSetup(Nr, Nang, alpha)
grid.plot_grid()

#_____________________________________________________Resolution and plot________________________________________
H = hamiltonian(Z,mu,grid,Ne)

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
