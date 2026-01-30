from pylebedev import PyLebedev # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

class GridSetup:

    def __init__(self, Nr, Nang, alpha=[(0,0,0)], m=1, rmin=1e-4, rmax=20) -> None:
        
        self.alpha = alpha
        u = np.linspace((1-np.exp(-1))**(1/m),(1-np.exp(-rmin))**(1/m),Nr)[::-1]
        self.r = -rmax*np.log(1-u**m)
        
        leblib = PyLebedev()
        R, w = leblib.get_points_and_weights(Nang)

        x, y, z = R[:,0], R[:,1], R[:,2]
        unit_vecs = np.stack([x, y, z], axis=1)
        grid_points = self.r[:, np.newaxis, np.newaxis] * unit_vecs[np.newaxis, :, :]
        grid_points_flat = grid_points.reshape(-1, 3)
        x0,y0,z0 = grid_points_flat[:, 0], grid_points_flat[:, 1], grid_points_flat[:, 2]
        """ UNCOMMENT FOR H2+
        mask = x0<1.27/2
        self.x,self.y,self.z = x0[mask],y0[mask],z0[mask]"""
        for a in alpha:
            xa, ya, za = a[0], a[1], a[2]
            if xa == ya == za == 0:
                continue
            """ UNCOMMENT FOR H2+
            x1 = x0+xa
            y1 = y0+ya
            z1 = z0+za
            mask = x1>1.27/2
            self.x = np.concatenate([self.x, x1[mask]])
            self.y = np.concatenate([self.y, y1[mask]])
            self.z = np.concatenate([self.z, z1[mask]])
            """
            """ Comment the next three lines for H2+"""
            self.x = np.concatenate([self.x, grid_points_flat[:, 0] + xa])
            self.y = np.concatenate([self.y, grid_points_flat[:, 1] + ya])
            self.z = np.concatenate([self.z, grid_points_flat[:, 2] + za])
        print('Total number of grid points = ', len(self.x))
        print('Number of angular points = ', len(self.x)/Nr)
        self.positions = np.stack([self.x, self.y, self.z], axis=-1)

        self.r, self.t, self.p = self.cartesian_to_spherical(self.x, self.y, self.z)
        
    def plot_grid(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    
    def cartesian_to_spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / r, -1.0, 1.0)) 
        phi = np.arctan2(y, x) % (2 * np.pi)
        return r, theta, phi