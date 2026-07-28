# RLSQC

This repository implements the electronic Hamiltonian of simple molecules using the Voronoi finite-volume discretization scheme.

## File Descriptions

### `grid_generator.py`

Constructs a **3D real-space grid** using:

- Becke partitioning along the radial direction
- Lebedev quadrature for angular coordinates

The total number of grid points is controlled by:
- N_r: number of radial points
- N_{ang}: number of angular points

IMPORTANT: Uncomment the required lines for H2+ to get a non-overlapping grid for better results

---

### `voronoi.py`

Implements the **Voronoi finite-volume discretization** of differential operators.

Includes matrix representations of:
- The Laplacian (with symmetrized transformation)
- First derivatives with respect to the nuclear–electron distance
- First derivatives with respect to the electron–electron distance

The first-derivative operators arise from the **transcorrelation transformation** (see appendix A of the article).

---

### `NTC_general.py`

Constructs the **non-transcorrelated Hamiltonian** by assembling the matrix terms. Diagonalizes the Hamiltonian, provides an estimate of the **ground-state energy** and plots the corresponding eigenstate in the radial direction.

---

### `TC_general.py`

Same as `NTC_general.py` but for the **transcorrelated Hamiltonian**

---

### Note:
Larger grids were taken for more accurate results and the corresponding Hamiltonian were diagonalized using a Davidson solver (check part 5 of the article).
