# export_data_davidson.py

import os
import numpy as np
from scipy.sparse import csr_matrix


def export_sparse_matrix_structure(H: csr_matrix, output_dir="."):
    H = H.tocsr()

    assert H.shape[0] == H.shape[1], "Matrix must be square"

    N = H.shape[0]
    row_nonzero_counts = np.diff(H.indptr)
    Kmax = row_nonzero_counts.max()

    indices_matrix = np.full((N, Kmax + 1), -1, dtype=int)
    values_matrix = np.zeros((N, Kmax), dtype=float)

    for i in range(N):
        start = H.indptr[i]
        end = H.indptr[i + 1]

        cols = H.indices[start:end]
        vals = H.data[start:end]

        k = len(cols)
        indices_matrix[i, 0] = k
        indices_matrix[i, 1:k + 1] = cols + 1  # Fortran / 1-based indexing
        values_matrix[i, :k] = vals

    np.savetxt(os.path.join(output_dir, "indices.txt"), indices_matrix, fmt="%d")
    np.savetxt(os.path.join(output_dir, "values.txt"), values_matrix, fmt="%.8f")

    with open(os.path.join(output_dir, "sparsity.txt"), "w") as f:
        f.write(f"{Kmax}\n")


def export_grid(grid, output_dir="."):
    positions = grid.positions

    np.savetxt(
        os.path.join(output_dir, "xyz.txt"),
        positions,
        fmt="%.6f",
        delimiter=" ",
        comments="",
    )

    with open(os.path.join(output_dir, "dim.txt"), "w") as f:
        f.write(str(len(positions)))


def export_tensor_indices(grid, Ne, output_dir="."):
    if Ne != 2:
        return

    N = len(grid.positions)
    ij_pairs = [(i, j) for i in range(1, N + 1) for j in range(1, N + 1)]

    np.savetxt(
        os.path.join(output_dir, "tensor.txt"),
        ij_pairs,
        fmt="%d",
        delimiter=" ",
        comments="",
    )


def export_initial_guess(psi_guess, output_dir="."):
    psi_guess = np.asarray(psi_guess, dtype=float)
    psi_guess /= np.linalg.norm(psi_guess)

    np.savetxt(
        os.path.join(output_dir, "psi_guess.txt"),
        psi_guess,
        fmt="%.6f",
        delimiter=",",
        comments="",
    )


def export_davidson_input(grid, H, psi_guess, Ne, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    export_grid(grid, output_dir)
    export_tensor_indices(grid, Ne, output_dir)
    export_sparse_matrix_structure(H, output_dir)
    export_initial_guess(psi_guess, output_dir)
