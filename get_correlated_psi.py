#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:20:54 2026

@author: cesar
"""

import numpy as np
from pyscf import gto, scf, fci


def normalize(psi):
    return psi / np.linalg.norm(psi)


def make_mol(atom, basis, spin=0, unit="Bohr"):
    return gto.M(
        atom=atom,
        basis=basis,
        spin=spin,
        unit=unit,
        verbose=0,
    )


def hf_orbitals_on_grid(mol, positions):
    mf = scf.RHF(mol).run()
    ao = mol.eval_gto("GTOval_sph", positions)
    return mf, ao @ mf.mo_coeff


def two_electron_fci_wavefunction(mol, positions):
    mf, mo = hf_orbitals_on_grid(mol, positions)
    energy, ci = fci.FCI(mol, mf.mo_coeff).kernel()

    # For two electrons: ci[p, q] in the alpha/beta MO basis.
    ci = np.asarray(ci)
    psi = np.einsum("pi,ij,qj->pq", mo, ci, mo, optimize=True)

    print(f"FCI ground-state energy: {energy:.8f} Ha")
    return normalize(psi.ravel())


def two_electron_rhf_wavefunction(mol, positions):
    mf, mo = hf_orbitals_on_grid(mol, positions)
    occupied = np.flatnonzero(mf.mo_occ > 0)

    if len(occupied) != 1:
        raise ValueError(
            f"Expected one doubly occupied RHF orbital, got {len(occupied)}."
        )

    orbital = mo[:, occupied[0]]
    return normalize(np.outer(orbital, orbital).ravel())


def get_psi_correlated_He(positions, basis="cc-pVTZ"):
    mol = make_mol("He 0 0 0", basis)
    return two_electron_fci_wavefunction(mol, positions)


def get_psi_HF_He(positions, basis="6-31G"):
    mol = make_mol("He 0 0 0", basis)
    return two_electron_rhf_wavefunction(mol, positions)


def get_psi_HF_H2(positions, dist=1.8897259886, basis="6-31G"):
    mol = make_mol(f"H 0 0 0; H 0 0 {dist}", basis)
    return two_electron_rhf_wavefunction(mol, positions)


def get_psi_correlated_H2(positions, dist, basis="cc-pVDZ"):
    mol = make_mol(f"H 0 0 0; H 0 0 {dist}", basis)
    return two_electron_fci_wavefunction(mol, positions)


def get_psi_H(positions, basis="cc-pV5Z"):
    mol = make_mol("H 0 0 0", basis, spin=1)
    mf, mo = hf_orbitals_on_grid(mol, positions)
    orbital = mo[:, np.flatnonzero(mf.mo_occ > 0)[0]]
    return normalize(orbital)