"""Count the CCSD response contractions on separated waters and interacting H2 chains.

Run with a Psi4 build of this branch. Counts measure the new response pass.
"""
import psi4


def counts(system, size):
    atoms = []
    for i in range(size):
        if system == "water":
            x = 17 * i
            atoms.extend([f"O {x} 0 0", f"H {x} -.758 .586", f"H {x} .758 .586"])
        else:
            x = 3.5 * i
            atoms.extend([f"H {x} 0 -.37", f"H {x} 0 .37"])
    mol = psi4.geometry("\n".join(["0 1", *atoms, "units angstrom", "symmetry c1", "no_com", "no_reorient"]))
    _, wfn = psi4.energy("dlpno-ccsd(t)", molecule=mol, return_wfn=True)
    return tuple(int(wfn.variable("DLPNO CCSD RESPONSE " + key))
                 for key in ("ORDERED PAIRS", "OVERLAP BLOCKS", "KL CANDIDATES"))


if __name__ == "__main__":
    psi4.set_memory("1.5 GB")
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis="cc-pvdz", reference="rhf", freeze_core=True, guess="sad",
                          scf_type="df", cc_type="df", df_basis_scf="cc-pvdz-jkfit", df_basis_cc="cc-pvdz-ri",
                          e_convergence=1e-11, d_convergence=1e-11, r_convergence=1e-10, dlpno_maxiter=120,
                          dlpno_gradient_model=True, dlpno_local_orbitals="BOYS", low_memory_overlap=True,
                          t_cut_pairs_mp2=1e-6, t_cut_pairs=1e-5, t_cut_tno=1e-5, min_tnos=3, dlpno_ao_ints_tol=1e-14))
    print("system, molecules, ordered pairs, overlap blocks, kl candidates")
    for system, sizes in (("water", (1, 2, 3)), ("hydrogen", (4, 8, 12))):
        for size in sizes:
            print(system, size, *counts(system, size), flush=True)
