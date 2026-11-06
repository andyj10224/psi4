"""Fixed-frame water/cc-pVDZ Hessian reference; run as a standalone Psi4 script."""
import numpy as np
import psi4

STEPS = (0.008, 0.004, 0.002)  # bohr
# Atom order O, H1, H2. Keep H1 along +z and put H2 in the yz plane.
angle = np.deg2rad(104.4776)
length = 0.958 / psi4.constants.bohr2angstroms
BASE = length*np.array([[0., 0., 0.], [0., 0., 1.], [0., np.sin(angle), np.cos(angle)]])
STRETCH = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]])
PROBE = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])


def molecule(coords):
    rows = [f"{a} {x:.16f} {y:.16f} {z:.16f}" for a, (x,y,z) in zip(("O","H","H"), coords)]
    mol = psi4.geometry("\n".join(rows + ["units bohr", "symmetry c1", "no_reorient", "no_com"]))
    np.testing.assert_allclose(mol.geometry().np, coords, rtol=0, atol=1.e-13)
    return mol


def gradient(method, coords):
    psi4.core.clean()
    g, w = psi4.gradient(method, molecule=molecule(coords), dertype=1, return_wfn=True)
    if method == 'ccsd(t)':
        assert abs(w.variable('(T) CORRECTION ENERGY')) > 1.e-3
    return g.np.copy()


def product(method, direction):
    estimates = []
    for h in STEPS:
        plus = gradient(method, BASE+h*direction)
        minus = gradient(method, BASE-h*direction)
        estimates.append((plus-minus)/(2*h))
    coarse, middle, fine = estimates
    r1, r2 = (4*middle-coarse)/3, (4*fine-middle)/3
    delta = np.max(abs(r1-r2))
    print("Richardson change", delta, flush=True)
    assert delta < 2.e-7
    np.testing.assert_allclose(r2.sum(axis=0), 0., atol=2.e-8, rtol=0)
    return r2


def main():
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz', reference='rhf', scf_type='df', cc_type='df',
                          df_basis_scf='cc-pvdz-jkfit', df_basis_cc='cc-pvdz-ri',
                          e_convergence=1.e-12, d_convergence=1.e-12, r_convergence=1.e-11))
    # Verify the frame, units and sign against a genuinely analytic Hessian first.
    exact = psi4.hessian('scf', molecule=molecule(BASE), dertype=2).np @ STRETCH.ravel()
    scf_fd = product('scf', STRETCH).ravel()
    print('SCF analytic maximum error', np.max(abs(scf_fd-exact)), flush=True)
    np.testing.assert_allclose(scf_fd, exact, rtol=0, atol=2.e-7)
    for frozen in (True, False):
        psi4.set_options({'freeze_core': frozen})
        hv = product('ccsd(t)', STRETCH)
        reference = ([[0., .028971128, -.555946126], [0., -.015913673, .571083886],
                      [0., -.013057455, -.015137761]] if frozen else
                     [[0., .029003036, -.554947671], [0., -.015913543, .570071950],
                      [0., -.013089494, -.015124279]])
        np.testing.assert_allclose(hv, reference, rtol=0, atol=2.e-7)
        hu = product('ccsd(t)', PROBE)
        error = abs(np.vdot(PROBE, hv)-np.vdot(STRETCH, hu))
        print("frozen", frozen, "stretch product", hv, "reciprocity error", error, flush=True)
        assert error < 2.e-7


if __name__ == '__main__':
    main()
