"""Native DF response against independent NumPy contractions, without SCF."""
import numpy as np
import pytest
import psi4

from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_product
from psi4.driver.procrouting.response.dlpno_rhf_curvature import mixed_df_fock


def native(q, r, d):
    panels = lambda blocks: [psi4.core.Matrix.from_array(x.reshape(x.shape[0], -1)) for x in blocks]
    return np.array([x.np for x in psi4.core.dlpno_df_fock_response(panels(q), panels(r), panels(d))])


def oracle(q, r, d):
    if len(q) == 1:
        return np.array([2*np.einsum('Qmn,Qrs,rs->mn', q[0], r[0], d[0])
                         -np.einsum('Qmr,Qns,rs->mn', q[0], r[0], d[0])])
    j = mixed_product('Qmn,Qrs,rs->mn', MixedBlock(*q), MixedBlock(*r), MixedBlock(*d))
    k = mixed_product('Qmr,Qns,rs->mn', MixedBlock(*q), MixedBlock(*r), MixedBlock(*d))
    return 2*np.array(j)-np.array(k)


@pytest.mark.parametrize('count', [1, 4])
@pytest.mark.parametrize('n,naux', [(1, 1), (5, 7), (48, 17)])
def test_native_df_fock_product_rule_and_threads(count, n, naux):
    rng = np.random.default_rng(721)
    # Deliberately nonsymmetric, unrelated operands detect hidden transposes.
    q, r = (rng.normal(scale=.1, size=(count, naux, n, n)) for _ in range(2))
    d = rng.normal(scale=.1, size=(count, n, n))
    expected = oracle(q, r, d)
    threads = psi4.get_num_threads()
    try:
        psi4.set_num_threads(1)
        serial = native(q, r, d)
        psi4.set_num_threads(4)
        parallel = native(q, r, d)
    finally:
        psi4.set_num_threads(threads)
    # The largest shape activates OpenMP packing. BLAS reductions may differ
    # with the vendor/thread setting, so do not require bitwise identity.
    np.testing.assert_allclose(serial, expected, atol=2e-11, rtol=0.)
    np.testing.assert_allclose(parallel, expected, atol=2e-11, rtol=0.)
    np.testing.assert_allclose(parallel, serial, atol=2e-11, rtol=0.)


def test_df_fock_metric_response_against_mixed_finite_difference():
    rng = np.random.default_rng(31)
    n, a = 3, 5
    blocks = lambda shape: MixedBlock(*(rng.normal(scale=.1, size=shape) for _ in range(4)))
    j = blocks((a,a)); j = MixedBlock(*(x+x.T for x in j))
    j = j._replace(value=j.value@j.value.T+np.eye(a))
    q, d = blocks((a,n,n)), blocks((n,n))
    actual = mixed_df_fock(j,q,d)
    def evaluate(u,v):
        at = lambda b: b.value+u*b.u+v*b.v+u*v*b.uv
        ja, qa, da = at(j), at(q), at(d)
        fitted = np.linalg.solve(ja,qa.reshape(a,-1)).reshape(qa.shape)
        return oracle([qa],[fitted],[da])[0]
    central=[]
    for step in (.004,.002):
        central.append((evaluate(step,step)-evaluate(step,-step)
                        -evaluate(-step,step)+evaluate(-step,-step))/(4*step**2))
    np.testing.assert_allclose(actual.uv,(4*central[1]-central[0])/3,atol=2e-9,rtol=0.)


@pytest.mark.parametrize('count', [0, 2, 3])
def test_df_fock_rejects_invalid_component_count(count):
    m = psi4.core.Matrix.from_array(np.ones((2,2)))
    with pytest.raises(RuntimeError, match='one or four'):
        psi4.core.dlpno_df_fock_response([m]*count,[m]*count,[m]*count)


def test_df_fock_rejects_inconsistent_dimensions():
    q = psi4.core.Matrix.from_array(np.ones((3,4)))
    d = psi4.core.Matrix.from_array(np.ones((2,2)))
    with pytest.raises(RuntimeError, match='dimensions'):
        psi4.core.dlpno_df_fock_response([q],[d],[d])
