"""Mixed-contraction path reuse against independent product-rule expansion."""
from itertools import product
import numpy as np
import pytest
from psi4.driver.procrouting.response.dlpno_triples_curvature import (
    MixedBlock,mixed_product,_mixed_contraction_path)


@pytest.mark.parametrize('equation,shapes',[
    ('ab,bc->ac',[(3,4),(4,2)]),
    ('ma,Qmn,nb->Qab',[(4,2),(5,4,3),(3,2)]),
    ('a,a->',[(7,),(7,)]),
    ('ab->ba',[(4,3)]),
    (',ab->ab',[(),(3,2)]),
])
def test_mixed_product_cached_path(equation,shapes):
    rng=np.random.default_rng(55)
    blocks=[MixedBlock(*(rng.normal(size=shape) for _ in range(4))) for shape in shapes]
    # Derivative masks 0, u, v, uv. Disjoint masks give the full product rule.
    expected=[np.zeros(np.einsum(equation,*(b[0] for b in blocks)).shape) for _ in range(4)]
    for choices in product(range(4),repeat=len(blocks)):
        used=0
        for mask in choices:
            if used & mask:break
            used|=mask
        else:
            expected[used]+=np.einsum(equation,*(b[k] for b,k in zip(blocks,choices)),optimize=False)
    _mixed_contraction_path.cache_clear()
    actual=mixed_product(equation,*blocks)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0.)
    info=_mixed_contraction_path.cache_info()
    assert info.misses==1 and info.currsize==1
    # New, noncontiguous values of the same shapes share only path metadata.
    other=[MixedBlock(*(np.asarray(x)[tuple(slice(None,None,-1) for _ in x.shape)] for x in b)) for b in blocks]
    repeated=mixed_product(equation,*other)
    assert _mixed_contraction_path.cache_info().hits==info.hits+1
    np.testing.assert_allclose(repeated.value,np.einsum(equation,*(b[0] for b in other)),atol=2e-12,rtol=0.)
    assert _mixed_contraction_path.cache_info().maxsize==1024
