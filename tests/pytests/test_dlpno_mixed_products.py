"""Mixed-contraction path reuse against independent product-rule expansion."""
from itertools import product
import numpy as np
import pytest
from psi4.driver.procrouting.response.dlpno_triples_curvature import (
    MixedBlock,mixed_product,_mixed_contraction_path,_mixed_tree)


@pytest.mark.parametrize('equation,shapes',[
    ('ab,bc->ac',[(3,4),(4,2)]),
    ('ma,Qmn,nb->Qab',[(4,2),(5,4,3),(3,2)]),
    ('a,a->',[(7,),(7,)]),
    ('ab->ba',[(4,3)]),
    (',ab->ab',[(),(3,2)]),
    ('ia,jb,kc,abc->ijk',[(3,2),(4,3),(2,5),(2,3,5)]),
    ('Qa,QR,Rb->ab',[(5,3),(5,5),(5,4)]),
    ('ab,ab->ab',[(3,4),(3,4)]),
    ('aab,bc->ac',[(3,3,4),(4,2)]),
    ('ab,bc->ac',[(3,0),(0,2)]),
    ('...ab,...bc->...ac',[(2,3,4),(2,4,5)]),
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
    _mixed_tree.cache_clear()
    actual=mixed_product(equation,*blocks)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0.)
    info=_mixed_tree.cache_info()
    assert info.misses==int(len(blocks)>1)
    # New, noncontiguous values of the same shapes share only path metadata.
    other=[MixedBlock(*(np.asarray(x)[tuple(slice(None,None,-1) for _ in x.shape)] for x in b)) for b in blocks]
    repeated=mixed_product(equation,*other)
    assert _mixed_tree.cache_info().hits==info.hits+int(len(blocks)>1)
    np.testing.assert_allclose(repeated.value,np.einsum(equation,*(b[0] for b in other)),atol=2e-12,rtol=0.)
    assert _mixed_contraction_path.cache_info().maxsize==1024
    assert _mixed_tree.cache_info().maxsize==1024


# Literal contractions in the installed DLPNO response equations.
@pytest.mark.parametrize("equation", [
    ',a->a', ',ab->ab', ',ac,cd,bd->ab', ',k,l->kl',
    ',kl->kl', ',uc,cd,vd->uv', 'Q,QR,Rab->ab', 'Q,QR,Rac->ac',
    'Qa,QR,Rb->ab', 'Qa,QR,Rbd->abd', 'Qa,QR,Rc->ac', 'Qa,Qb->ab',
    'Qa,ab->Qb', 'Qac,c->Qa', 'Qb,QR,Rc->bc', 'Qk,ka->Qa',
    'Qkc,c->Qk', 'Ql,QR,Rc->lc', 'Qma,ab->Qmb', 'Qmc,mc->Q',
    'a,a->', 'a,ab,b->', 'a,ax->x', 'a,b->ab',
    'a,bc->abc', 'a,c->ac', 'ab,ab->', 'ab,ac,cd,db->',
    'ab,b->a', 'ab,c->abc', 'abd,cd->abc', 'ac,bc,by->ay',
    'ac,bc->ab', 'ac,cb,b->a', 'ac,cb->ab', 'ac,cd,bd,ab->',
    'ad,de,ec->ac', 'ax,ac,c->x', 'ax,cda,cd->x', 'b,ba,ac->c',
    'b,bac->ac', 'b,bca->ac', 'b,cab->ac', 'bx,bd,cd,cy->xy',
    'ca,ba,b->c', 'ca,cb->ab', 'cd,ca,ab,db->', 'ia,i->a',
    'ia,ij,jb->ab', 'ijk,abc,jb,kc->ia', 'k,k->', 'ka,Qkc->Qac',
    'ka,kl,lb->ab', 'la,lc->ac', 'lb,bc->lc', 'lb,cb->lc',
    'lc,lc->', 'm,Qmn,n->Q', 'm,Qmn,na->Qa', 'm,Qmn,nc->Qc',
    'm,mn,n->', 'm,mn,na->a', 'm,mn,nb->b', 'ma,Qmn,nb->Qab',
    'ma,Qmn,nc->Qac', 'ma,a->m', 'ma,a->ma', 'ma,ab,bc->mc',
    'ma,ab,nb->mn', 'ma,ab->mb', 'ma,m->a', 'ma,mb,b->a',
    'ma,mb->ab', 'ma,mn,na->a', 'ma,mn,nb->ab', 'mi,ia->ma',
    'mi,ij,jn->mn', 'mi,ma->ia', 'mi,mn,nj->ij', 'mn,n->m',
    'mn,na->ma', 'mn,ni->mi', 'mr,mn,nc->rc', 'xa,ac,cy->xy',
])
def test_response_equations_against_independent_product_rule(equation):
    rng=np.random.default_rng(701)
    dimensions={label:2+(i%2) for i,label in enumerate(sorted(set(equation.replace('->','').replace(',',''))))}
    shapes=[tuple(dimensions[label] for label in term) for term in equation.split('->')[0].split(',')]
    blocks=[MixedBlock(*(rng.normal(scale=.1,size=shape) for _ in range(4))) for shape in shapes]
    shape=np.einsum(equation,*(b.value for b in blocks),optimize=False).shape
    expected=[np.zeros(shape) for _ in range(4)]
    for choices in product(range(4),repeat=len(blocks)):
        used=0
        for mask in choices:
            if used&mask:break
            used|=mask
        else:expected[used]+=np.einsum(equation,*(b[k] for b,k in zip(blocks,choices)),optimize=False)
    np.testing.assert_allclose(mixed_product(equation,*blocks),expected,atol=2e-12,rtol=0.)
