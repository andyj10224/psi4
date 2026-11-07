"""Independent scalar finite-displacement checks of the PNO curvature kernel."""
import numpy as np
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock
from psi4.driver.procrouting.response.dlpno_pno_curvature import (
    semicanonical_mp2, pno_truncation_loss, pno_truncation_energy)


def at(block,u,v):
    return block.value+u*block.u+v*block.v+u*v*block.uv


def scalar(K,t,F,f,o,rank,diagonal):
    spin = 2*t-t.T
    density = (spin@t.T+spin.T@t)/(1+diagonal)
    _, x = np.linalg.eigh(density)
    x = x[:,-rank:]
    value = np.sum(K*spin)-np.sum((x.T@K@x)*(x.T@spin@x))
    if diagonal:
        value += f@np.linalg.solve(o*np.eye(len(f))-F,f)
        b = x.T@f
        value -= b@np.linalg.solve(o*np.eye(rank)-x.T@F@x,b)
    return value


def verify():
    rng = np.random.default_rng(852)
    def block(shape,symmetric=False):
        values = [rng.normal(size=shape)*s for s in (0.2,.03,.04,.01)]
        if symmetric:
            values = [(x+x.T)/2 for x in values]
        return MixedBlock(*values)
    maximum=0.
    for diagonal in (False,True):
        K=block((5,5),diagonal)
        F=block((5,5),True)
        F=F._replace(value=F.value+np.diag(np.arange(5)+1.))
        f=block((5,))
        o=MixedBlock(-.8,.02,-.03,.01)
        osum=MixedBlock(*(2*x for x in o))
        for initial in (False,True):
            t=semicanonical_mp2(K,F,osum) if initial else block((5,5),diagonal)
            for rank in (2,5):
                args=dict(exchange=K,amplitudes=t,fock=F,fov=f,occupied_energy=o,rank=rank)
                result=pno_truncation_loss(**args,diagonal=diagonal)
                swapped={k:(MixedBlock(v.value,v.v,v.u,v.uv) if isinstance(v,MixedBlock) else v)
                         for k,v in args.items()}
                reverse=pno_truncation_loss(**swapped,diagonal=diagonal)
                np.testing.assert_allclose(result.uv,reverse.uv,rtol=0,atol=1.e-13)
                def sample(u,v):
                    kk,ff,fo,oo=(at(x,u,v) for x in (K,F,f,o))
                    tt=at(t,u,v)
                    if initial:
                        e,c=np.linalg.eigh(ff)
                        tt=c@((c.T@kk@c)/(at(osum,u,v)-e[:,None]-e[None,:]))@c.T
                    return scalar(kk,tt,ff,fo,oo,rank,diagonal)
                def fd(h):
                    return sum(a*b*sample(a*h,b*h) for a in (-1,1) for b in (-1,1))/(4*h*h)
                numerical=(4*fd(.001)-fd(.002))/3
                error=abs(result.uv-numerical)
                maximum=max(maximum,error)
                assert error < 2.e-8,(diagonal,initial,rank,result,numerical)
                if rank==5:
                    assert max(abs(np.asarray(result))) < 1.e-12
                pair=(0,0) if diagonal else (0,1)
                summed=pno_truncation_energy({pair:args},{pair:args})
                assert np.allclose(summed,np.asarray(result)*(2 if diagonal else 4))
    # A zero density at a partial selection has no differentiable boundary.
    z=MixedBlock(*(np.zeros((5,5)) for _ in range(4)))
    try:
        pno_truncation_loss(K,z,F,f,o,2,diagonal=False)
    except ValueError:
        pass
    else:
        raise AssertionError('Degenerate selection was accepted')
    print('PNO mixed curvature maximum error',maximum)


def verify_lmp2_residual():
    from psi4.driver.procrouting.response.dlpno_pno_curvature import lmp2_residual, lmp2_mixed_residual
    rng=np.random.default_rng(732)
    def block(shape,sym=False):
        parts=[rng.normal(size=shape)*scale for scale in (1.,.1,.1,.03)]
        return MixedBlock(*((x+x.T)/2 if sym else x for x in parts))
    ranks={(0,0):2,(0,1):3,(1,1):2}
    final={p:dict(exchange=block((n,n),p[0]==p[1]),fock=block((n,n),True),
                  amplitudes=block((n,n),p[0]==p[1])) for p,n in ranks.items()}
    foo=block((2,2),True)
    edges={(0,0):[(1,0,0,1),(0,1,0,1)],(0,1):[(0,0,0,1),(1,1,0,1)],
           (1,1):[(0,1,0,1),(1,0,0,1)]}
    ov={(p,tuple(sorted((k,l)))):block((ranks[p],ranks[tuple(sorted((k,l)))]))
        for p,es in edges.items() for k,l,_,_ in es}
    def explicit(u,v):
        occupied=at(foo,u,v)
        result={}
        for p,inputs in final.items():
            i,j=p
            K,F,t=(at(inputs[key],u,v) for key in ('exchange','fock','amplitudes'))
            r=K+F@t+t@F-(occupied[i,i]+occupied[j,j])*t
            for k,l,a,b in edges[p]:
                q=tuple(sorted((k,l)))
                tq=at(final[q]['amplitudes'],u,v)
                if k>l: tq=tq.T
                S=at(ov[p,q],u,v)
                r-=occupied[a,b]*S@tq@S.T
            result[p]=(r+r.T)/2 if i==j else r
        return result
    actual=lmp2_residual(final,foo,edges,lambda p,q:ov[p,q])
    def fd(h):
        samples={(a,b):explicit(a*h,b*h) for a in (-1,1) for b in (-1,1)}
        return {p:sum(a*b*x[p] for (a,b),x in samples.items())/(4*h*h) for p in ranks}
    coarse,fine=fd(.002),fd(.001)
    error=max(np.max(abs(actual[p].uv-(4*fine[p]-coarse[p])/3)) for p in ranks)
    assert error<2.e-8,error
    source=lmp2_mixed_residual(final,foo,edges,lambda p,q:ov[p,q])
    const=lambda x:MixedBlock(x,np.zeros_like(x),np.zeros_like(x),np.zeros_like(x))
    trial={p:dict(exchange=const(np.zeros_like(a['exchange'].value)),fock=const(a['fock'].value),
                  amplitudes=const(a['amplitudes'].uv)) for p,a in final.items()}
    action=lmp2_residual(trial,const(foo.value),edges,lambda p,q:const(ov[p,q].value))
    for p in ranks:
        np.testing.assert_allclose(actual[p].uv,source[p]+action[p].value,rtol=0,atol=1.e-13)
    print('LMP2 mixed residual maximum error',error)


if __name__=='__main__':
    verify()
    verify_lmp2_residual()
