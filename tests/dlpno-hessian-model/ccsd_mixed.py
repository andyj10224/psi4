"""Native residual and mixed nuclear source checks on the full-domain water model."""
import argparse
from functools import lru_cache
import numpy as np
import psi4
from localization_integration import configure
from lmp2_tangent import calculation, check
from final_pno_mixed import analytic_final_pno_inputs
from triples_curvature import aligned_inputs, molecular_blocks, mixed_stencil
from water_hvp import BASE
from psi4.driver.procrouting.response.dlpno_ccsd_mixed import mixed_ccsd_residual
from psi4.driver.procrouting.response.dlpno_ccsd_response import (
    transported_ccsd_residual, semi_analytic_ccsd_mixed_source, TriplesCCSDAdjoint, ccsd_amplitude_curvature)


def pack(blocks, component=None):
    one,two = blocks
    arrays = [*one,*two.values()]
    return np.concatenate([(a if component is None else a[component]).ravel() for a in arrays])


def analytic_source(w,u,v,base,lower_u,lower_v,pno_data=None):
    def forbidden(key):
        raise AssertionError(f'Analytic CCSD source requested a sampled input: {key}')
    inputs = molecular_blocks(w,u,v,base,lower_u,lower_v,forbidden,True,pno_data)
    replay = mixed_ccsd_residual(w,*inputs)
    return ([x.uv for x in replay[0]],{p:x.uv for p,x in replay[1].items()}),replay,inputs


def amplitude_check(w, inputs, evaluator, base):
    """Isolate R_tt with random symmetric amplitudes and native quartic extraction."""
    from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock
    electronic,overlap,local_df = inputs
    rng = np.random.default_rng(3982)
    constant = lambda x:MixedBlock(x.value,*[np.zeros_like(x.value) for _ in range(3)])
    directions = {}
    def trial(key):
        x = electronic(key)
        if isinstance(key,tuple) and key[0] in ('t1','t2'):
            if key not in directions:
                u,v = (rng.normal(size=x.value.shape)*.01 for _ in range(2))
                if key[0]=='t2' and key[1][0]==key[1][1]: u,v = (u+u.T)/2,(v+v.T)/2
                directions[key] = MixedBlock(x.value,u,v,np.zeros_like(x.value))
            return directions[key]
        return constant(x)
    replay = mixed_ccsd_residual(w,trial,constant(overlap),lambda a:tuple(constant(x) for x in local_df(a)))
    one,two = base
    def direction(c):
        return [directions['t1',i][c] for i in range(len(one))],{p:directions['t2',p][c] for p in two}
    exact = ccsd_amplitude_curvature(evaluator,base,direction(1),direction(2))
    err = np.max(abs(pack(replay,3)-pack(exact)))
    print('pure amplitude mixed residual error',err,flush=True)
    np.testing.assert_allclose(pack(replay,3),pack(exact),atol=2.e-11,rtol=0.)
    # The public source contract must reject already-mixed amplitudes.
    bad = ('t1',0)
    def invalid(k):
        x = trial(k)
        return MixedBlock(x.value,x.u,x.v,np.ones_like(x.value)) if k==bad else x
    try: mixed_ccsd_residual(w,invalid,constant(overlap),local_df)
    except ValueError as exc: assert 'affine' in str(exc)
    else: raise AssertionError('Accepted a mixed CCSD amplitude.')


def verify(frozen, coordinates=(1,5), coarse_step=.016):
    psi4.set_options({'freeze_core':frozen})
    w = calculation(BASE)
    rows = psi4.core.dlpno_triples_response_state(w)
    v = 5
    zero = (0.,)*9
    @lru_cache(None)
    def model(delta): return w if not any(delta) else calculation(BASE+np.array(delta).reshape(3,3))
    @lru_cache(None)
    def state(delta):
        # The sixth-order oracle spans four times the original displacement.
        # Allow the corresponding smooth occupied rotation; all PNO/TNO ranks,
        # pair labels and response edges still undergo the original checks.
        return aligned_inputs(model(delta),w,rows,minimum_occupied_overlap=.995)
    def evaluator(delta):
        s = state(delta)
        return lambda a,b:transported_ccsd_residual(model(delta),a,b,s.order,s.signs,s.pair_rotations)
    base = state(zero).electronic
    pairs = tuple(state(zero).pair_rotations)
    def blocks(lower):
        return [lower['t1',i] for i in range(base['lmo'].shape[1])],{p:lower['t2',p] for p in pairs}
    lower_v = {}
    check(frozen,row_at=divmod(v,3),reference=w,lower_order=lower_v)
    adjoint = TriplesCCSDAdjoint(w)
    for u in coordinates:
        lower_u = lower_v if u==v else {}
        if u!=v: check(frozen,row_at=divmod(u,3),reference=w,lower_order=lower_u)
        pno = analytic_final_pno_inputs(w,u,v)
        source,replay,inputs = analytic_source(w,u,v,base,lower_u,lower_v,pno)
        if u == coordinates[0]: amplitude_check(w,inputs,evaluator(zero),blocks(base))
        native = evaluator(zero)(*blocks(base))
        print('residual value/first maxima',frozen,u,np.max(abs(pack(replay,0)-pack(native))),
              np.max(abs(pack(replay,1))),np.max(abs(pack(replay,2))),flush=True)
        np.testing.assert_allclose(pack(replay,0),pack(native),atol=2.e-10,rtol=0.)
        np.testing.assert_allclose(pack(replay,1),0.,atol=2.e-9,rtol=0.)
        np.testing.assert_allclose(pack(replay,2),0.,atol=2.e-9,rtol=0.)
        # Swap every mixed block, independent of any finite-difference step.
        def swap(x): return type(x)(x.value,x.v,x.u,x.uv)
        electronic,overlap,local_df = inputs
        reverse = mixed_ccsd_residual(w,lambda k:swap(electronic(k)),swap(overlap),
                                     lambda a:tuple(swap(x) for x in local_df(a)))
        np.testing.assert_allclose(pack(replay,3),pack(reverse,3),atol=2.e-11,rtol=0.)
        from ccsd_mixed_tangent_check import source_from_tangents
        oracle = source_from_tangents(w,rows,model,frozen,u,v,.004)
        err = np.max(abs(pack(source)-pack(oracle)))
        print('transported analytic first-tangent source error',frozen,u,err,flush=True)
        np.testing.assert_allclose(pack(source),pack(oracle),atol=2.e-7,rtol=0.)
        numerical = []
        for h in (coarse_step,coarse_step/2,coarse_step/4):
            sample = semi_analytic_ccsd_mixed_source(evaluator(zero),blocks(base),blocks(lower_u),blocks(lower_v),
                      ((du,dv,weight,evaluator(delta)) for delta,du,dv,weight in mixed_stencil(u,v,h)))
            numerical.append(sample)
            print('mixed source raw stencil error',frozen,u,h,np.max(abs(pack(source)-pack(sample))),
                  'core singles',np.max(abs(source[0][0]-sample[0][0])),flush=True)
        error = pack(source)-(pack(numerical[0])-20*pack(numerical[1])+64*pack(numerical[2]))/45
        scalar_error = adjoint.correction(*source)-(adjoint.correction(*numerical[0])-20*adjoint.correction(*numerical[1])+64*adjoint.correction(*numerical[2]))/45
        print('analytic mixed CCSD residual',frozen,u,'max',np.max(abs(pack(source))),
              'sixth-order error',np.max(abs(error)),'triples adjoint error',scalar_error,flush=True)
        # Native residual second differences amplify core-singles cancellation.
        # The independent transported-first-tangent oracle above validates every
        # source component at 2e-7; retain this separate scalar-adjoint check.
        assert abs(scalar_error) < 2.e-9


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',choices=('both','frozen','all'),default='both')
    parser.add_argument('--coordinates',type=int,nargs='+',default=[1,5])
    parser.add_argument('--coarse-step',type=float,default=.016)
    args = parser.parse_args()
    if not args.coarse_step > 0 or any(not 0 <= u < 9 for u in args.coordinates):
        parser.error('Expected positive spacing and Cartesian indices 0 through 8.')
    configure()
    psi4.set_options(dict(t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4,t_cut_tno=1.e-5,
                          scf__ints_tolerance=1.e-14,e_convergence=1.e-14,d_convergence=1.e-14,
                          solver_convergence=1.e-14,local_gradient_convergence=1.e-13))
    psi4.set_output_file('ccsd_mixed.out',False)
    for frozen in ((True,False) if args.core=='both' else (args.core=='frozen',)):
        verify(frozen,tuple(args.coordinates),args.coarse_step)
