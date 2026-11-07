"""Water stretch column of the ordinary (T) block on a fixed local branch.

Only the mixed lower-order electronic response comes from displaced native states.
The independent local check differences stationary (T) gradients. The canonical
reference differences analytic DF-CCSD(T) minus DF-CCSD gradients.
"""
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment

from lmp2_tangent import calculation, check
from first_pno import raw
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_product, project_mixed_triples_row
from water_hvp import BASE, molecule
from psi4.driver.procrouting.response.dlpno_triples_response import TriplesHessianColumn, TriplesMixedRow


class NativeInputs(NamedTuple):
    occupied_fock: np.ndarray
    virtual_fock: dict
    moments: dict
    overlaps: dict
    electronic: dict


def stationary_gradient(direction):
    return sum(np.vdot(direction.energy_seed_tangents[r],direction.amplitudes[r])
               + np.vdot(direction.adjoints[r],direction.right_source[r]) for r in direction.edges)


def aligned_inputs(model, reference, rows):
    """Express native inputs at one geometry in the reference smooth frame."""
    ref,other = reference.reference_wavefunction(),model.reference_wavefunction()
    l0 = reference.variable('DLPNO RESPONSE ACTIVE LMOS').np
    lm = model.variable('DLPNO RESPONSE ACTIVE LMOS').np
    cross = psi4.core.MintsHelper(other).ao_overlap(ref.basisset(),other.basisset()).np
    overlap = l0.T @ cross @ lm
    _,order = linear_sum_assignment(-abs(overlap))
    matched = overlap[np.arange(len(order)),order]
    assert min(abs(matched)) > .999
    signs = np.sign(matched)
    lm = lm[:,order]*signs
    new_rows = psi4.core.dlpno_triples_response_state(model)
    assert len(new_rows) == len(rows)
    # The TNOs depend on both pair selections. Check both ranks as well.
    lower = {'lmo':lm,'fock':other.Fa().np.copy()}
    native_t2 = psi4.core.dlpno_lmp2_blocks(model,'CCSD_DOUBLES')
    for kind in ('FRAME','FINAL_FRAME'):
        old = psi4.core.dlpno_lmp2_blocks(reference,kind)
        new = psi4.core.dlpno_lmp2_blocks(model,kind)
        assert len(old) == len(new)
        for (i,j),x in old.items():
            a,b = order[i],order[j]
            p = tuple(sorted((a,b)))
            assert x.np.shape == new[p].np.shape
            if kind == 'FINAL_FRAME':
                y = new[p].np
                left,_,right = np.linalg.svd(y.T@ref.S().np@x.np)
                rot = left@right
                lower['pno',(i,j)] = y@rot
                t = native_t2[p].np if a <= b else native_t2[p].np.T
                lower['t2',(i,j)] = signs[i]*signs[j]*(rot.T@t@rot)
                if i == j:
                    dom,ts,_ = psi4.core.dlpno_ccsd_df_pair_state(model,int(a),int(a))
                    lower['t1',i] = signs[i]*(rot.T@ts[dom.index(a)].np).ravel()
    frames,moments,virtual,edges = {},{},{},{}
    for r,row in rows.items():
        occupied = tuple(order[i] for i in r)
        new = new_rows[tuple(sorted(occupied))]
        y = new[2].np
        assert y.shape == row[2].np.shape
        assert sorted((order[l],order[m],tuple(sorted(order[a] for a in (i,j,k))))
                      for l,m,i,j,k,_ in row[5]) == sorted((l,m,tuple(sorted((i,j,k)))) for l,m,i,j,k,_ in new[5])
        a,_,bt = np.linalg.svd(y.T @ ref.S().np @ row[2].np)
        rot = a @ bt
        frames[r] = y @ rot
        virtual[r] = frames[r].T @ other.Fa().np @ frames[r]
        axes = np.argsort(np.argsort(occupied,kind='stable'))
        n = y.shape[1]
        def moment(block):
            tensor = block.np.reshape((n,)*3).transpose(axes)
            return np.prod(signs[list(r)]) * np.einsum('ai,bj,ck,abc->ijk',rot,rot,rot,tensor,optimize=True)
        moments[r] = moment(new[4]),moment(new[7])
    for r,row in rows.items():
        for _,_,i,j,k,_ in row[5]:
            q = tuple(sorted((i,j,k)))
            edges[r,q] = frames[r].T @ other.S().np @ frames[q]
    lower.update({('tno',r):y for r,y in frames.items()})
    return NativeInputs(lm.T @ other.Fa().np @ lm,virtual,moments,edges,lower)


def mixed_coefficients(data,u,v,h):
    """Return central mixed differences of fitted moments, Fock and overlap blocks."""
    if u == v:
        weights = (1.,-2.,1.)
        values = [data(tuple(k*h if j == v else 0. for j in range(9))) for k in (1,0,-1)]
        denom = h*h
    else:
        weights = (1.,-1.,-1.,1.)
        values = []
        for su,sv in ((1,1),(1,-1),(-1,1),(-1,-1)):
            delta = np.zeros(9)
            delta[[u,v]] = [su*h,sv*h]
            values.append(data(tuple(delta)))
        denom = 4*h*h
    def difference(get):
        return sum(weight*get(x) for weight,x in zip(weights,values)) / denom
    def row(r):
        return TriplesMixedRow(difference(lambda x:x.moments[r][0]),difference(lambda x:x.moments[r][1]),
                               difference(lambda x:x.virtual_fock[r]))
    return (difference(lambda x:x.occupied_fock),row,lambda r,q:difference(lambda x:x.overlaps[r,q])), lambda key:difference(lambda x:x.electronic[key])


def molecular_inputs(w,rows,u,v,lower,lower_u,lower_v,lower_uv):
    ref = w.reference_wavefunction()
    mints = psi4.core.MintsHelper(ref)
    au,xu = divmod(u,3)
    av,xv = divmod(v,3)
    overlap = MixedBlock(ref.S().np,mints.ao_oei_deriv1('OVERLAP',au)[xu].np,
                         mints.ao_oei_deriv1('OVERLAP',av)[xv].np,
                         mints.ao_oei_deriv2('OVERLAP',au,av)[3*xu+xv].np)
    def electronic(key):
        return MixedBlock(lower[key],lower_u[key],lower_v[key],lower_uv(key))
    lmo,fock = electronic('lmo'),electronic('fock')
    metric,q = raw(ref,ref.get_basisset('DF_BASIS_CC'))
    def df1(atom,xyz):
        m = mints.ao_metric_deriv1(atom,'DF_BASIS_CC')[xyz].np
        q = mints.ao_3center_deriv1(atom,'DF_BASIS_CC')[xyz].np
        return m,q
    mu,qu = df1(au,xu)
    mv,qv = df1(av,xv)
    # Full reference AO domain for this small-system verification.
    ao_ids = list(range(ref.basisset().nbf()))
    @lru_cache(None)
    def local_df(aux):
        indices = list(aux)
        m2,q2 = psi4.core.dlpno_df_mixed_integrals(ref,'DF_BASIS_CC',indices,ao_ids,u,v)
        ix = np.ix_(indices,indices)
        return MixedBlock(metric[ix],mu[ix],mv[ix],m2.np), MixedBlock(q[indices],qu[indices],qv[indices],q2.np)
    def ordered(a,b):
        t = electronic(('t2',tuple(sorted((a,b)))))
        return t if a <= b else MixedBlock(*(x.T for x in t))
    def mixed_row(r):
        dom,aux = rows[r][:2]
        m,three = local_df(tuple(aux))
        return project_mixed_triples_row(r,dom,m,three,overlap,fock,lmo,electronic(('tno',r)),
                lambda i,j:electronic(('pno',tuple(sorted((i,j))))),
                lambda i:electronic(('t1',i)),ordered)
    def edge(r,q):
        return mixed_product('ma,mn,nb->ab',electronic(('tno',r)),overlap,electronic(('tno',q))).uv
    @lru_cache(None)
    def occupied(i,j):
        return mixed_product('m,mn,n->',lmo.slice((slice(None),i)),fock,lmo.slice((slice(None),j))).uv
    return occupied,mixed_row,edge


def benchmark(frozen, coordinates=range(9)):
    psi4.set_options({'freeze_core':frozen})
    w = calculation(BASE)
    rows = psi4.core.dlpno_triples_response_state(w)
    v = 5  # H1 z
    lower_v = {}
    column_row = check(frozen,row_at=divmod(v,3),reference=w,lower_order=lower_v)
    column = TriplesHessianColumn(column_row)

    @lru_cache(None)
    def data(delta):
        displaced = calculation(BASE+np.array(delta).reshape(3,3)) if any(delta) else w
        return aligned_inputs(displaced,w,rows)
    @lru_cache(None)
    def along_column(h):
        d = np.zeros(9)
        d[v] = h
        return calculation(BASE+d.reshape(3,3))
    results = []
    for u in coordinates:
        lower_u = lower_v if u == v else {}
        row = column_row if u == v else check(frozen,row_at=divmod(u,3),reference=w,lower_order=lower_u)
        relax = column.relaxation(row)
        reverse = TriplesHessianColumn(row).relaxation(column_row)
        np.testing.assert_allclose(relax,reverse,rtol=0.,atol=2.e-11)
        elements,reference = [],[]
        for h in (.002,.001):
            projected,lower_uv = mixed_coefficients(data,u,v,h)
            # Use analytic AO and DF derivatives with lower-order mixed response
            # from the native displaced states. Separately check the result
            # against differentiating the already-assembled native inputs.
            inputs = molecular_inputs(w,rows,u,v,data((0.,)*9).electronic,lower_u,lower_v,lower_uv)
            element = column.element(row,*inputs)
            direct = column.element(row,*projected)
            print('mixed provider difference',frozen,u,h,element-direct,flush=True)
            assert abs(element-direct) < 5.e-6
            elements.append(element)
            derivatives = [stationary_gradient(check(frozen,row_at=divmod(u,3),reference=along_column(k*h)))
                           for k in (1,-1)]
            reference.append((derivatives[0]-derivatives[1])/(2*h))
        estimated = (4*elements[1]-elements[0])/3
        numerical = (4*reference[1]-reference[0])/3
        print('triples curvature',frozen,u,relax,estimated,numerical,estimated-numerical,flush=True)
        results.append((relax,estimated,numerical))
    result = np.asarray(results)
    np.testing.assert_allclose(result[:,1],result[:,2],rtol=0.,atol=2.e-7)
    if list(coordinates) == list(range(9)):
        np.testing.assert_allclose(result[:,2].reshape(3,3).sum(axis=0),0.,rtol=0.,atol=2.e-6)
    return result


def canonical_reference(frozen):
    """Isolate the canonical (T) contribution on the same stretch."""
    psi4.set_options({'freeze_core':frozen,'cc_type':'df'})
    estimates = []
    for h in (.004,.002):
        grads = []
        for sign in (1,-1):
            coords = BASE.copy()
            coords[1,2] += sign*h
            mol = molecule(coords)
            def grad(method):
                psi4.core.clean()
                return psi4.gradient(method,molecule=mol,dertype=1).np.copy()
            grads.append(grad('ccsd(t)')-grad('ccsd'))
        estimates.append((grads[0]-grads[1])/(2*h))
    column = (4*estimates[-1]-estimates[-2])/3
    np.testing.assert_allclose(column.sum(axis=0),0.,atol=1.e-8,rtol=0.)
    print('canonical (T) column',frozen,column.ravel(),flush=True)
    return column


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(scf__diis=False,scf_initial_accelerator='NONE',basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_cc='cc-pvdz-ri',
                          df_basis_scf='cc-pvdz-jkfit',e_convergence=1.e-13,d_convergence=1.e-13,
                          r_convergence=1.e-11,solver_convergence=1.e-12,dlpno_maxiter=600,
                          local_convergence=1.e-14,local_gradient_convergence=1.e-8,
                          dlpno_keep_lmp2_response=True,dlpno_gradient_model=True,
                          dlpno_local_orbitals='BOYS',t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4,
                          t_cut_trace_mp2=0.,t_cut_energy_mp2=0.,t_cut_trace=0.,t_cut_energy=0.,
                          t_cut_pairs=1.e-9,t_cut_pairs_mp2=1.e-9,dlpno_keep_triples_response=True,
                          min_tnos=3,t_cut_tno=1.e-5))
    psi4.set_options({k:0. for k in ('t_cut_do t_cut_do_pre t_cut_do_ij t_cut_do_uv t_cut_mkn '
                                      't_cut_pre t_cut_clmo t_cut_cpao dlpno_ao_ints_tol').split()})
    canonical_columns = {}
    for frozen in (True,False):
        local = benchmark(frozen)
        canonical = canonical_columns[frozen] = canonical_reference(frozen)
        print('max local/canonical (T) difference',frozen,
              np.max(abs(local[:,1]-canonical.ravel())),flush=True)
    # Keep the nonzero pair cutoff. On this molecule it retains all pairs.
    psi4.set_options(dict(t_cut_pno=0.,t_cut_pno_mp2=0.,t_cut_tno=0.,t_cut_do_triples=0.,
                          t_cut_mkn_triples=0.,f_cut=0.,f_cut_t=0.))
    for frozen in (True,False):
        result = benchmark(frozen,(5,))
        np.testing.assert_allclose(result[0,1],canonical_columns[frozen][1,2],atol=2.e-7,rtol=0.)
