"""Water stretch column of retained CCSD and ordinary (T) on a fixed local branch.

The CCSD-stationary route eliminates mixed T1/T2 and builds mixed TNO frames.
Both PNO selections and their coupled-LMP2 parent response are analytic. The
mixed CCSD nuclear residual is replayed analytically at the reference geometry.
Aligned displaced states provide independent checks only.
The independent local check differences stationary (T) gradients. The canonical
reference differences analytic DF-CCSD(T) minus DF-CCSD gradients.
"""
from functools import lru_cache, partial
from typing import NamedTuple

import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment

from lmp2_tangent import calculation, check, blocks
from psi4.driver.procrouting.response.dlpno_orbital import pair_exchange_tangent
from psi4.driver.procrouting.response.dlpno_ccsd_curvature import RetainedCCSDCurvature, project_retained_ccsd_energy
from first_pno import raw
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_product, project_mixed_triples_row, mixed_tno_frame
from psi4.driver.procrouting.response.dlpno_ccsd_mixed import mixed_ccsd_residual
from psi4.driver.procrouting.response.dlpno_ccsd_response import TriplesCCSDAdjoint, transported_ccsd_residual, semi_analytic_ccsd_mixed_source, ccsd_weighted_dot
from water_hvp import BASE, molecule
from psi4.driver.procrouting.response.dlpno_triples_response import TriplesHessianColumn, TriplesMixedRow


class NativeInputs(NamedTuple):
    occupied_fock: np.ndarray
    virtual_fock: dict
    moments: dict
    overlaps: dict
    electronic: dict
    order: tuple
    signs: np.ndarray
    pair_rotations: dict


def stationary_gradient(direction):
    return sum(np.vdot(direction.energy_seed_tangents[r],direction.amplitudes[r])
               + np.vdot(direction.adjoints[r],direction.right_source[r]) for r in direction.edges)


def aligned_inputs(model, reference, rows, *, minimum_occupied_overlap=.999):
    """Express native inputs at one geometry in the reference smooth frame."""
    ref,other = reference.reference_wavefunction(),model.reference_wavefunction()
    l0 = reference.variable('DLPNO RESPONSE ACTIVE LMOS').np
    lm = model.variable('DLPNO RESPONSE ACTIVE LMOS').np
    cross = psi4.core.MintsHelper(other).ao_overlap(ref.basisset(),other.basisset()).np
    overlap = l0.T @ cross @ lm
    _,order = linear_sum_assignment(-abs(overlap))
    matched = overlap[np.arange(len(order)),order]
    assert min(abs(matched)) > minimum_occupied_overlap, min(abs(matched))
    signs = np.sign(matched)
    lm = lm[:,order]*signs
    new_rows = psi4.core.dlpno_triples_response_state(model)
    assert len(new_rows) == len(rows)
    # The TNOs depend on both pair selections. Check both ranks as well.
    occupied = other.Ca_subset('AO','OCC').np
    lower = {'lmo':lm,'fock':other.Fa().np.copy(),'occupied_density':occupied@occupied.T}
    vir = other.Ca_subset('AO','VIR').np
    x,_,y = np.linalg.svd(vir.T@ref.S().np@ref.Ca_subset('AO','VIR').np)
    lower['parent'] = vir@(x@y)
    rotations = {}
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
                rotations[i,j] = rot
                lower['pno',(i,j)] = y@rot
                t = native_t2[p].np if a <= b else native_t2[p].np.T
                lower['t2',(i,j)] = signs[i]*signs[j]*(rot.T@t@rot)
                if i == j:
                    block = lower['t2',(i,j)]
                    lower['t2',(i,j)] = .5*(block+block.T)
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
    return NativeInputs(lm.T @ other.Fa().np @ lm,virtual,moments,edges,lower,tuple(order),signs,rotations)


def mixed_stencil(u,v,h):
    if u == v:
        nodes = ((1,0,1), (0,0,-2), (-1,0,1))
        scale = h*h
    else:
        nodes = ((1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1))
        scale = 4*h*h
    samples = []
    for a,b,weight in nodes:
        delta = np.zeros(9)
        delta[u] += a*h
        delta[v] += b*h
        samples.append((tuple(delta),a*h,b*h,weight/scale))
    return samples


def mixed_coefficients(data,u,v,h):
    """Return central mixed differences of fitted moments, Fock and overlap blocks."""
    samples = mixed_stencil(u,v,h)
    values = [(weight,data(delta)) for delta,_,_,weight in samples]
    def difference(get):
        return sum(weight*get(x) for weight,x in values)
    def row(r):
        return TriplesMixedRow(difference(lambda x:x.moments[r][0]),difference(lambda x:x.moments[r][1]),
                               difference(lambda x:x.virtual_fock[r]))
    return (difference(lambda x:x.occupied_fock),row,lambda r,q:difference(lambda x:x.overlaps[r,q])), lambda key:difference(lambda x:x.electronic[key])


def molecular_blocks(w,u,v,lower,lower_u,lower_v,lower_uv,trial_ccsd,pno_data=None):
    ref = w.reference_wavefunction()
    mints = psi4.core.MintsHelper(ref)
    au,xu = divmod(u,3)
    av,xv = divmod(v,3)
    overlap = MixedBlock(ref.S().np,mints.ao_oei_deriv1('OVERLAP',au)[xu].np,
                         mints.ao_oei_deriv1('OVERLAP',av)[xv].np,
                         mints.ao_oei_deriv2('OVERLAP',au,av)[3*xu+xv].np)
    @lru_cache(None)
    def rhf_second():
        from rhf_mixed import analytic_inputs
        return analytic_inputs(ref,u,v,lower_u['occupied_density'],lower_v['occupied_density'],partition=True)
    @lru_cache(None)
    def final_pnos():
        if pno_data is not None: return pno_data['final_selection']
        from final_pno_mixed import analytic_final_pno_inputs
        return analytic_final_pno_inputs(w,u,v)['final_selection']
    @lru_cache(None)
    def electronic(key):
        if key in ('occupied_density','fock'):
            result = rhf_second()[0]
            mixed = result.density if key=='occupied_density' else result.fock
            return MixedBlock(lower[key],lower_u[key],lower_v[key],mixed)
        if key == 'lmo':
            from localization_mixed import molecular_localization
            kind = 'BOYS' if w.variable('DLPNO RESPONSE BOYS') else 'PIPEK_MEZEY'
            frame = molecular_localization(ref,lower['lmo'],kind,u,v,rhf_second()[2])
            np.testing.assert_allclose(frame.u,lower_u['lmo'],atol=2.e-8,rtol=0.)
            np.testing.assert_allclose(frame.v,lower_v['lmo'],atol=2.e-8,rtol=0.)
            return frame
        if key == 'parent':
            from psi4.driver.procrouting.response.dlpno_frame_curvature import mixed_virtual_parent
            density = electronic('occupied_density')
            frame = mixed_virtual_parent(lambda panel:mixed_product('mn,na->ma',density,panel),
                                         overlap,lower['parent'])
            np.testing.assert_allclose(frame.u,lower_u['parent'],atol=2.e-8,rtol=0.)
            np.testing.assert_allclose(frame.v,lower_v['parent'],atol=2.e-8,rtol=0.)
            return frame
        if isinstance(key,tuple) and key[0]=='pno':
            frame = final_pnos()[key[1]].frame
            np.testing.assert_allclose(frame.value,lower[key],atol=2.e-8,rtol=0.)
            np.testing.assert_allclose(frame.u,lower_u[key],atol=2.e-8,rtol=0.)
            np.testing.assert_allclose(frame.v,lower_v[key],atol=2.e-8,rtol=0.)
            return frame
        if trial_ccsd and isinstance(key,tuple):
            if key[0] in ('t1','t2'):
                return MixedBlock(lower[key],lower_u[key],lower_v[key],np.zeros_like(lower[key]))
            if key[0] == 'tno':
                i,j,k = r = key[1]
                pairs = ((i,j),(j,k),(i,k))
                tno = mixed_tno_frame(r,electronic('parent'),overlap,
                        {p:electronic(('pno',p)) for p in pairs},{p:electronic(('t2',p)) for p in pairs},
                        lower[key])
                np.testing.assert_allclose(tno.u,lower_u[key],atol=2.e-8,rtol=0.)
                np.testing.assert_allclose(tno.v,lower_v[key],atol=2.e-8,rtol=0.)
                return tno
        return MixedBlock(lower[key],lower_u[key],lower_v[key],lower_uv(key))
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
    return electronic,overlap,local_df


def molecular_inputs(w,rows,u,v,lower,lower_u,lower_v,lower_uv,*,trial_ccsd=False,pno_data=None,providers=None):
    electronic,overlap,local_df = (molecular_blocks(w,u,v,lower,lower_u,lower_v,lower_uv,trial_ccsd,pno_data)
                                  if providers is None else providers)
    lmo,fock = electronic('lmo'),electronic('fock')
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


def retained_inputs(w,u,v,base,lower_u,lower_v,lower_uv,affine,pno_data=None,providers=None):
    electronic,s,df = (molecular_blocks(w,u,v,base,lower_u,lower_v,lower_uv,affine,pno_data)
                       if providers is None else providers)
    lmo = electronic('lmo')
    pairs = {k[1]:electronic(k) for k in base if isinstance(k,tuple) and k[0]=='pno'}
    singles = [electronic(('t1',i)) for i in range(lmo.value.shape[1])]
    doubles = {p:electronic(('t2',p)) for p in pairs}
    def pair_fit(p):
        i,j = p
        _,_,state = psi4.core.dlpno_ccsd_df_pair_state(w,i,j)
        m,q = df(tuple(state['AUX'].np[:,0].astype(int)))
        def fitted(a):
            return mixed_product('m,Qmn,nb->Qb',lmo.slice((slice(None),a)),q,pairs[p])
        return m,fitted(i),fitted(j)
    return lmo,pairs,s,electronic('fock'),singles,doubles,pair_fit


def retained_gradient(w,u,lower):
    """Independent first derivative of the ordered native energy on the relaxed path."""
    ref = w.reference_wavefunction()
    atom,xyz = divmod(u,3)
    mints = psi4.core.MintsHelper(ref)
    s,ds = ref.S().np,mints.ao_oei_deriv1('OVERLAP',atom)[xyz].np
    m,q = raw(ref,ref.get_basisset('DF_BASIS_CC'))
    dm = mints.ao_metric_deriv1(atom,'DF_BASIS_CC')[xyz].np
    dq = mints.ao_3center_deriv1(atom,'DF_BASIS_CC')[xyz].np
    l,dl = w.variable('DLPNO RESPONSE ACTIVE LMOS').np,lower['lmo']
    f,df = ref.Fa().np,lower['fock']
    frames,t2 = blocks(w,'FINAL_FRAME'),blocks(w,'CCSD_DOUBLES')
    t1 = []
    gradient = 0.
    for i in range(l.shape[1]):
        dom,ts,_ = psi4.core.dlpno_ccsd_df_pair_state(w,i,i)
        t = ts[dom.index(i)].np.ravel()
        t1.append(t)
        y,dy = frames[i,i],lower['pno',(i,i)]
        fov = l[:,i]@f@y
        dfov = dl[:,i]@f@y + l[:,i]@df@y + l[:,i]@f@dy
        gradient += 2*(dfov@t + fov@lower['t1',i])
    for i,j in frames:
        p = i,j
        y,dy = frames[p],lower['pno',p]
        _,_,state = psi4.core.dlpno_ccsd_df_pair_state(w,i,j)
        aux = state['AUX'].np[:,0].astype(int)
        def ov(a):
            return l[:,a]@q[aux]@y, dl[:,a]@q[aux]@y+l[:,a]@dq[aux]@y+l[:,a]@q[aux]@dy
        ix = np.ix_(aux,aux)
        k,dk = pair_exchange_tangent(m[ix],dm[ix],*ov(i),*ov(j))
        def single(a):
            z,dz = frames[a,a],lower['pno',(a,a)]
            ov = y.T@s@z
            dov = dy.T@s@z+y.T@ds@z+y.T@s@dz
            return ov@t1[a],dov@t1[a]+ov@lower['t1',a]
        ti,dti = single(i)
        tj,dtj = single(j)
        tau = t2[p]+np.outer(ti,tj)
        dtau = lower['t2',p]+np.outer(dti,tj)+np.outer(ti,dtj)
        gradient += (1 if i==j else 2)*(np.vdot(tau,2*dk-dk.T)+np.vdot(dtau,2*k-k.T))
    return gradient


def check_retained_adjoint(w,adjoint,base):
    """Independently differentiate the retained energy w.r.t. the packed t1/t2."""
    ref = w.reference_wavefunction()
    metric,q = raw(ref,ref.get_basisset('DF_BASIS_CC'))
    l,f,s = base['lmo'],base['fock'],ref.S().np
    one = [2*l[:,i]@f@base['pno',(i,i)] for i in range(l.shape[1])]
    two = {}
    for p in adjoint.doubles:
        i,j = p
        y = base['pno',p]
        _,_,state = psi4.core.dlpno_ccsd_df_pair_state(w,i,j)
        aux = state['AUX'].np[:,0].astype(int)
        qi,qj = l[:,i]@q[aux]@y,l[:,j]@q[aux]@y
        k = np.linalg.solve(metric[np.ix_(aux,aux)],qi).T@qj
        spin = 2*k-k.T
        si,sj = y.T@s@base['pno',(i,i)],y.T@s@base['pno',(j,j)]
        mult = 1 if i==j else 2
        one[i] += mult*si.T@spin@sj@base['t1',j]
        one[j] += mult*sj.T@spin.T@si@base['t1',i]
        two[p] = 2*spin
    matrix = psi4.core.Matrix.from_array
    j1,j2 = psi4.core.dlpno_ccsd_amplitude_action(w,[matrix(x[:,None]) for x in adjoint.singles],
                                                  {p:matrix(x) for p,x in adjoint.doubles.items()},True)
    err = max([np.max(abs(x.np.ravel()+y)) for x,y in zip(j1,one)]
              + [np.max(abs(j2[p].np+y)) for p,y in two.items()])
    print('retained CCSD adjoint residual',err,flush=True)
    assert err < 2.e-8


def trial_tno_frame(triplet, parent, metric, pair_frames, doubles, reference_frame, reference_metric):
    """Rebuild a fixed-rank triplet subspace on the trial CCSD path and polar-align it.

    parent is the orthonormal PAO span at the sample; pair frames and doubles
    use the same occupied labeling as the reference triplet.
    """
    i,j,k = triplet
    density = np.zeros((parent.shape[1],)*2)
    for p in ((i,j),(j,k),(i,k)):
        t = doubles[p]
        spin = 2*t-t.T
        projected = pair_frames[p].T@metric@parent
        density += projected.T@(spin@t.T+spin.T@t)@projected / (3*(1+(p[0]==p[1])))
    values,vec = np.linalg.eigh(density)
    kept = reference_frame.shape[1]
    if not 0 < kept <= len(values):
        raise ValueError(f"Invalid trial TNO rank for {triplet}.")
    if kept < len(values) and values[-kept]-values[-kept-1] <= 64*np.finfo(float).eps*max(abs(values)):
        raise ValueError(f"Unresolved trial TNO occupation boundary for {triplet}.")
    frame = parent@vec[:,-kept:]
    a,s,b = np.linalg.svd(frame.T@reference_metric@reference_frame)
    if min(s) < np.sqrt(np.finfo(float).eps):
        raise ValueError(f"Singular trial TNO frame alignment for {triplet}.")
    return frame@(a@b)


def trial_mixed_inputs(w, rows, u, v, h, model_at, data, lower_u, lower_v, lower_uv, pno_data=None):
    """Independent stencil for trial TNOs and geometry stencil for the residual."""
    base = data((0.,)*9).electronic
    pairs = list(data((0.,)*9).pair_rotations)
    def blocks(lower):
        return [lower['t1',i] for i in range(base['lmo'].shape[1])], {p:lower['t2',p] for p in pairs}
    def evaluator(delta):
        state = data(delta)
        return partial(transported_ccsd_residual,model_at(delta),order=state.order,
                       signs=state.signs,rotations=state.pair_rotations)
    nodes = mixed_stencil(u,v,h)
    mixed_residual = semi_analytic_ccsd_mixed_source(
        evaluator((0.,)*9),blocks(base),blocks(lower_u),blocks(lower_v),
        ((du,dv,weight,evaluator(delta)) for delta,du,dv,weight in nodes))
    frame_uv = {r:np.zeros_like(row[2].np) for r,row in rows.items()}
    for delta,du,dv,weight in nodes:
        model = model_at(delta)
        state = data(delta)
        def affine(key):
            return base[key] + du*lower_u[key] + dv*lower_v[key]
        t2 = {p:affine(('t2',p)) for p in pairs}
        other = model.reference_wavefunction()
        parent = other.Ca_subset('AO','VIR').np
        frames = {p:state.electronic['pno',p] for p in pairs}
        for r in rows:
            trial = trial_tno_frame(r,parent,other.S().np,frames,t2,rows[r][2].np,w.reference_wavefunction().S().np)
            if not any(delta):
                np.testing.assert_allclose(trial,rows[r][2].np,atol=2.e-8,rtol=0.)
            frame_uv[r] += weight*trial
    def adjusted(key):
        if isinstance(key,tuple):
            if key[0] in ('t1','t2'):
                return np.zeros_like(base[key])
            if key[0] == 'tno':
                return frame_uv[key[1]]
        return lower_uv(key)
    return molecular_inputs(w,rows,u,v,base,lower_u,lower_v,adjusted,pno_data=pno_data),mixed_residual


def check_residual_replay(w, state):
    lower = state.electronic
    one = [lower['t1',i] for i in range(len(state.order))]
    two = {p:lower['t2',p] for p in state.pair_rotations}
    def replay(one,two):
        return transported_ccsd_residual(w,one,two,state.order,state.signs,state.pair_rotations)
    pair = next(p for p in two if p[0]!=p[1])
    def snapshot():
        _,_,blocks = psi4.core.dlpno_ccsd_df_pair_state(w,*pair)
        return {key:x.np.copy() for key,x in blocks.items()}
    before = snapshot()
    direction = [np.full_like(x,.01) for x in one]
    shifted,_ = replay([x+y for x,y in zip(one,direction)],two)
    assert max(np.max(abs(x)) for x in shifted) > 1.e-5
    after = snapshot()
    for key in before:
        np.testing.assert_allclose(before[key],after[key],atol=1.e-13,rtol=0.)
    base1,base2 = replay(one,two)
    assert max(np.max(abs(x)) for x in [*base1,*base2.values()]) < 1.e-8


def benchmark(frozen, coordinates=range(9), *, complete_spaces=False, coarse_step=.002, include_pno=False):
    psi4.set_options({'freeze_core':frozen})
    w = calculation(BASE)
    rows = psi4.core.dlpno_triples_response_state(w)
    v = 5  # H1 z
    lower_v = {}
    column_row = check(frozen,row_at=divmod(v,3),reference=w,lower_order=lower_v)
    column = TriplesHessianColumn(column_row)
    ccsd_adjoint = TriplesCCSDAdjoint(w)
    retained = RetainedCCSDCurvature(w,ccsd_adjoint)
    from final_pno_mixed import analytic_final_pno_inputs
    if include_pno:
        from pno_curvature_water import sampled_entry
        from psi4.driver.procrouting.response.dlpno_pno_curvature import PNOTruncationCurvature
        pno_adjoint = PNOTruncationCurvature(w)
    eta1,eta2 = psi4.core.dlpno_triples_ccsd_source(w)
    eta1,eta2 = [x.np.ravel() for x in eta1], {p:x.np for p,x in eta2.items()}

    @lru_cache(maxsize=4)
    def model_at(delta):
        model = calculation(BASE+np.array(delta).reshape(3,3)) if any(delta) else w
        if complete_spaces:
            corrections = ('DLPNO LMP2 WEAK PAIR ENERGY','DLPNO SC-LMP2 PAIR ENERGY',
                           'DLPNO DIPOLE ENERGY','DLPNO PNO TRUNCATION ERROR')
            for key in corrections:
                np.testing.assert_allclose(model.variable(key),0.,atol=1.e-10,rtol=0.,err_msg=key)
        return model
    @lru_cache(None)
    def data(delta):
        return aligned_inputs(model_at(delta),w,rows)
    @lru_cache(None)
    def along_column(h):
        d = np.zeros(9)
        d[v] = h
        return calculation(BASE+d.reshape(3,3))
    check_residual_replay(w,data((0.,)*9))
    check_retained_adjoint(w,retained,data((0.,)*9).electronic)
    results = []
    for u in coordinates:
        lower_u = lower_v if u == v else {}
        row = column_row if u == v else check(frozen,row_at=divmod(u,3),reference=w,lower_order=lower_u)
        pno_data = analytic_final_pno_inputs(w,u,v)
        base = data((0.,)*9).electronic
        def forbidden_mixed(key):
            raise AssertionError(f'Analytic CCSD residual requested a sampled mixed input: {key}')
        replay = mixed_ccsd_residual(w,*molecular_blocks(w,u,v,base,lower_u,lower_v,
                                                       forbidden_mixed,True,pno_data))
        residual = [x.uv for x in replay[0]],{p:x.uv for p,x in replay[1].items()}
        for component in (1,2):
            np.testing.assert_allclose(np.concatenate([x[component].ravel() for x in
                [*replay[0],*replay[1].values()]]),0.,atol=2.e-9,rtol=0.)
        residual_checks = []
        relax = column.relaxation(row)
        reverse = TriplesHessianColumn(row).relaxation(column_row)
        np.testing.assert_allclose(relax,reverse,rtol=0.,atol=2.e-11)
        elements,stationary,reference,tno_errors = [],[],[],[]
        constraints,source_errors = [],[]
        retained_values,retained_direct,retained_reference = [],[],[]
        for h in (coarse_step,coarse_step/2):
            projected,lower_uv = mixed_coefficients(data,u,v,h)
            # Use analytic AO and DF derivatives with lower-order mixed response
            # from the native displaced states. Separately check the result
            # against differentiating the already-assembled native inputs.
            inputs = molecular_inputs(w,rows,u,v,data((0.,)*9).electronic,lower_u,lower_v,lower_uv,pno_data=pno_data)
            element = column.element(row,*inputs)
            direct = column.element(row,*projected)
            print('mixed provider difference',frozen,u,h,element-direct,flush=True)
            assert abs(element-direct) < 5.e-6
            elements.append(element)
            trial_fd,sampled_residual = trial_mixed_inputs(w,rows,u,v,h,model_at,data,lower_u,lower_v,lower_uv,pno_data)
            residual_checks.append(np.concatenate(
                [a-b for a,b in zip(residual[0],sampled_residual[0])]
                + [(a-sampled_residual[1][p]).ravel() for p,a in residual[1].items()]))
            matrix = psi4.core.Matrix.from_array
            nocc = len(residual[0])
            jt1,jt2 = psi4.core.dlpno_ccsd_amplitude_action(w,
                [matrix(lower_uv(('t1',i))[:,None]) for i in range(nocc)],
                {p:matrix(lower_uv(('t2',p))) for p in residual[1]})
            oracle = ccsd_weighted_dot(eta1,eta2,[lower_uv(('t1',i)) for i in range(nocc)],
                                       {p:lower_uv(('t2',p)) for p in eta2})
            source_errors.append(ccsd_adjoint.correction(*residual)-oracle)
            constraints.append(np.concatenate(
                [x.np.ravel()+y for x,y in zip(jt1,residual[0])]
                + [(jt2[p].np+y).ravel() for p,y in residual[1].items()]))
            def remaining_mixed(key):
                if key in ('occupied_density','fock','lmo','parent') or (
                        isinstance(key,tuple) and key[0] in ('t1','t2','tno','pno')):
                    raise AssertionError("The CCSD-stationary path requested an eliminated mixed input.")
                return lower_uv(key)
            trial = molecular_inputs(w,rows,u,v,data((0.,)*9).electronic,lower_u,lower_v,remaining_mixed,
                                     trial_ccsd=True,pno_data=pno_data)
            tno_comparison = column.element(row,*trial_fd)-column.element(row,*trial)
            print('TNO mixed curvature comparison',frozen,u,h,tno_comparison,flush=True)
            tno_errors.append(tno_comparison)
            adjoint_value = column.ccsd_stationary_element(row,*trial,ccsd_adjoint,residual)
            stationary.append(adjoint_value)
            print('CCSD adjoint correction',frozen,u,h,ccsd_adjoint.correction(*residual),adjoint_value-element,flush=True)
            base = data((0.,)*9).electronic
            direct_cc = project_retained_ccsd_energy(*retained_inputs(w,u,v,base,lower_u,lower_v,lower_uv,False,pno_data))
            affine_inputs = retained_inputs(w,u,v,base,lower_u,lower_v,remaining_mixed,True,pno_data)
            retained_values.append(retained.element(*affine_inputs,residual))
            retained_direct.append(direct_cc.uv)
            derivatives,cc_gradients = [],[]
            for sign in (1,-1):
                model = along_column(sign*h)
                moved = {}
                d = check(frozen,row_at=divmod(u,3),reference=model,lower_order=moved)
                derivatives.append(stationary_gradient(d))
                cc_gradients.append(retained_gradient(model,u,moved))
            retained_reference.append((cc_gradients[0]-cc_gradients[1])/(2*h))
            reference.append((derivatives[0]-derivatives[1])/(2*h))
        residual_error = np.max(abs(4*residual_checks[1]-residual_checks[0]))/3
        print('analytic CCSD residual vs stencil',frozen,u,residual_error,flush=True)
        # Componentwise source validation uses transported analytic first tangents
        # in ccsd_mixed.py. This short energy-verification stencil can amplify native
        # integral/frame roundoff; the mixed constraint and adjoint checks below
        # remain independent requirements for this assembled calculation.
        source_error = abs(4*source_errors[1]-source_errors[0])/3
        print('triples-only adjoint contraction error',frozen,u,source_error,flush=True)
        assert source_error < 1.e-7
        constraint_error = np.max(abs(4*constraints[1]-constraints[0]))/3
        np.testing.assert_allclose((4*tno_errors[1]-tno_errors[0])/3,0.,atol=2.e-7,rtol=0.)
        print('mixed CCSD constraint error',frozen,u,constraint_error,flush=True)
        assert constraint_error < 5.e-5
        estimated = (4*stationary[1]-stationary[0])/3
        previous = (4*elements[1]-elements[0])/3
        print('CCSD elimination check',frozen,u,estimated,previous,estimated-previous,flush=True)
        np.testing.assert_allclose(estimated,previous,atol=2.e-7,rtol=0.)
        numerical = (4*reference[1]-reference[0])/3
        print('triples curvature',frozen,u,relax,estimated,numerical,estimated-numerical,flush=True)
        cc,cc_direct,cc_reference = ((4*values[1]-values[0])/3 for values in
                                     (retained_values,retained_direct,retained_reference))
        print('retained CCSD curvature',frozen,u,cc,cc_direct,cc_reference,cc-cc_reference,flush=True)
        np.testing.assert_allclose(cc,[cc_direct,cc_reference],atol=2.e-6,rtol=0.)
        output = (relax,estimated,numerical,cc,cc_reference)
        if include_pno:
            pno = pno_adjoint.element(pno_data['initial'],pno_data['final'],pno_data['source'])
            # Displaced inputs are an independent oracle, never the assembled value.
            values = [sampled_entry(w,model_at,u,v,h,pno_adjoint) for h in (coarse_step,coarse_step/2)]
            pno_sampled,pno_direct = (4*np.asarray(values[1])-np.asarray(values[0]))/3
            np.testing.assert_allclose(pno,[pno_sampled,pno_direct],rtol=0,atol=3.e-7)
            print('assembled retained CCSD + triples + PNO curvature',frozen,u,cc+estimated+pno,
                  'analytic PNO',pno,'sampled/direct errors',pno-np.array([pno_sampled,pno_direct]),flush=True)
            output += (pno,)
        results.append(output)
    result = np.asarray(results)
    np.testing.assert_allclose(result[:,1],result[:,2],rtol=0.,atol=2.e-7)
    if list(coordinates) == list(range(9)):
        np.testing.assert_allclose(result[:,3].reshape(3,3).sum(axis=0),0.,rtol=0.,atol=2.e-6)
        np.testing.assert_allclose(result[:,2].reshape(3,3).sum(axis=0),0.,rtol=0.,atol=2.e-6)
    return result


def canonical_correlation_columns(frozen):
    """Difference analytic gradients to isolate the canonical CCSD and (T) blocks."""
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
            ccsd = grad('ccsd')
            total = grad('ccsd(t)')
            grads.append(np.stack((total-ccsd,ccsd-grad('scf'),total)))
        estimates.append((grads[0]-grads[1])/(2*h))
    columns = (4*estimates[-1]-estimates[-2])/3
    np.testing.assert_allclose(columns.sum(axis=1),0.,atol=1.e-8,rtol=0.)
    print('canonical triples, CCSD correlation and total columns',frozen,columns,flush=True)
    return columns


def canonical_reference(frozen):
    return canonical_correlation_columns(frozen)[0]


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(scf__diis=False,scf_initial_accelerator='NONE',basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_cc='cc-pvdz-ri',
                          df_basis_scf='cc-pvdz-jkfit',e_convergence=1.e-13,d_convergence=1.e-13,
                          r_convergence=1.e-11,solver_convergence=1.e-12,dlpno_maxiter=600,
                          local_convergence=1.e-14,local_gradient_convergence=1.e-12,
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
        tcol,cccol,_ = canonical_columns[frozen] = canonical_correlation_columns(frozen)
        print('max local/canonical (T) difference',frozen,
              np.max(abs(local[:,1]-tcol.ravel())),flush=True)
        print('max retained/canonical CCSD difference',frozen,
              np.max(abs(local[:,3]-cccol.ravel())),flush=True)
    # Check the displacement sensitivity of the all-electron O x and H1 z entries.
    benchmark(False,(0,5),coarse_step=.003)
    # Keep the nonzero pair cutoff. On this molecule it retains all pairs.
    psi4.set_options(dict(t_cut_pno=0.,t_cut_pno_mp2=0.,t_cut_tno=0.,t_cut_do_triples=0.,
                          t_cut_mkn_triples=0.,f_cut=0.,f_cut_t=0.))
    for frozen in (True,False):
        output_rows = (1,5)  # O y and H1 z
        result = benchmark(frozen,output_rows,complete_spaces=True)
        can = canonical_columns[frozen].reshape(3,9)[:,output_rows].T
        np.testing.assert_allclose(result[:,[1,3]],can[:,:2],atol=2.e-7,rtol=0.)
        hf = psi4.hessian('scf',molecule=molecule(BASE),dertype=2).np[list(output_rows),5]
        assembled = hf + result[:,3] + result[:,1]
        print('complete-space total components',frozen,assembled,can[:,2],assembled-can[:,2],flush=True)
        np.testing.assert_allclose(assembled,can[:,2],atol=2.e-7,rtol=0.)
