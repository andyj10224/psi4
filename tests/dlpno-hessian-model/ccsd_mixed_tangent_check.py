"""Independent mixed CCSD source oracle from transported analytic first tangents."""
import numpy as np
import psi4
from scipy.linalg import solve_sylvester
from lmp2_tangent import check, blocks
from triples_curvature import aligned_inputs


def transported_tangent(model, reference, rows, frozen, direction):
    """Differentiate the reference polar alignment, not just the native amplitudes.

    A=Y(native)^T S(reference) Y(reference)=Q H. For the native local
    polar-frame first derivative, Q'=Q Omega and H Omega+Omega H=
    Q^T A' - A'^T Q. This rotates both the amplitudes and their derivatives.
    """
    lower = {}
    check(frozen,row_at=divmod(direction,3),reference=model,lower_order=lower)
    state = aligned_inputs(model,reference,rows)
    y0 = blocks(reference,'FINAL_FRAME')
    S0 = reference.reference_wavefunction().S().np
    y = blocks(model,'FINAL_FRAME')
    t = blocks(model,'CCSD_DOUBLES')
    one,two = [None]*len(state.order),{}
    for p,Q in state.pair_rotations.items():
        i,j = p
        a,b = state.order[i],state.order[j]
        q = min(a,b),max(a,b)
        A = y[q].T@S0@y0[p]
        dA = lower['pno',q].T@S0@y0[p]
        H = Q.T@A
        H = (H+H.T)/2
        Omega = solve_sylvester(H,H,Q.T@dA-dA.T@Q)
        np.testing.assert_allclose(Omega+Omega.T,0.,atol=2.e-12,rtol=0.)
        native,derivative = t[q],lower['t2',q]
        if a>b: native,derivative = native.T,derivative.T
        aligned = Q.T@native@Q
        two[p] = state.signs[i]*state.signs[j]*(Q.T@derivative@Q+Omega.T@aligned+aligned@Omega)
        if i==j:
            dom,ts,_ = psi4.core.dlpno_ccsd_df_pair_state(model,a,a)
            ti = ts[dom.index(a)].np.ravel()
            one[i] = state.signs[i]*(Q.T@lower['t1',a]+Omega.T@(Q.T@ti))
    return one,two


def source_from_tangents(w,rows,model_at,frozen,u,v,h):
    tangent = []
    for step in (h,h/2):
        delta = np.zeros(9)
        delta[v] = step
        plus = transported_tangent(model_at(tuple(delta)),w,rows,frozen,u)
        minus = transported_tangent(model_at(tuple(-delta)),w,rows,frozen,u)
        tangent.append(([ (a-b)/(2*step) for a,b in zip(plus[0],minus[0])],
                        {p:(a-minus[1][p])/(2*step) for p,a in plus[1].items()}))
    tuv = ([ (4*a-b)/3 for a,b in zip(tangent[1][0],tangent[0][0])],
           {p:(4*a-tangent[0][1][p])/3 for p,a in tangent[1][1].items()})
    matrix = psi4.core.Matrix.from_array
    one,two = psi4.core.dlpno_ccsd_amplitude_action(w,[matrix(a[:,None]) for a in tuv[0]],
                                                {p:matrix(a) for p,a in tuv[1].items()})
    return [-a.np.ravel() for a in one],{p:-a.np for p,a in two.items()}
