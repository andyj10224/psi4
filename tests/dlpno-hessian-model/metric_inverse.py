"""Conditioned metric inverse responses against exact rotated spectral inverses."""
import json
from pathlib import Path
import numpy as np
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock,_symmetric_inverse as _inverse


def verify():
    rng=np.random.default_rng(97351)
    q,_=np.linalg.qr(rng.normal(size=(6,6)))
    k,l=(.1*(x-x.T) for x in rng.normal(size=(2,6,6)))
    comm=lambda a,b:a@b-b@a
    records=[]
    for small in (1.e-6,-1.e-6):
        eigenvalues=np.array([small,.02,.2,.7,1.4,2.])
        m=(q*eigenvalues)@q.T;g=(q/eigenvalues)@q.T
        def rotated(x):return MixedBlock(x,comm(k,x),comm(l,x),comm(k,comm(l,x)))
        metric=rotated(m);expected=rotated(g);actual=_inverse(metric)
        errors=[float(np.max(abs(a-b))/max(1.,np.max(abs(b)))) for a,b in zip(actual,expected)]
        residuals=[metric.value@actual.value-np.eye(6),
                   metric.value@actual.u+metric.u@actual.value,
                   metric.value@actual.v+metric.v@actual.value,
                   metric.value@actual.uv+metric.u@actual.v+metric.v@actual.u+metric.uv@actual.value]
        defects=[float(np.max(abs(x-x.T))) for x in actual]
        checks=dict(small_eigenvalue=small,condition=2.e6,relative_errors=errors,
                    inverse_identity_residuals=[float(np.max(abs(x))) for x in residuals],symmetry_defects=defects)
        print(checks,flush=True);records.append(checks)
        np.testing.assert_allclose(errors,0.,atol=2.e-9,rtol=0.)
        np.testing.assert_allclose(checks['inverse_identity_residuals'],0.,atol=5.e-9,rtol=0.)
        np.testing.assert_array_equal(defects,np.zeros(4))
        for component in range(4):
            bad=[x.copy() for x in metric];bad[component][0,1]+=.01
            try:_inverse(MixedBlock(*bad))
            except ValueError as exc:assert 'symmetric' in str(exc)
            else:raise AssertionError('Accepted an asymmetric metric block.')
    Path('metric_inverse_results.json').write_text(json.dumps(dict(passed=True,checks=records),indent=2)+'\n')


if __name__=='__main__':verify()
