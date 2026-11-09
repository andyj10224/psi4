"""Compare published-mode residuals against a probe built from --ref BASE.

Usage: python lambda_reconciliation_legacy.py OLD_PROBE CURRENT_PROBE
Each extension is loaded in its own process. At Fov=0 the original energy-mode
equations must be unchanged, including finite PNO and screened-pair choices.
At nonzero Fov, the only change from base 13d74e0 must be the added 2 Fia
singles source, with exactly zero change to the doubles residual.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

if sys.argv[1] == '--dump':
    sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
    import numpy as np
    from lambda_reconciliation_fixture import fixture, screened, evaluate

    rows = []
    graphs = [(3, []), (3, [(0, 2), (2, 0)]),
              (4, [(0, 2), (2, 0), (1, 3), (3, 1)]), (4, [(0, 3), (3, 0)])]
    for seed in (17, 91):
        for no, missing in graphs:
            for truncated in (False, True):
                for local_fit in (False, True):
                    for local_pao in (False, True):
                        for fov in (False, True):
                            x = screened(fixture(seed=seed, no=no, truncated=truncated,
                                                 local_fit=local_fit, local_pao=local_pao,
                                                 fov=fov), missing)
                            out = evaluate(x, 'published')
                            d=x[0];source=[]
                            for i in range(no):
                                p=d['i_j_to_ij_'][i][i]
                                source.append(2*d['F_lmo_pao_'][i,d['lmopair_to_paos_'][p]]@d['X_pno_'][p])
                            source += [np.zeros_like(a).ravel() for a in out[1]]
                            rows.append({'fov':fov,
                                         'residual':np.concatenate([a.ravel() for b in out for a in b]).tolist(),
                                         'source':np.concatenate(source).tolist()})
    Path(sys.argv[3]).write_text(json.dumps(rows))
else:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with tempfile.TemporaryDirectory(prefix='lambda-legacy-') as tmp:
        values = []
        for i, probe in enumerate(sys.argv[1:]):
            output = Path(tmp) / f'{i}.json'
            subprocess.run([sys.executable, __file__, '--dump', probe, str(output)], check=True)
            values.append(json.loads(output.read_text()))
    assert len(values[0]) == len(values[1]) == 128
    errors={False:0.,True:0.}
    for a,b in zip(*values):
        assert a['fov']==b['fov'] and len(a['residual'])==len(b['residual'])
        error=max(abs(v-u-s) for u,v,s in zip(a['residual'],b['residual'],b['source']))
        errors[a['fov']]=max(errors[a['fov']],error)
    assert max(errors.values()) < 2.e-14, errors
    print(json.dumps({'legacy_fixtures':64,'max_legacy_difference':errors[False],
                      'nonzero_fov_fixtures':64,'max_2fia_source_error':errors[True]},indent=2))
