"""Compare published-mode residuals against a probe built from --ref BASE.

Usage: python lambda_reconciliation_legacy.py OLD_PROBE CURRENT_PROBE
Each extension is loaded in its own process. At Fov=0 the original energy-mode
equations must be unchanged, including finite PNO and screened-pair choices.
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
                        x = screened(fixture(seed=seed, no=no, truncated=truncated,
                                             local_fit=local_fit, local_pao=local_pao,
                                             fov=False), missing)
                        rows.append(np.concatenate([a.ravel() for b in evaluate(x, 'published') for a in b]).tolist())
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
    assert len(values[0]) == len(values[1]) == 64
    assert all(len(a) == len(b) for a, b in zip(*values))
    error = max(abs(u-v) for a, b in zip(*values) for u, v in zip(a, b))
    assert error < 2.e-14, error
    print(json.dumps({'legacy_fixtures': 64, 'max_legacy_difference': error}, indent=2))
