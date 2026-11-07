"""Export the complete Hessian equation record without altering its mathematics."""
import argparse
from pathlib import Path
import re


def render(source):
    body = source.split('\\begin{document}\n',1)[1].rsplit('\\end{document}',1)[0]
    body = body.replace('\\maketitle\n','',1).strip()
    section = 0
    subsection = 0
    lines = []
    for line in body.splitlines():
        match = re.fullmatch(r'\\(subsection|section)\{(.*)\}',line)
        if match:
            kind,title = match.groups()
            if kind=='section':
                section += 1
                subsection = 0
                line = f'{section}. {title}'
            else:
                subsection += 1
                line = f'{section}.{subsection}. {title}'
        lines.append(line)
    result = ('DLPNO-CCSD(T) NUCLEAR HESSIAN: COMPLETE WORKING EQUATIONS\n'
              'Quantum Chemistry Grand Battleplan — 7 November 2026\n\n'
              'Plain-text companion to DLPNO_HESSIAN_EQUATIONS.tex.\n'
              'All mathematical notation, equation labels and cross-references are\n'
              'retained in LaTeX form to preserve the complete derivations exactly.\n'
              'Section headings are numbered for navigation. The chronological\n'
              'milestone/validation ledger is DLPNO_HESSIAN_WORKING_MEMORY.md.\n'
              'Regenerate with tests/dlpno-hessian-model/export_equations.py.\n\n'
              + '\n'.join(lines) + '\n')
    # No display-equation block may be dropped or changed during export.
    pattern = r'\\begin\{(align|equation)\}(.*?)\\end\{\1\}'
    assert re.findall(pattern,source,re.S) == re.findall(pattern,result,re.S)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Fail if the tracked text export is stale.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    text = render((root/'DLPNO_HESSIAN_EQUATIONS.tex').read_text())
    target = root/'DLPNO_HESSIAN_EQUATIONS.txt'
    if args.check:
        if not target.exists() or target.read_text()!=text:
            raise SystemExit('DLPNO_HESSIAN_EQUATIONS.txt needs regeneration.')
    else:
        target.write_text(text)
    print(f'Complete equation export: {len(text.splitlines())} lines; all display equations preserved.')
