"""Export complete gradient or Hessian equations without altering their mathematics."""
import argparse
from pathlib import Path
import re


def render(source, record='hessian'):
    name = record.upper()
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
    # Preserve shorthand definitions too, so the text is usable independently
    # of its TeX preamble (the gradient record defines transpose and pairing).
    macros = [line for line in source.split('\\begin{document}', 1)[0].splitlines()
              if line.startswith('\\newcommand')]
    definitions = ('Mathematical shorthand definitions:\n' + '\n'.join(macros) + '\n\n') if macros else ''
    command = '' if record == 'hessian' else ' --record gradient'
    result = (f'DLPNO-CCSD(T) NUCLEAR {name}: COMPLETE WORKING EQUATIONS\n'
              'Quantum Chemistry Grand Battleplan — 7 November 2026\n\n'
              f'Plain-text companion to DLPNO_{name}_EQUATIONS.tex.\n'
              'All mathematical notation, equation labels and cross-references are\n'
              'retained in LaTeX form to preserve the complete derivations exactly.\n'
              'Section headings are numbered for navigation. The chronological\n'
              f'milestone/validation ledger is DLPNO_{name}_WORKING_MEMORY.md.\n'
              f'Regenerate with tests/dlpno-hessian-model/export_equations.py{command}.\n\n'
              + definitions + '\n'.join(lines) + '\n')
    # No display-equation block may be dropped or changed during export.
    pattern = r'\\begin\{(align|equation)\}(.*?)\\end\{\1\}'
    assert re.findall(pattern,source,re.S) == re.findall(pattern,result,re.S)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', choices=('gradient', 'hessian', 'all'), default='hessian',
                        help='Equation record to export; default preserves the existing Hessian command.')
    parser.add_argument('--check',action='store_true',help='Fail if the tracked text export is stale.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    records = ('gradient', 'hessian') if args.record == 'all' else (args.record,)
    for record in records:
        stem = f'DLPNO_{record.upper()}_EQUATIONS'
        text = render((root/f'{stem}.tex').read_text(), record)
        target = root/f'{stem}.txt'
        if args.check:
            if not target.exists() or target.read_text()!=text:
                raise SystemExit(f'{target.name} needs regeneration.')
        else:
            target.write_text(text)
        print(f'Complete {record} export: {len(text.splitlines())} lines; all display equations preserved.')
