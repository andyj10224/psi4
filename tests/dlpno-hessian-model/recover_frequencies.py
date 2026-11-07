"""Recover harmonic modes from a completed Psi4 .log and matching .out.

Uses NumPy only; no SCF, correlation or Hessian calculation is repeated.
Logged matrices have eight decimal places, so results are reported to 0.01
cm^-1 and are not full-precision derivative-validation references.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

# Psi4's CODATA 2014 values (psi4/include/psi4/physconst.h), and the
# conversion used by psi4/driver/qcdb/vib.py::harmonic_analysis.
AVOGADRO=6.022140857e23
HARTREE_J=4.359744650e-18
LIGHT_MS=299792458.
BOHR_ANGSTROM=.52917721067
WAVENUMBER_FACTOR=np.sqrt(AVOGADRO*HARTREE_J*1.e19)/(2*np.pi*LIGHT_MS*BOHR_ANGSTROM)


def read_hessian(text):
    marker='Return hessian():'
    if marker not in text:
        raise ValueError('No completed Hessian record in the log.')
    tail=text[text.rindex(marker):]
    energy=float(tail.splitlines()[0].split(':',1)[1])
    block=re.search(r'\[\[.*?\]\]',tail,re.S)
    if block is None or '...' in block.group():
        raise ValueError('The logged Hessian is missing, incomplete or abbreviated.')
    # NumPy array_str writes whitespace-separated numbers without commas.
    tokens=block.group().replace('[',' ').replace(']',' ').split()
    values=np.array([float(x) for x in tokens])
    n=int(round(np.sqrt(len(values))))
    if n*n!=len(values) or n%3 or not np.isfinite(values).all():
        raise ValueError('Expected a finite square Cartesian Hessian.')
    return values.reshape(n,n),energy


def read_geometry(text):
    sections=text.split('Geometry (in Angstrom)')
    pattern=re.compile(r'^\s*([A-Z][a-z]?)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+'
                       r'([-+\d.eE]+)\s+([-+\d.eE]+)\s*$')
    for section in reversed(sections[1:]):
        rows=[]
        for line in section.splitlines():
            match=pattern.match(line)
            if match:rows.append(match.groups())
            elif rows:break
        if rows:
            atoms=[r[0] for r in rows]
            xyz=np.array([[float(x) for x in r[1:4]] for r in rows])
            masses=np.array([float(r[4]) for r in rows])
            if not np.isfinite(xyz).all() or not np.isfinite(masses).all() or np.any(masses<=0.):
                raise ValueError('Invalid coordinates or isotope masses in the output.')
            return atoms,xyz,masses
    raise ValueError('No Angstrom geometry with isotope masses found in the output.')


def analyze(hessian,xyz,masses):
    n=len(masses);size=3*n
    if hessian.shape!=(size,size) or xyz.shape!=(n,3):
        raise ValueError('The Hessian and geometry have different atom counts.')
    asymmetry=float(np.max(abs(hessian-hessian.T)))
    if asymmetry>2.e-7:
        raise ValueError(f'Logged Hessian is not symmetric: max difference {asymmetry:.3e}.')
    weights=np.repeat(1/np.sqrt(masses),3)
    mw=weights[:,None]*((hessian+hessian.T)*.5)*weights[None,:]
    centered=xyz-np.average(xyz,axis=0,weights=masses)
    root_mass=np.repeat(np.sqrt(masses),3)[:,None]
    translations=np.tile(np.eye(3),(n,1))*root_mass
    rotations=np.stack([np.cross(axis,centered).ravel() for axis in np.eye(3)],axis=1)*root_mass
    vectors,singular,_=np.linalg.svd(np.column_stack((translations,rotations)),full_matrices=True)
    rank=int(np.count_nonzero(singular>1.e-10*max(singular.max(),1.)))
    vibrational=vectors[:,rank:]
    eigenvalues,modes=np.linalg.eigh(vibrational.T@mw@vibrational)
    signed=lambda x:np.sign(x)*np.sqrt(abs(x))*WAVENUMBER_FACTOR
    return dict(frequencies_cm_inverse=signed(eigenvalues).tolist(),
        imaginary=(eigenvalues<0).tolist(),rigid_motion_rank=rank,
        unprojected_signed_frequencies=signed(np.linalg.eigvalsh(mw)).tolist(),
        hessian_asymmetry=asymmetry,
        cartesian_modes=(weights[:,None]*(vibrational@modes)).reshape(n,3,-1).tolist())


def recover(log,output):
    hessian,energy=read_hessian(log)
    atoms,xyz,masses=read_geometry(output)
    # Catch mismatched runs when both files supply the nuclear repulsion.
    log_nre=re.findall(r'Compute hessian\(\):.*?nre=([-+\d.eE]+)',log)
    out_nre=re.findall(r'Nuclear repulsion\s*=\s*([-+\d.eE]+)',output)
    if log_nre and out_nre and abs(float(log_nre[-1])-float(out_nre[-1]))>1.e-8:
        raise ValueError('The .log and .out nuclear repulsion energies disagree.')
    return dict(energy=energy,atoms=atoms,coordinates_angstrom=xyz.tolist(),masses_amu=masses.tolist(),
        hessian=hessian.tolist(),**analyze(hessian,xyz,masses),
        precision='Hessian recovered from eight-decimal-place log; report frequencies to 0.01 cm^-1.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',required=True)
    parser.add_argument('--output',required=True,help='Matching completed Psi4 .out file')
    parser.add_argument('--json',help='Optional destination for recovered matrix, modes and frequencies')
    args=parser.parse_args()
    result=recover(Path(args.log).read_text(),Path(args.output).read_text())
    for i,value in enumerate(result['frequencies_cm_inverse'],1):
        print(f'Mode {i:3d}: {abs(value):10.2f}{"i" if value<0 else " "} cm^-1')
    print(result['precision'])
    if args.json:Path(args.json).write_text(json.dumps(result,indent=2)+'\n')
