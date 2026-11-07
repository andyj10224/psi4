"""Nonplanar screened molecular fixtures with frozen cores and mixed TNO ranks.

These deliberately finite-cutoff fixtures extend the fragment-chain tests.
Individual PAO-pair screening is off so the coverage checks do not conflate
triples/core response with its separately documented rotation defect.
"""
import numpy as np
import psi4
from localization_integration import configure

SYSTEM_SIZES={'water_h2':15,'ammonia_h2':18}


def fixture(system,localization):
    if system not in SYSTEM_SIZES:raise ValueError('Unknown screened molecular fixture.')
    configure()
    psi4.set_options(dict(freeze_core=True,save_jk=True,t_cut_pairs_mp2=1.e-6,t_cut_pairs=1.e-6,
        t_cut_pre=1.e-6,t_cut_do_ij=1.e-5,t_cut_do=.01,t_cut_do_pre=.03,t_cut_mkn=.001,
        t_cut_clmo=.01,t_cut_cpao=.01,t_cut_do_uv=0.,dlpno__debug=0,dlpno_local_orbitals=localization,
        t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-3,t_cut_tno=1.e-4,
        t_cut_do_triples=.01,t_cut_mkn_triples=.001,
        scf__ints_tolerance=1.e-14,scf__maxiter=600,e_convergence=1.e-14,d_convergence=1.e-14,
        solver_convergence=1.e-14,local_gradient_convergence=1.e-13))
    atoms=([('O',0.,0.,0.),('H',.76,0.,.58),('H',-.72,.03,.64)] if system=='water_h2' else
           [('N',0.,0.,0.),('H',.94,0.,.31),('H',-.46,.81,.36),('H',-.48,-.80,.32)])
    distance=8. if system=='water_h2' else 8.5
    atoms += [('H',distance,.85,.3),('H',distance+.21,1.17,.94)]
    mol=psi4.geometry('0 1\n'+'\n'.join(f'{a} {x} {y} {z}' for a,x,y,z in atoms)+
                     '\nunits angstrom\nsymmetry c1\nno_com\nno_reorient')
    _,w=psi4.energy('dlpno-ccsd(t)',molecule=mol,return_wfn=True)
    coverage(w)
    return mol,w


def coverage(w):
    """Assert actual retained domains, ranks and nonzero rectangular couplings."""
    ref=w.reference_wavefunction();L=w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    nocc=L.shape[1];nbf=w.basisset().nbf();naux=w.get_basisset('DF_BASIS_CC').nbf()
    domains=psi4.core.dlpno_response_domain_state(w)
    initial,blocks=psi4.core.dlpno_first_pno_state(w)
    final=psi4.core.dlpno_lmp2_blocks(w,'FINAL_FRAME')
    rows=psi4.core.dlpno_triples_response_state(w);basis=psi4.core.dlpno_triples_basis_state(w)
    ranks={r:row[2].np.shape[1] for r,row in rows.items()};foo=L.T@ref.Fa().np@L
    rectangular=[];nonzero=0
    for r,row in rows.items():
        for l,m,i,j,k,_ in row[5]:
            q=tuple(sorted((i,j,k)))
            if ranks[r]!=ranks[q]:
                rectangular.append(dict(source=list(r),target=list(q),occupied=[int(l),int(m)],ranks=[ranks[r],ranks[q]]))
                nonzero+=abs(foo[l,m])>1.e-8
    pair_ranks=[dict(pair=p,parent=blocks[tuple(p)]['PARENT'].np.shape[1],
                     first=blocks[tuple(p)]['FRAME'].np.shape[1],final=final[tuple(p)].np.shape[1])
                for p in initial['PAIRS']]
    triplets=[dict(indices=list(r),rank=ranks[r],parent=basis[r]['PARENT'].np.shape[1],
                   paos=int(basis[r]['PAOS'].np.size),auxiliary=len(row[1]),edges=len(row[5])) for r,row in rows.items()]
    distinct=sum(len(set(r))==3 for r in rows)
    offdiagonal=sum(i!=j for i,j in domains['PAIRS'])
    excluded,_=psi4.core.dlpno_screened_pair_state(w)
    xyz=w.molecule().geometry().np
    assert ref.nfrzc()>0 and nocc>=3 and offdiagonal>0
    assert len(domains['PAIRS'])<nocc*(nocc+1)//2
    assert distinct>0 and len(set(ranks.values()))>1 and nonzero>0
    assert any(len(x)<nbf for x in domains['PAIR_PAO']) and any(len(x)<naux for x in domains['PAIR_AUX'])
    assert any(t['paos']<nbf and t['auxiliary']<naux and t['rank']<t['parent'] for t in triplets)
    assert any(p['first']<p['parent'] for p in pair_ranks) and any(p['final']<p['first'] for p in pair_ranks)
    assert np.linalg.svd(xyz-xyz.mean(axis=0),compute_uv=False)[-1]>.01
    assert excluded['SCMP2_PAIRS'] and excluded['DIPOLE_PAIRS']
    for name in ('DLPNO INITIAL PNO TRUNCATION ENERGY','DLPNO FINAL PNO TRUNCATION ENERGY',
                 'DLPNO SC-LMP2 PAIR ENERGY','DLPNO DIPOLE ENERGY','(T) CORRECTION ENERGY'):
        assert abs(w.variable(name))>1.e-10,name
    return dict(frozen_orbitals=ref.nfrzc(),active_occupied=nocc,nbf=nbf,naux=naux,
                retained_offdiagonal_pairs=offdiagonal,pair_ranks=pair_ranks,triplets=triplets,
                distinct_triplets=distinct,rectangular_edges=rectangular,nonzero_rectangular_edges=int(nonzero),
                excluded_scmp2_pairs=excluded['SCMP2_PAIRS'],dipole_pairs=excluded['DIPOLE_PAIRS'],
                nonplanar=True,individual_pao_pair_screen=False)
