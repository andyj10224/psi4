/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "dlpno.h"

#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/exception.h"

namespace psi {
namespace dlpno {

SharedMatrix orbital_response_action(SharedWavefunction wfn, const std::string& block, SharedMatrix rhs) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("Orbital response requires a gradient-model RHF DLPNO-CCSD(T) wavefunction.");
    return local->orbital_response_action(block,rhs);
}

DLPNOCCSD_T::PairBlocks lmp2_blocks(SharedWavefunction wfn, const std::string& block) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("LMP2 frames require an RHF DLPNO gradient model.");
    return local->lmp2_blocks(block);
}

DLPNOCCSD_T::PairBlocks lmp2_tangent(SharedWavefunction wfn, const DLPNOCCSD_T::PairBlocks& dy,
                                    const DLPNOCCSD_T::PairBlocks& dk, SharedMatrix ds, SharedMatrix df,
                                    SharedMatrix dfoo) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("LMP2 tangent requires an RHF DLPNO gradient model.");
    return local->lmp2_tangent(dy,dk,ds,df,dfoo);
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> triples_ccsd_source(SharedWavefunction wfn) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("Expected an RHF DLPNO triples wavefunction.");
    return local->triples_ccsd_source();
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> gradient_ccsd_lambda(SharedWavefunction wfn) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("Expected an RHF DLPNO gradient-model wavefunction.");
    return local->gradient_ccsd_lambda();
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> ccsd_residual_at(SharedWavefunction wfn,
    const std::vector<SharedMatrix>& singles, const DLPNOCCSD::PairBlocks& doubles) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("Expected an RHF DLPNO gradient model wavefunction.");
    return local->ccsd_residual_at(singles,doubles);
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> ccsd_amplitude_action(
    SharedWavefunction wfn, const std::vector<SharedMatrix>& singles, const DLPNOCCSD::PairBlocks& doubles, bool transpose) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("CCSD amplitude response requires an RHF DLPNO gradient model.");
    return local->ccsd_amplitude_action(singles, doubles, transpose);
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> ccsd_fock_source(
    SharedWavefunction wfn, SharedMatrix oo, const DLPNOCCSD::PairBlocks& ov, const DLPNOCCSD::PairBlocks& vv) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("CCSD residual response requires an RHF DLPNO gradient model.");
    return local->ccsd_fock_source(oo,ov,vv);
}

std::map<std::tuple<int,int,int>,DLPNOCCSD_T::TriplesRow> triples_response_state(SharedWavefunction wfn) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("Expected an RHF DLPNO triples response wavefunction.");
    return local->triples_response_state();
}

DLPNOCCSD::DFPairState ccsd_df_pair_state(SharedWavefunction wfn, int i, int j) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("CCSD response requires an RHF DLPNO gradient model.");
    return local->ccsd_df_pair_state(i,j);
}

std::vector<SharedMatrix> ccsd_mixed_state(SharedWavefunction wfn, int i, int j, int k) {
    auto local = std::dynamic_pointer_cast<DLPNOCCSD_T>(wfn);
    if (!local) throw PSIEXCEPTION("Mixed CCSD response requires an RHF DLPNO gradient model.");
    return local->ccsd_mixed_state(i,j,k);
}

SharedWavefunction dlpno(SharedWavefunction ref_wfn, Options& options) {

    const auto reference = options.get_str("REFERENCE");
    const auto algorithm = options.get_str("DLPNO_ALGORITHM");
    const auto orbitals = options.get_str("DLPNO_REFERENCE_ORBITALS");
    const int orbital_rank = orbitals == "BCCDTQ" ? 4 : orbitals == "BCCDT" ? 3 :
                             (orbitals == "BCCD" || options.get_bool("DLPNO_BRUECKNER_ORBS")) ? 2 : 0;
    const int energy_rank = algorithm == "CCSDTQ" ? 4 :
                            (algorithm == "CCSDT" || algorithm == "CCSDT(Q)") ? 3 :
                            algorithm == "MP2" ? 0 : 2;
    if (orbital_rank > energy_rank)
        throw PSIEXCEPTION("DLPNO_REFERENCE_ORBITALS cannot exceed the final iterative CC rank.");
    const bool lambda = options.get_bool("DLPNO_DO_LAMBDA") || options.get_bool("DLPNO_DO_ONEPDM");
    if (options.get_bool("DLPNO_GRADIENT_MODEL")) {
        if (reference != "RHF" || ref_wfn->nalpha() != ref_wfn->nbeta() || !ref_wfn->same_a_b_orbs() ||
            algorithm != "CCSD(T)" || orbital_rank || lambda || options.get_bool("T0_APPROXIMATION"))
            throw PSIEXCEPTION("DLPNO_GRADIENT_MODEL requires closed-shell RHF DLPNO-CCSD(T) with HF orbitals and ordinary iterative triples.");
        const auto scf_type = options.get_str("SCF_TYPE");
        if (scf_type != "DF" && scf_type != "DISK_DF" && scf_type != "MEM_DF")
            throw PSIEXCEPTION("DLPNO_GRADIENT_MODEL requires a density-fitted RHF reference.");
        const auto localizer = options.get_str("DLPNO_LOCAL_ORBITALS");
        if (localizer != "BOYS" && localizer != "PIPEK_MEZEY" && localizer != "IBO")
            throw PSIEXCEPTION("DLPNO_GRADIENT_MODEL supports Boys, Pipek-Mezey, and IBO localization.");
    }
    if (lambda && (reference != "RHF" || energy_rank != 2))
        throw PSIEXCEPTION("DLPNO Lambda/OPDM is available only for RHF CCSD and CCSD(T).");
    if (reference == "ROHF" && (orbital_rank > 2 || energy_rank != 2))
        throw PSIEXCEPTION("ROHF DLPNO supports CCSD/CCSD(T) with HF or BCCD orbitals; BCCDT/BCCDTQ require RHF.");
    if (reference == "UHF" && orbital_rank)
        throw PSIEXCEPTION("DLPNO Brueckner optimization requires RHF or ROHF; the UHF-QRO route supports HF orbitals only.");

    std::shared_ptr<Wavefunction> dlpno;
    if (options.get_str("REFERENCE") == "RHF") {
        if (options.get_str("DLPNO_ALGORITHM") == "MP2") {
            dlpno = std::make_shared<DLPNOMP2>(ref_wfn, options);
        } else if (options.get_str("DLPNO_ALGORITHM") == "CCSD") {
            dlpno = std::make_shared<DLPNOCCSD>(ref_wfn, options);
        } else if (options.get_str("DLPNO_ALGORITHM") == "CCSD(T)") {
            dlpno = std::make_shared<DLPNOCCSD_T>(ref_wfn, options);
#ifdef USING_Einsums
        } else if (options.get_str("DLPNO_ALGORITHM") == "CCSDT") {
            dlpno = std::make_shared<DLPNOCCSDT>(ref_wfn, options);
        } else if (options.get_str("DLPNO_ALGORITHM") == "CCSDT(Q)") {
            dlpno = std::make_shared<DLPNOCCSDT_Q>(ref_wfn, options);
        } else if (options.get_str("DLPNO_ALGORITHM") == "CCSDTQ") {
            dlpno = std::make_shared<DLPNOCCSDTQ>(ref_wfn, options);
#else
        } else if (options.get_str("DLPNO_ALGORITHM") == "CCSDT" ||
                   options.get_str("DLPNO_ALGORITHM") == "CCSDT(Q)" ||
                   options.get_str("DLPNO_ALGORITHM") == "CCSDTQ") {
            throw PSIEXCEPTION("DLPNO-CCSDT, DLPNO-CCSDT(Q), and DLPNO-CCSDTQ require Psi4 built with Einsums support (ENABLE_Einsums=ON).");
#endif
        } else {
            throw PSIEXCEPTION("Requested DLPNO method is not available: " +
                               options.get_str("DLPNO_ALGORITHM"));
        }
    } else if (options.get_str("REFERENCE") == "ROHF") {
        if (options.get_str("DLPNO_ALGORITHM") == "CCSD") {
            dlpno = std::make_shared<RO_DLPNOCCSD>(ref_wfn, options);
        } else if (options.get_str("DLPNO_ALGORITHM") == "CCSD(T)") {
            dlpno = std::make_shared<RO_DLPNOCCSD_T>(ref_wfn, options);
        } else {
            throw PSIEXCEPTION("Requested DLPNO method is not yet available for ROHF reference!");
        }
    } else if (options.get_str("REFERENCE") == "UHF") {
        if (options.get_str("DLPNO_ALGORITHM") == "CCSD") {
            dlpno = std::make_shared<RO_DLPNOCCSD>(ref_wfn, options);
        } else {
            throw PSIEXCEPTION(
                "UHF references are supported only for DLPNO-CCSD through the UHF -> QRO transformation.");
        }
    } else {
        throw PSIEXCEPTION("Requested DLPNO reference is not supported.\n");
    }

    return dlpno;
}
}  // namespace dlpno
}  // namespace psi
