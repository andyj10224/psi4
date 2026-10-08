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
