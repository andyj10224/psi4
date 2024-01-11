/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2022 The Psi4 Developers.
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
#include "sparse.h"

#include "psi4/lib3index/3index.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/local.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/orthog.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"

#include <ctime>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {
namespace dlpno {

DLPNOCCSD::DLPNOCCSD(SharedWavefunction ref_wfn, Options& options) : DLPNOBase(ref_wfn, options) {}
DLPNOCCSD::~DLPNOCCSD() {}

inline SharedMatrix DLPNOCCSD::S_PNO(const int ij, const int mn) {
    int i, j, m, n;
    std::tie(i, j) = ij_to_i_j_[ij];
    std::tie(m, n) = ij_to_i_j_[mn];

    int ji = ij_to_ji_[ij];
    int nm = ij_to_ji_[mn];

    if (i == m) { // S(ij, mn) -> S(ij, in) -> S(ji, ni)
        return S_pno_ij_kj_[ji][n];
    } else if (i == n) { // S(ij, mn) -> S(ij, mi) -> S(ji, mi)
        return S_pno_ij_kj_[ji][m];
    } else if (j == m) { // S(ij, mn) -> S(ij, jn) -> S(ij, nj)
        return S_pno_ij_kj_[ij][n];
    } else if (j == n) { // S(ij, mn) -> S(ij, mj)
        return S_pno_ij_kj_[ij][m];
    } else {
        int i, j, m, n;
        std::tie(i, j) = ij_to_i_j_[ij];
        std::tie(m, n) = ij_to_i_j_[mn];

        const int m_ij = lmopair_to_lmos_dense_[ij][m], n_ij = lmopair_to_lmos_dense_[ij][n];
        if (m_ij == -1 || n_ij == -1) {
            // outfile->Printf("Invalid PNO Pairs (%d, %d) and (%d, %d)\n", i, j, m, n);
            // throw PSIEXCEPTION("Invalid PNO pairs!");
            auto S_ij_mn = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[mn]);
            return linalg::triplet(X_pno_[ij], S_ij_mn, X_pno_[mn], true, false, false);
        }
        
        const int nlmo_ij = lmopair_to_lmos_[ij].size();

        int mn_ij; 
        if (m_ij > n_ij) {
            mn_ij = n_ij * nlmo_ij + m_ij;
        } else {
            mn_ij = m_ij * nlmo_ij + n_ij;
        }

        if (i > j) {
            const int ji = ij_to_ji_[ij];
            return S_pno_ij_mn_[ji][mn_ij];
        } else {
            return S_pno_ij_mn_[ij][mn_ij];
        }
    
    }
}

void DLPNOCCSD::compute_pno_overlaps() {

    const int naocc = i_j_to_ij_.size();
    const int n_lmo_pairs = ij_to_i_j_.size();
    
    S_pno_ij_kj_.resize(n_lmo_pairs);
    S_pno_ij_mn_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];

        S_pno_ij_kj_[ij].resize(naocc);

        const int npno_ij = n_pno_[ij];
        const int nlmo_ij = lmopair_to_lmos_[ij].size();

        for (int k = 0; k < naocc; ++k) {
            int kj = i_j_to_ij_[k][j];

            if (kj == -1) continue;

            S_pno_ij_kj_[ij][k] = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[kj]);
            S_pno_ij_kj_[ij][k] = linalg::triplet(X_pno_[ij], S_pno_ij_kj_[ij][k], X_pno_[kj], true, false, false);
        }

        if (i > j) continue;

        S_pno_ij_mn_[ij].resize(nlmo_ij * nlmo_ij);

        for (int mn_ij = 0; mn_ij < nlmo_ij * nlmo_ij; ++mn_ij) {
            const int m_ij = mn_ij / nlmo_ij, n_ij = mn_ij % nlmo_ij;
            const int m = lmopair_to_lmos_[ij][m_ij], n = lmopair_to_lmos_[ij][n_ij];
            const int mn = i_j_to_ij_[m][n];

            if (mn == -1 || m_ij > n_ij) continue;

            S_pno_ij_mn_[ij][mn_ij] = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[mn]);
            S_pno_ij_mn_[ij][mn_ij] = linalg::triplet(X_pno_[ij], S_pno_ij_mn_[ij][mn_ij], X_pno_[mn], true, false, false);
        }
    }
}

void DLPNOCCSD::estimate_memory() {
    outfile->Printf("  ==> DLPNO-CCSD Memory Estimate <== \n\n");

    int naocc = i_j_to_ij_.size();
    int n_lmo_pairs = ij_to_i_j_.size();

    size_t pno_overlap_memory = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : pno_overlap_memory)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];

        const int npno_ij = n_pno_[ij];
        const int nlmo_ij = lmopair_to_lmos_[ij].size();
        if (i > j) continue;

        for (int mn_ij = 0; mn_ij < nlmo_ij * nlmo_ij; mn_ij++) {
            const int m_ij = mn_ij / nlmo_ij, n_ij = mn_ij % nlmo_ij;
            const int m = lmopair_to_lmos_[ij][m_ij], n = lmopair_to_lmos_[ij][n_ij];
            const int mn = i_j_to_ij_[m][n];
            if (mn == -1 || m_ij > n_ij) continue;

            pno_overlap_memory += n_pno_[ij] * n_pno_[mn];
        }
    }

    size_t oooo = 0;
    size_t ooov = 0;
    size_t oovv = 0;
    size_t ovvv = 0;
    size_t qov = 0;
    size_t qvv = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : oooo, ooov, oovv, ovvv, qov, qvv)
    for (int ij = 0; ij < n_lmo_pairs; ij++) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];

        const int naux_ij = lmopair_to_ribfs_[ij].size();
        const int nlmo_ij = lmopair_to_lmos_[ij].size();
        const int npno_ij = n_pno_[ij];

        oooo += nlmo_ij * nlmo_ij;
        ooov += 3 * nlmo_ij * npno_ij;
        oovv += 4 * npno_ij * npno_ij;
        ovvv += 3 * npno_ij * npno_ij * npno_ij;
        if (i >= j && i_j_to_ij_strong_[i][j] != -1) {
            qov += naux_ij * nlmo_ij * npno_ij;
            qvv += naux_ij * npno_ij * npno_ij;
        }
    }

    if (write_qab_pao_) qab_memory_ = 0;

    if (qvv * sizeof(double) > 0.5 * (memory_ - qab_memory_ * sizeof(double))) {
        write_qab_pno_ = true;
        outfile->Printf("    Writing (aux | pno * pno) integrals to disk...\n\n");
    } else {
        write_qab_pno_ = false;
        outfile->Printf("    Keeping (aux | pno * pno) integrals in core...\n\n");
    }
    
    if (write_qab_pno_) qvv  = 0;

    const size_t total_df_memory = qij_memory_ + qia_memory_ + qab_memory_;
    const size_t total_pno_int_memory = oooo + ooov + oovv + ovvv + qov + qvv;
    const size_t total_memory = total_df_memory + pno_overlap_memory + total_pno_int_memory;

    // 2^30 bytes per GiB
    outfile->Printf("    (q | i j) integrals    : %.3f [GiB]\n", qij_memory_ * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (q | i a) integrals    : %.3f [GiB]\n", qia_memory_ * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (q | a b) integrals    : %.3f [GiB]\n", qab_memory_ * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (m i | n j) integrals  : %.3f [GiB]\n", oooo * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    1-virtual PNO integrals: %.3f [GiB]\n", ooov * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    2-virtual PNO integrals: %.3f [GiB]\n", oovv * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    3-virtual PNO integrals: %.3f [GiB]\n", ovvv * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ij}|m_{ij} a_{ij}) : %.3f [GiB]\n", qov * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ij}|a_{ij} b_{ij}) : %.3f [GiB]\n", qvv * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    PNO/PNO overlaps       : %.3f [GiB]\n\n", pno_overlap_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    Total Memory Given     : %.3f [GiB]\n", memory_ * pow(2.0, -30));
    outfile->Printf("    Total Memory Required  : %.3f [GiB]\n\n", total_memory * pow(2.0, -30) * sizeof(double));

}

template<bool crude> std::vector<double> DLPNOCCSD::compute_pair_energies() {
    /*
        If crude, runs semicanonical (non-iterative) MP2
        If non-crude, computes PNOs (through PNO transform), runs full iterative LMP2
    */

    int nbf = basisset_->nbf();
    int naocc = i_j_to_ij_.size();
    int n_lmo_pairs = ij_to_i_j_.size();
    std::vector<double> e_ijs(n_lmo_pairs);

    outfile->Printf("\n  ==> Computing SC-LMP2 Pair Energies <==\n\n");

    outfile->Printf("    Starting semicanonical (non-iterative) LMP2 step...\n\n");

    if constexpr (!crude) {
        outfile->Printf("\n  ==> Forming Pair Natural Orbitals (for LMP2) <==\n");

        K_iajb_.resize(n_lmo_pairs);
        T_iajb_.resize(n_lmo_pairs);
        Tt_iajb_.resize(n_lmo_pairs);
        X_pno_.resize(n_lmo_pairs);
        e_pno_.resize(n_lmo_pairs);

        n_pno_.resize(n_lmo_pairs);
        occ_pno_.resize(n_lmo_pairs);
        trace_pno_.resize(n_lmo_pairs);
        de_pno_.resize(n_lmo_pairs);
        // de_pno_os_.resize(n_lmo_pairs);
        // de_pno_ss_.resize(n_lmo_pairs);
    }

    double e_sc_lmp2 = 0.0;

    // Step 1: compute SC-LMP2 pair energies
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : e_sc_lmp2)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        if (i > j) continue;

        //                                                   //
        // ==> Assemble (ia|jb) for pair ij in PAO basis <== //
        //                                                   //

        // number of PAOs in the pair domain (before removing linear dependencies)
        int npao_ij = lmopair_to_paos_[ij].size();  // X_pao_ij->rowspi(0);

        // number of auxiliary basis in the domain
        int naux_ij = lmopair_to_ribfs_[ij].size();

        auto i_qa = std::make_shared<Matrix>("Three-index Integrals", naux_ij, npao_ij);
        auto j_qa = std::make_shared<Matrix>("Three-index Integrals", naux_ij, npao_ij);

        for (int q_ij = 0; q_ij < naux_ij; q_ij++) {
            int q = lmopair_to_ribfs_[ij][q_ij];
            int centerq = ribasis_->function_to_center(q);
            for (int a_ij = 0; a_ij < npao_ij; a_ij++) {
                int a = lmopair_to_paos_[ij][a_ij];
                i_qa->set(q_ij, a_ij, qia_[q]->get(riatom_to_lmos_ext_dense_[centerq][i], riatom_to_paos_ext_dense_[centerq][a]));
                j_qa->set(q_ij, a_ij, qia_[q]->get(riatom_to_lmos_ext_dense_[centerq][j], riatom_to_paos_ext_dense_[centerq][a]));
            }
        }

        auto A_solve = submatrix_rows_and_cols(*full_metric_, lmopair_to_ribfs_[ij], lmopair_to_ribfs_[ij]);
        C_DGESV_wrapper(A_solve, i_qa);

        auto K_pao_ij = linalg::doublet(i_qa, j_qa, true, false);

        //                                      //
        // ==> Canonicalize PAOs of pair ij <== //
        //                                      //

        auto S_pao_ij = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ij]);
        auto F_pao_ij = submatrix_rows_and_cols(*F_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ij]);

        SharedMatrix X_pao_ij;  // canonical transformation of this domain's PAOs to
        SharedVector e_pao_ij;  // energies of the canonical PAOs
        std::tie(X_pao_ij, e_pao_ij) = orthocanonicalizer(S_pao_ij, F_pao_ij);

        // S_pao_ij = linalg::triplet(X_pao_ij, S_pao_ij, X_pao_ij, true, false, false);
        F_pao_ij = linalg::triplet(X_pao_ij, F_pao_ij, X_pao_ij, true, false, false);
        K_pao_ij = linalg::triplet(X_pao_ij, K_pao_ij, X_pao_ij, true, false, false);

        // number of PAOs in the domain after removing linear dependencies
        int npao_can_ij = X_pao_ij->colspi(0);
        auto T_pao_ij = K_pao_ij->clone();
        for (int a = 0; a < npao_can_ij; ++a) {
            for (int b = 0; b < npao_can_ij; ++b) {
                T_pao_ij->set(a, b, T_pao_ij->get(a, b) /
                                        (-e_pao_ij->get(b) + -e_pao_ij->get(a) + F_lmo_->get(i, i) + F_lmo_->get(j, j)));
            }
        }

        // Compute SC-LMP2 pair energy using PAOs

        size_t nvir_ij = K_pao_ij->rowspi(0);

        auto Tt_pao_ij = T_pao_ij->clone();
        Tt_pao_ij->scale(2.0);
        Tt_pao_ij->subtract(T_pao_ij->transpose());

        // mp2 energy of this LMO pair before transformation to PNOs
        double e_ij_initial = K_pao_ij->vector_dot(Tt_pao_ij);
        // double e_ij_os_initial = K_pao_ij->vector_dot(T_pao_ij);
        // double e_ij_ss_initial = e_ij_initial - e_ij_os_inital;

        double prefactor = (i == j) ? 1.0 : 2.0;
        e_sc_lmp2 += prefactor * e_ij_initial;

        e_ijs[ij] = e_ij_initial;
        if (i < j) {
            e_ijs[ji] = e_ij_initial;
        }

        if constexpr (!crude) {
            //                                           //
            // ==> Canonical PAOs  to Canonical PNOs <== //
            //                                           //

            // PNOs defined in (DOI: 10.1063/1.3086717), EQ 17 through EQ 24

            // Construct pair density from amplitudes
            auto D_ij = linalg::doublet(Tt_pao_ij, T_pao_ij, false, true);
            D_ij->add(linalg::doublet(Tt_pao_ij, T_pao_ij, true, false));
            if (i == j) D_ij->scale(0.5);

            // Diagonalization of pair density gives PNOs (in basis of the LMO's virtual domain) and PNO occ numbers
            auto X_pno_ij = std::make_shared<Matrix>("eigenvectors", nvir_ij, nvir_ij);
            Vector pno_occ("eigenvalues", nvir_ij);
            D_ij->diagonalize(*X_pno_ij, pno_occ, descending);

            double t_cut_scale = (i == j) ? 0.0 : 1.0;

            double occ_total = 0.0;
            for (size_t a = 0; a < nvir_ij; ++a) {
                occ_total += pno_occ.get(a);
            }

            double e_pno = 0.0;
            double occ_pno = 0.0;
            int nvir_ij_final = 0;
            std::vector<int> a_curr;
            auto K_pno_init = linalg::triplet(X_pno_ij, K_pao_ij, X_pno_ij, true, false, false);
            auto Tt_pno_init = linalg::triplet(X_pno_ij, Tt_pao_ij, X_pno_ij, true, false, false);
            for (size_t a = 0; a < nvir_ij; ++a) {
                a_curr.push_back(a);
                if (fabs(pno_occ.get(a)) >= t_cut_scale * T_CUT_PNO_MP2_ || occ_pno / occ_total < T_CUT_TRACE_MP2_ ||
                        std::fabs(e_pno) < T_CUT_ENERGY_MP2_ * std::fabs(e_ij_initial)) {
                    e_pno = submatrix_rows_and_cols(*K_pno_init, a_curr, a_curr)->vector_dot(submatrix_rows_and_cols(*Tt_pno_init, a_curr, a_curr));
                    occ_pno += pno_occ.get(a);
                    nvir_ij_final++;
                }
            }

            // Make sure there is at least one PNO per pair :)
            nvir_ij_final = std::max(1, nvir_ij_final);

            Dimension zero(1);
            Dimension dim_final(1);
            dim_final.fill(nvir_ij_final);

            // This transformation gives orbitals that are orthonormal but not canonical
            X_pno_ij = X_pno_ij->get_block({zero, X_pno_ij->rowspi()}, {zero, dim_final});
            pno_occ = pno_occ.get_block({zero, dim_final});

            SharedMatrix pno_canon;
            SharedVector e_pno_ij;
            std::tie(pno_canon, e_pno_ij) = canonicalizer(X_pno_ij, F_pao_ij);

            // This transformation gives orbitals that are orthonormal and canonical
            X_pno_ij = linalg::doublet(X_pno_ij, pno_canon, false, false);

            auto K_pno_ij = linalg::triplet(X_pno_ij, K_pao_ij, X_pno_ij, true, false, false);
            auto T_pno_ij = linalg::triplet(X_pno_ij, T_pao_ij, X_pno_ij, true, false, false);
            auto Tt_pno_ij = linalg::triplet(X_pno_ij, Tt_pao_ij, X_pno_ij, true, false, false);

            // mp2 energy of this LMO pair after transformation to PNOs and truncation
            double e_ij_trunc = K_pno_ij->vector_dot(Tt_pno_ij);
            // double e_ij_os_trunc = K_pno_ij->vector_dot(T_pno_ij);
            // double e_ij_ss_trunc = e_ij_trunc - e_ij_os_trunc;

            // truncation error
            double de_pno_ij = e_ij_initial - e_ij_trunc;
            // double de_pno_ij_os = e_ij_os_initial - e_ij_os_trunc;
            // double de_pno_ij_ss = e_ij_ss_initial - e_ij_ss_trunc;

            X_pno_ij = linalg::doublet(X_pao_ij, X_pno_ij, false, false);

            K_iajb_[ij] = K_pno_ij;
            T_iajb_[ij] = T_pno_ij;
            Tt_iajb_[ij] = Tt_pno_ij;
            X_pno_[ij] = X_pno_ij;
            e_pno_[ij] = e_pno_ij;
            n_pno_[ij] = X_pno_ij->colspi(0);
            occ_pno_[ij] = pno_occ.get(n_pno_[ij] - 1);
            trace_pno_[ij] = occ_pno / occ_total;
            de_pno_[ij] = de_pno_ij;
            // de_pno_os_[ij] = de_pno_ij_os;
            // de_pno_ss_[ij] = de_pno_ij_ss;

            // account for symmetry
            if (i < j) {
                K_iajb_[ji] = K_iajb_[ij]->transpose();
                T_iajb_[ji] = T_iajb_[ij]->transpose();
                Tt_iajb_[ji] = Tt_iajb_[ij]->transpose();
                X_pno_[ji] = X_pno_[ij];
                e_pno_[ji] = e_pno_[ij];
                n_pno_[ji] = n_pno_[ij];
                occ_pno_[ji] = occ_pno_[ij];
                trace_pno_[ji] = trace_pno_[ij];
                de_pno_[ji] = de_pno_ij;
                // de_pno_os_[ji] = de_pno_ij_os;
                // de_pno_ss_[ji] = de_pno_ij_ss;
            }
        }
    } // end for (ij pairs)

    outfile->Printf("    PAO-SC-LMP2 Energy: %16.12f\n\n", e_sc_lmp2);

    if constexpr (!crude) {
        // Print out PNO domain information
        int pno_count_total = 0, pno_count_min = nbf, pno_count_max = 0;
        double occ_number_total = 0.0, occ_number_min = 2.0, occ_number_max = 0.0;
        double trace_total = 0.0, trace_min = 1.0, trace_max = 0.0;
        de_pno_total_ = 0.0, de_pno_total_os_ = 0.0, de_pno_total_ss_ = 0.0;
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            pno_count_total += n_pno_[ij];
            pno_count_min = std::min(pno_count_min, n_pno_[ij]);
            pno_count_max = std::max(pno_count_max, n_pno_[ij]);
            occ_number_total += occ_pno_[ij];
            occ_number_min = std::min(occ_number_min, occ_pno_[ij]);
            occ_number_max = std::max(occ_number_max, occ_pno_[ij]);
            trace_total += trace_pno_[ij];
            trace_min = std::min(trace_min, trace_pno_[ij]);
            trace_max = std::max(trace_max, trace_pno_[ij]);
            de_pno_total_ += de_pno_[ij];
        }

        outfile->Printf("  \n");
        outfile->Printf("    Natural Orbitals per Local MO pair:\n");
        outfile->Printf("      Avg: %3d NOs \n", pno_count_total / n_lmo_pairs);
        outfile->Printf("      Min: %3d NOs \n", pno_count_min);
        outfile->Printf("      Max: %3d NOs \n", pno_count_max);
        outfile->Printf("      Avg Occ Number Tol: %.3e \n", occ_number_total / n_lmo_pairs);
        outfile->Printf("      Min Occ Number Tol: %.3e \n", occ_number_min);
        outfile->Printf("      Max Occ Number Tol: %.3e \n", occ_number_max);
        outfile->Printf("      Avg Trace Sum: %.6f \n", trace_total / n_lmo_pairs);
        outfile->Printf("      Min Trace Sum: %.6f \n", trace_min);
        outfile->Printf("      Max Trace Sum: %.6f \n", trace_max);
        outfile->Printf("  \n");
        outfile->Printf("    PNO truncation energy = %.12f\n", de_pno_total_);
    }

    return e_ijs;
}

void DLPNOCCSD::pno_lmp2_iterations() {

    int nbf = basisset_->nbf();
    int naocc = i_j_to_ij_.size();
    int n_lmo_pairs = ij_to_i_j_.size();

    outfile->Printf("  ==> PNO-LMP2 Memory Estimate <== \n\n");

    double F_CUT_PAIR = options_.get_double("F_CUT");

    size_t pno_overlap_memory = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : pno_overlap_memory)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        if (i > j) continue;

        for (int k = 0; k < naocc; ++k) {
            int kj = i_j_to_ij_[k][j];
            int ik = i_j_to_ij_[i][k];

            if (kj != -1 && i != k && fabs(F_lmo_->get(i, k)) > F_CUT_PAIR) {
                pno_overlap_memory += n_pno_[ij] * n_pno_[kj];
            }
            if (ik != -1 && j != k && fabs(F_lmo_->get(k, j)) > F_CUT_PAIR) {
                pno_overlap_memory += n_pno_[ij] * n_pno_[ik];
            }
        }
    }
        
    // 2^30 bytes per GiB
    outfile->Printf("    PNO/PNO overlaps       : %.3f [GiB]\n", pno_overlap_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    Total Memory Given     : %.3f [GiB]\n\n", memory_ * pow(2.0, -30));

    std::vector<std::vector<SharedMatrix>> S_pno_ij_ik(n_lmo_pairs);
    std::vector<std::vector<SharedMatrix>> S_pno_ij_kj(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        if (i > j) continue;

        S_pno_ij_ik[ij].resize(naocc);
        S_pno_ij_kj[ij].resize(naocc);

        for (int k = 0; k < naocc; ++k) {
            int kj = i_j_to_ij_[k][j];
            int ik = i_j_to_ij_[i][k];

            if (kj != -1 && i != k && fabs(F_lmo_->get(i, k)) > F_CUT_PAIR) {
                S_pno_ij_kj[ij][k] = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[kj]);
                S_pno_ij_kj[ij][k] = linalg::triplet(X_pno_[ij], S_pno_ij_kj[ij][k], X_pno_[kj], true, false, false);
            }
            if (ik != -1 && j != k && fabs(F_lmo_->get(k, j)) > F_CUT_PAIR) {
                S_pno_ij_ik[ij][k] = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ik]);
                S_pno_ij_ik[ij][k] = linalg::triplet(X_pno_[ij], S_pno_ij_ik[ij][k], X_pno_[ik], true, false, false);
            }
        } // end k
    } // end ij

    // => Computing Truncated LMP2 energies (basically running DLPNO-MP2 here)
    outfile->Printf("\n  ==> Iterative Local MP2 with Pair Natural Orbitals (PNOs) <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    outfile->Printf("                         Corr. Energy    Delta E     Max R     Time (s)\n");

    std::vector<SharedMatrix> R_iajb(n_lmo_pairs);

    int iteration = 0, max_iteration = options_.get_int("DLPNO_MAXITER");
    double e_curr = 0.0, e_prev = 0.0, r_curr = 0.0;
    bool e_converged = false, r_converged = false;
    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), "LMP2 DIIS", DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);

    while (!(e_converged && r_converged)) {
        // RMS of residual per LMO pair, for assessing convergence
        std::vector<double> R_iajb_rms(n_lmo_pairs, 0.0);

        std::time_t time_start = std::time(nullptr);

        // Calculate residuals from current amplitudes
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];

            if (i > j) continue;

            R_iajb[ij] = std::make_shared<Matrix>("Residual", n_pno_[ij], n_pno_[ij]);

            if (n_pno_[ij] == 0) continue;

            for (int a = 0; a < n_pno_[ij]; ++a) {
                for (int b = 0; b < n_pno_[ij]; ++b) {
                    R_iajb[ij]->set(a, b,
                                    K_iajb_[ij]->get(a, b) +
                                        (e_pno_[ij]->get(a) + e_pno_[ij]->get(b) - F_lmo_->get(i, i) - F_lmo_->get(j, j)) *
                                            T_iajb_[ij]->get(a, b));
                }
            }

            for (int k = 0; k < naocc; ++k) {
                int kj = i_j_to_ij_[k][j];
                int ik = i_j_to_ij_[i][k];

                if (kj != -1 && i != k && fabs(F_lmo_->get(i, k)) > options_.get_double("F_CUT") && n_pno_[kj] > 0) {
                    auto S_ij_kj = S_pno_ij_kj[ij][k];
                    auto temp = linalg::triplet(S_ij_kj, T_iajb_[kj], S_ij_kj, false, false, true);
                    temp->scale(-1.0 * F_lmo_->get(i, k));
                    R_iajb[ij]->add(temp);
                }
                if (ik != -1 && j != k && fabs(F_lmo_->get(k, j)) > options_.get_double("F_CUT") && n_pno_[ik] > 0) {
                    auto S_ij_ik = S_pno_ij_ik[ij][k];
                    auto temp = linalg::triplet(S_ij_ik, T_iajb_[ik], S_ij_ik, false, false, true);
                    temp->scale(-1.0 * F_lmo_->get(k, j));
                    R_iajb[ij]->add(temp);
                }
            }

            R_iajb_rms[ij] = R_iajb[ij]->rms();

            if (i < j) {
                int ji = ij_to_ji_[ij];
                R_iajb[ji] = R_iajb[ij]->transpose();
                R_iajb_rms[ji] = R_iajb_rms[ij];
            }
        }

// use residuals to get next amplitudes
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            for (int a = 0; a < n_pno_[ij]; ++a) {
                for (int b = 0; b < n_pno_[ij]; ++b) {
                    T_iajb_[ij]->add(a, b, -R_iajb[ij]->get(a, b) / ((e_pno_[ij]->get(a) + e_pno_[ij]->get(b)) -
                                                                    (F_lmo_->get(i, i) + F_lmo_->get(j, j))));
                }
            }
        }

        // DIIS extrapolation
        auto T_iajb_flat = flatten_mats(T_iajb_);
        auto R_iajb_flat = flatten_mats(R_iajb);

        if (iteration == 0) {
            diis.set_error_vector_size(R_iajb_flat.get());
            diis.set_vector_size(T_iajb_flat.get());
        }

        diis.add_entry(R_iajb_flat.get(), T_iajb_flat.get());
        diis.extrapolate(T_iajb_flat.get());

        copy_flat_mats(T_iajb_flat, T_iajb_);

#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            Tt_iajb_[ij]->copy(T_iajb_[ij]);
            Tt_iajb_[ij]->scale(2.0);
            Tt_iajb_[ij]->subtract(T_iajb_[ij]->transpose());
        }

        // evaluate convergence using current amplitudes and residuals
        e_prev = e_curr;
        e_curr = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : e_curr)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];

            e_curr += K_iajb_[ij]->vector_dot(Tt_iajb_[ij]);
        }
        r_curr = *max_element(R_iajb_rms.begin(), R_iajb_rms.end());

        r_converged = (fabs(r_curr) < options_.get_double("R_CONVERGENCE"));
        e_converged = (fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE"));

        std::time_t time_stop = std::time(nullptr);

        outfile->Printf("  @PNO-LMP2 iter %3d: %16.12f %10.3e %10.3e %8d\n", iteration, e_curr, e_curr - e_prev, r_curr, (int)time_stop - (int)time_start);

        iteration++;

        if (iteration > max_iteration) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }
    }

    // Recompute Pair Natural Orbitals for LCCSD iterations

    outfile->Printf("\n  ==> Forming Pair Natural Orbitals (for LCCSD) <==\n");

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        if (i > j) continue;

        auto F_pao_ij = submatrix_rows_and_cols(*F_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ij]);
        auto F_pno_ij_init = linalg::triplet(X_pno_[ij], F_pao_ij, X_pno_[ij], true, false, false);

        // Construct pair density from amplitudes
        auto D_ij = linalg::doublet(Tt_iajb_[ij], T_iajb_[ij], false, true);
        D_ij->add(linalg::doublet(Tt_iajb_[ij], T_iajb_[ij], true, false));
        if (i == j) D_ij->scale(0.5);

        int nvir_ij = F_pno_ij_init->rowspi(0);

        // Diagonalization of pair density gives PNOs (in basis of the LMO's virtual domain) and PNO occ numbers
        auto X_pno_ij = std::make_shared<Matrix>("eigenvectors", nvir_ij, nvir_ij);
        Vector pno_occ("eigenvalues", nvir_ij);
        D_ij->diagonalize(*X_pno_ij, pno_occ, descending);

        double t_cut_scale = (i == j) ? T_CUT_PNO_DIAG_SCALE_ : 1.0;

        double occ_total = 0.0;
        for (size_t a = 0; a < nvir_ij; ++a) {
            occ_total += pno_occ.get(a);
        }

        double e_pno = 0.0;
        double e_ij_total = K_iajb_[ij]->vector_dot(Tt_iajb_[ij]);
        double occ_pno = 0.0;
        int nvir_ij_final = 0;
        std::vector<int> a_curr;
        auto K_pno_init = linalg::triplet(X_pno_ij, K_iajb_[ij], X_pno_ij, true, false, false);
        auto Tt_pno_init = linalg::triplet(X_pno_ij, Tt_iajb_[ij], X_pno_ij, true, false, false);
        for (size_t a = 0; a < nvir_ij; ++a) {
            a_curr.push_back(a);
            if (fabs(pno_occ.get(a)) >= t_cut_scale * T_CUT_PNO_ || occ_pno / occ_total < T_CUT_TRACE_ ||
                    std::fabs(e_pno) < T_CUT_ENERGY_ * std::fabs(e_ij_total)) {
                e_pno = submatrix_rows_and_cols(*K_pno_init, a_curr, a_curr)->vector_dot(submatrix_rows_and_cols(*Tt_pno_init, a_curr, a_curr));
                occ_pno += pno_occ.get(a);
                nvir_ij_final++;
            }
        }
        // Make sure there is at least one PNO per pair :)
        nvir_ij_final = std::max(1, nvir_ij_final);

        Dimension zero(1);
        Dimension dim_final(1);
        dim_final.fill(nvir_ij_final);

        // This transformation gives orbitals that are orthonormal but not canonical
        X_pno_ij = X_pno_ij->get_block({zero, X_pno_ij->rowspi()}, {zero, dim_final});
        pno_occ = pno_occ.get_block({zero, dim_final});

        SharedMatrix pno_canon;
        SharedVector e_pno_ij;
        std::tie(pno_canon, e_pno_ij) = canonicalizer(X_pno_ij, F_pno_ij_init);

        // This transformation gives orbitals that are orthonormal and canonical
        X_pno_ij = linalg::doublet(X_pno_ij, pno_canon, false, false);

        auto K_pno_ij = linalg::triplet(X_pno_ij, K_iajb_[ij], X_pno_ij, true, false, false);
        auto T_pno_ij = linalg::triplet(X_pno_ij, T_iajb_[ij], X_pno_ij, true, false, false);
        auto Tt_pno_ij = linalg::triplet(X_pno_ij, Tt_iajb_[ij], X_pno_ij, true, false, false);

        // (additional) truncation error
        double de_pno_ij = K_iajb_[ij]->vector_dot(Tt_iajb_[ij]) - K_pno_ij->vector_dot(Tt_pno_ij);

        // New PNO transformation matrix
        X_pno_ij = linalg::doublet(X_pno_[ij], X_pno_ij, false, false);

        K_iajb_[ij] = K_pno_ij;
        T_iajb_[ij] = T_pno_ij;
        Tt_iajb_[ij] = Tt_pno_ij;
        X_pno_[ij] = X_pno_ij;
        e_pno_[ij] = e_pno_ij;
        n_pno_[ij] = X_pno_ij->colspi(0);
        occ_pno_[ij] = pno_occ.get(n_pno_[ij] - 1);
        trace_pno_[ij] = occ_pno / occ_total;
        de_pno_[ij] += de_pno_ij;

        // account for symmetry
        if (i < j) {
            K_iajb_[ji] = K_iajb_[ij]->transpose();
            T_iajb_[ji] = T_iajb_[ij]->transpose();
            Tt_iajb_[ji] = Tt_iajb_[ij]->transpose();
            X_pno_[ji] = X_pno_[ij];
            e_pno_[ji] = e_pno_[ij];
            n_pno_[ji] = n_pno_[ij];
            occ_pno_[ji] = occ_pno_[ij];
            trace_pno_[ji] = trace_pno_[ij];
            de_pno_[ji] += de_pno_ij;
        } // end if (i < j)
    }

    // Print out PNO domain information
    int pno_count_total = 0, pno_count_min = nbf, pno_count_max = 0;
    double occ_number_total = 0.0, occ_number_min = 2.0, occ_number_max = 0.0;
        double trace_total = 0.0, trace_min = 1.0, trace_max = 0.0;
    de_lmp2_weak_ = 0.0, de_pno_total_ = 0.0, de_pno_total_os_ = 0.0, de_pno_total_ss_ = 0.0;
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        if (i_j_to_ij_strong_[i][j] == -1) de_lmp2_weak_ += K_iajb_[ij]->vector_dot(Tt_iajb_[ij]);

        pno_count_total += n_pno_[ij];
        pno_count_min = std::min(pno_count_min, n_pno_[ij]);
        pno_count_max = std::max(pno_count_max, n_pno_[ij]);
        occ_number_total += occ_pno_[ij];
        occ_number_min = std::min(occ_number_min, occ_pno_[ij]);
        occ_number_max = std::max(occ_number_max, occ_pno_[ij]);
        trace_total += trace_pno_[ij];
        trace_min = std::min(trace_min, trace_pno_[ij]);
        trace_max = std::max(trace_max, trace_pno_[ij]);
        de_pno_total_ += de_pno_[ij];
    }

    outfile->Printf("  \n");
    outfile->Printf("    Natural Orbitals per Local MO pair:\n");
    outfile->Printf("      Avg: %3d NOs \n", pno_count_total / n_lmo_pairs);
    outfile->Printf("      Min: %3d NOs \n", pno_count_min);
    outfile->Printf("      Max: %3d NOs \n", pno_count_max);
    outfile->Printf("      Avg Occ Number Tol: %.3e \n", occ_number_total / n_lmo_pairs);
    outfile->Printf("      Min Occ Number Tol: %.3e \n", occ_number_min);
    outfile->Printf("      Max Occ Number Tol: %.3e \n", occ_number_max);
    outfile->Printf("      Avg Trace Sum: %.6f \n", trace_total / n_lmo_pairs);
    outfile->Printf("      Min Trace Sum: %.6f \n", trace_min);
    outfile->Printf("      Max Trace Sum: %.6f \n", trace_max);
    outfile->Printf("  \n");
    outfile->Printf("    Weak pair energy      = %.12f\n", de_lmp2_weak_);
    outfile->Printf("    PNO truncation energy = %.12f\n", de_pno_total_);

}

template<bool crude> std::pair<double, double> DLPNOCCSD::filter_pairs(const std::vector<double>& e_ijs) {
    int natom = molecule_->natom();
    int nbf = basisset_->nbf();
    int naocc = i_j_to_ij_.size();
    int n_lmo_pairs = ij_to_i_j_.size();
    int naux = ribasis_->nbf();
    int npao = C_pao_->colspi(0);  // same as nbf

    // Step 2. Split up strong and weak pairs based on e_ijs
    int strong_pair_count = 0, weak_pair_count = 0;
    double delta_e_crude = 0.0, delta_e_weak = 0.0;

    std::vector<std::vector<int>> i_j_to_ij_strong_copy = i_j_to_ij_strong_;

    ij_to_i_j_strong_.clear();
    ij_to_i_j_weak_.clear();

    i_j_to_ij_strong_.clear();
    i_j_to_ij_weak_.clear();

    i_j_to_ij_strong_.resize(naocc);
    i_j_to_ij_weak_.resize(naocc);

    for (size_t i = 0; i < naocc; i++) {
        i_j_to_ij_strong_[i].resize(naocc, -1);
        i_j_to_ij_weak_[i].resize(naocc, -1);

        for (size_t j = 0; j < naocc; j++) {
            int ij = i_j_to_ij_[i][j];
            if (ij == -1) continue;

            if constexpr (crude) {
                if (std::fabs(e_ijs[ij]) >= T_CUT_PAIRS_) { // Strong Pair
                    i_j_to_ij_strong_[i][j] = strong_pair_count;
                    ij_to_i_j_strong_.push_back(std::make_pair(i,j));
                    ++strong_pair_count;
                } else if (std::fabs(e_ijs[ij]) >= T_CUT_PAIRS_MP2_) { // Weak Pair
                    i_j_to_ij_weak_[i][j] = weak_pair_count;
                    ij_to_i_j_weak_.push_back(std::make_pair(i,j));
                    ++weak_pair_count;
                } else { // Crude Pair
                    delta_e_crude += e_ijs[ij];
                }
            } else {
                if (std::fabs(e_ijs[ij]) >= T_CUT_PAIRS_ && i_j_to_ij_strong_copy[i][j] != -1) { // Strong Pair
                    i_j_to_ij_strong_[i][j] = strong_pair_count;
                    ij_to_i_j_strong_.push_back(std::make_pair(i,j));
                    ++strong_pair_count;
                } else { // Weak Pair
                    i_j_to_ij_weak_[i][j] = weak_pair_count;
                    ij_to_i_j_weak_.push_back(std::make_pair(i,j));
                    delta_e_weak += e_ijs[ij];
                    ++weak_pair_count;
                }
            }
            
        } // end j
    } // end i

    ij_to_ji_strong_.clear();
    ij_to_ji_weak_.clear();

    for (size_t ij = 0; ij < ij_to_i_j_strong_.size(); ++ij) {
        size_t i, j;
        std::tie(i, j) = ij_to_i_j_strong_[ij];
        ij_to_ji_strong_.push_back(i_j_to_ij_strong_[j][i]);
    }
    
    for (size_t ij = 0; ij < ij_to_i_j_weak_.size(); ++ij) {
        size_t i, j;
        std::tie(i, j) = ij_to_i_j_weak_[ij];
        ij_to_ji_weak_.push_back(i_j_to_ij_weak_[j][i]);
    }

    // Recompute global pair lists
    ij_to_i_j_.clear();
    i_j_to_ij_.clear();

    i_j_to_ij_.resize(naocc);

    int ij = 0;
    for (size_t i = 0; i < naocc; i++) {
        i_j_to_ij_[i].resize(naocc, -1);

        for (size_t j = 0; j < naocc; j++) {
            if (i_j_to_ij_strong_[i][j] != -1 || i_j_to_ij_weak_[i][j] != -1) {
                i_j_to_ij_[i][j] = ij;
                ij_to_i_j_.push_back(std::make_pair(i,j));
                ++ij;
            } // end if
        } // end j
    } // end i

    ij_to_ji_.clear();

    for (size_t ij = 0; ij < ij_to_i_j_.size(); ++ij) {
        size_t i, j;
        std::tie(i, j) = ij_to_i_j_[ij];
        ij_to_ji_.push_back(i_j_to_ij_[j][i]);
    }

    return std::make_pair(delta_e_crude, delta_e_weak);
}

template<bool crude> void DLPNOCCSD::pair_prescreening() {
    
    int naocc = i_j_to_ij_.size();

    if constexpr (crude) {
        outfile->Printf("\n  ==> Determining Strong and Weak Pairs (Crude Prescreening Step) <==\n");

        int n_lmo_pairs_init = ij_to_i_j_.size();

        const std::vector<double>& e_ijs_crude = compute_pair_energies<true>();
        std::tie(de_lmp2_eliminated_, de_lmp2_weak_) = filter_pairs<true>(e_ijs_crude);

        int n_strong_pairs = ij_to_i_j_strong_.size();
        int n_weak_pairs = ij_to_i_j_weak_.size();
        int n_surviving_pairs = n_strong_pairs + n_weak_pairs;
        int n_eliminated_pairs = n_lmo_pairs_init - n_surviving_pairs;

        outfile->Printf("    Eliminated Pairs                = %d\n", n_eliminated_pairs);
        outfile->Printf("    (Initial) Weak Pairs            = %d\n", n_weak_pairs);
        outfile->Printf("    (Initial) Strong Pairs          = %d\n", n_strong_pairs);
        outfile->Printf("    Surviving Pairs / Total Pairs   = (%.2f %%)\n", (100.0 * n_surviving_pairs) / (naocc * naocc));
        outfile->Printf("    Eliminated Pair dE              = %.12f\n\n", de_lmp2_eliminated_);
    } else {
        outfile->Printf("\n  ==> Determining Strong and Weak Pairs (Refined Prescreening Step) <==\n\n");

        int n_lmo_pairs_init = ij_to_i_j_.size();

        const std::vector<double>& e_ijs = compute_pair_energies<false>();
        double de_lmp2_eliminated_refined = 0.0;
        std::tie(de_lmp2_eliminated_refined, de_lmp2_weak_) = filter_pairs<false>(e_ijs);
        de_lmp2_eliminated_ += de_lmp2_eliminated_refined;

        int n_strong_pairs = ij_to_i_j_strong_.size();
        int n_weak_pairs = ij_to_i_j_weak_.size();
        int n_surviving_pairs = n_strong_pairs + n_weak_pairs;
        int n_eliminated_pairs = n_lmo_pairs_init - n_surviving_pairs;

        outfile->Printf("    (Additional) Eliminated Pairs   = %d\n", n_eliminated_pairs);
        outfile->Printf("    (Final) Weak Pairs              = %d\n", n_weak_pairs);
        outfile->Printf("    (Final) Strong Pairs            = %d\n", n_strong_pairs);
        outfile->Printf("    Strong Pairs / Total Pairs      = (%.2f %%)\n", (100.0 * n_strong_pairs) / (naocc * naocc));
        outfile->Printf("    (Cumulative) Eliminated Pair dE = %.12f\n", de_lmp2_eliminated_);
        outfile->Printf("    Weak Pair dE                    = %.12f\n\n", de_lmp2_weak_);
    }
}

void DLPNOCCSD::compute_cc_integrals() {
    outfile->Printf("    Computing CC integrals...\n\n");

    int n_lmo_pairs = ij_to_i_j_.size();
    // 0 virtual
    K_mnij_.resize(n_lmo_pairs);
    // 1 virtual
    K_bar_chem_.resize(n_lmo_pairs);
    K_bar_.resize(n_lmo_pairs);
    L_bar_.resize(n_lmo_pairs);
    // 2 virtual
    J_ijab_.resize(n_lmo_pairs);
    L_iajb_.resize(n_lmo_pairs);
    M_iajb_.resize(n_lmo_pairs);
    // 3 virtual
    K_tilde_chem_.resize(n_lmo_pairs);
    K_tilde_phys_.resize(n_lmo_pairs);
    L_tilde_.resize(n_lmo_pairs);
    // 4 virtual

    // DF integrals (used in DLPNO-CCSD with T1 Transformed Hamiltonian)
    Qma_ij_.resize(n_lmo_pairs);
    Qab_ij_.resize(n_lmo_pairs);

    i_Qa_ij_.resize(n_lmo_pairs);
    i_Qk_ij_.resize(n_lmo_pairs);

    n_svd_.resize(n_lmo_pairs);

    size_t qvv_memory = 0;
    size_t qvv_svd_memory = 0;

    if (write_qab_pao_) {
        psio_->open(PSIF_DLPNO_QAB_PAO, PSIO_OPEN_OLD);
    }

    if (write_qab_pno_) {
        psio_->open(PSIF_DLPNO_QAB_PNO, PSIO_OPEN_NEW);
    }

#pragma omp parallel for schedule(dynamic, 1) reduction(+ : qvv_memory) reduction(+ : qvv_svd_memory)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];
        const int ji = ij_to_ji_[ij];

        // number of PNOs in the pair domain
        const int npno_ij = n_pno_[ij];
        if (npno_ij == 0) continue;

        // number of LMOs in the pair domain
        const int nlmo_ij = lmopair_to_lmos_[ij].size();
        // number of PAOs in the pair domain (before removing linear dependencies)
        const int npao_ij = lmopair_to_paos_[ij].size();
        // number of auxiliary functions in the pair domain
        const int naux_ij = lmopair_to_ribfs_[ij].size();

        auto q_pair = std::make_shared<Matrix>(naux_ij, 1);

        auto q_io = std::make_shared<Matrix>(naux_ij, nlmo_ij);
        auto q_jo = std::make_shared<Matrix>(naux_ij, nlmo_ij);

        auto q_jv = std::make_shared<Matrix>(naux_ij, npno_ij);

        auto q_oo = std::make_shared<Matrix>(naux_ij, nlmo_ij * nlmo_ij);
        auto q_ov = std::make_shared<Matrix>(naux_ij, nlmo_ij * npno_ij);
        auto q_vv = std::make_shared<Matrix>(naux_ij, npno_ij * npno_ij);

        for (int q_ij = 0; q_ij < naux_ij; q_ij++) {
            const int q = lmopair_to_ribfs_[ij][q_ij];
            const int centerq = ribasis_->function_to_center(q);

            const int i_sparse = riatom_to_lmos_ext_dense_[centerq][i];
            const int j_sparse = riatom_to_lmos_ext_dense_[centerq][j];
            const std::vector<int> i_slice(1, i_sparse);
            const std::vector<int> j_slice(1, j_sparse);

            q_pair->set(q_ij, 0, (*qij_[q])(i_sparse, j_sparse));
            
            auto q_io_tmp = submatrix_rows_and_cols(*qij_[q], i_slice, 
                                lmopair_lmo_to_riatom_lmo_[ij][q_ij]);
            C_DCOPY(nlmo_ij, &(*q_io_tmp)(0,0), 1, &(*q_io)(q_ij, 0), 1);

            auto q_jo_tmp = submatrix_rows_and_cols(*qij_[q], j_slice, 
                                lmopair_lmo_to_riatom_lmo_[ij][q_ij]);
            C_DCOPY(nlmo_ij, &(*q_jo_tmp)(0,0), 1, &(*q_jo)(q_ij, 0), 1);

            auto q_oo_tmp = submatrix_rows_and_cols(*qij_[q], lmopair_lmo_to_riatom_lmo_[ij][q_ij],
                                lmopair_lmo_to_riatom_lmo_[ij][q_ij]);
            C_DCOPY(nlmo_ij * nlmo_ij, &(*q_oo_tmp)(0,0), 1, &(*q_oo)(q_ij, 0), 1);

            auto q_jv_tmp = submatrix_rows_and_cols(*qia_[q], j_slice,
                                lmopair_pao_to_riatom_pao_[ij][q_ij]);
            q_jv_tmp = linalg::doublet(q_jv_tmp, X_pno_[ij], false, false);
            C_DCOPY(npno_ij, &(*q_jv_tmp)(0,0), 1, &(*q_jv)(q_ij, 0), 1);

            std::vector<int> m_ij_indices;
            for (const int &m : lmopair_to_lmos_[ij]) {
                const int m_sparse = riatom_to_lmos_ext_dense_[centerq][m];
                m_ij_indices.push_back(m_sparse);
            }

            auto q_ov_tmp = submatrix_rows_and_cols(*qia_[q], m_ij_indices, lmopair_pao_to_riatom_pao_[ij][q_ij]);
            q_ov_tmp = linalg::doublet(q_ov_tmp, X_pno_[ij], false, false);
            C_DCOPY(nlmo_ij * npno_ij, &(*q_ov_tmp)(0,0), 1, &(*q_ov)(q_ij, 0), 1);

            SharedMatrix q_vv_tmp;
            if (write_qab_pao_) {
                std::stringstream toc_entry;
                toc_entry << "QAB (PAO) " << q;
                int npao_q = riatom_to_paos_ext_[centerq].size();
                q_vv_tmp = std::make_shared<Matrix>(toc_entry.str(), npao_q, npao_q);
#pragma omp critical
                q_vv_tmp->load(psio_, PSIF_DLPNO_QAB_PAO, psi::Matrix::LowerTriangle);
                q_vv_tmp = submatrix_rows_and_cols(*q_vv_tmp, lmopair_pao_to_riatom_pao_[ij][q_ij],
                                lmopair_pao_to_riatom_pao_[ij][q_ij]);
            } else {
                q_vv_tmp = submatrix_rows_and_cols(*qab_[q], lmopair_pao_to_riatom_pao_[ij][q_ij],
                                lmopair_pao_to_riatom_pao_[ij][q_ij]);
            }
            q_vv_tmp = linalg::triplet(X_pno_[ij], q_vv_tmp, X_pno_[ij], true, false, false);
            
            C_DCOPY(npno_ij * npno_ij, &(*q_vv_tmp)(0,0), 1, &(*q_vv)(q_ij, 0), 1);
        }

        auto A_solve = submatrix_rows_and_cols(*full_metric_, lmopair_to_ribfs_[ij], lmopair_to_ribfs_[ij]);
        A_solve->power(0.5, 1.0e-14);

        C_DGESV_wrapper(A_solve->clone(), q_pair);
        C_DGESV_wrapper(A_solve->clone(), q_io);
        C_DGESV_wrapper(A_solve->clone(), q_jo);
        C_DGESV_wrapper(A_solve->clone(), q_jv);
        C_DGESV_wrapper(A_solve->clone(), q_oo);
        C_DGESV_wrapper(A_solve->clone(), q_ov);
        C_DGESV_wrapper(A_solve, q_vv);

        K_mnij_[ij] = linalg::doublet(q_io, q_jo, true, false);
        K_bar_[ij] = linalg::doublet(q_io, q_jv, true, false);
        J_ijab_[ij] = linalg::doublet(q_pair, q_vv, true, false);
        J_ijab_[ij]->reshape(npno_ij, npno_ij);

        K_bar_chem_[ij] = linalg::doublet(q_pair, q_ov, true, false);
        K_bar_chem_[ij]->reshape(nlmo_ij, npno_ij);

        K_tilde_chem_[ji] = linalg::doublet(q_jv, q_vv, true, false);
        K_tilde_phys_[ji] = std::make_shared<Matrix>(npno_ij, npno_ij * npno_ij);
        auto K_tilde_temp = std::make_shared<Matrix>(npno_ij, npno_ij * npno_ij);

        for (int a_ij = 0; a_ij < npno_ij; a_ij++) {
            for (int e_ij = 0; e_ij < npno_ij; e_ij++) {
                for (int f_ij = 0; f_ij < npno_ij; f_ij++) {
                    (*K_tilde_phys_[ji])(a_ij, e_ij * npno_ij + f_ij) = 
                                    (*K_tilde_chem_[ji])(e_ij, a_ij * npno_ij + f_ij);
                    (*K_tilde_temp)(a_ij, e_ij * npno_ij + f_ij) = 
                                    (*K_tilde_chem_[ji])(f_ij, a_ij * npno_ij + e_ij);
                }
            }
        }
        L_tilde_[ji] = K_tilde_chem_[ji]->clone();
        L_tilde_[ji]->scale(2.0);
        L_tilde_[ji]->subtract(K_tilde_temp);

        /*
        bool is_strong_pair = (i_j_to_ij_strong_[i][j] != -1);

        if (is_strong_pair && i <= j) {
            // SVD Decomposition of DF-ERIs
            // DOI: 10.1063/1.4905005
            if (T_CUT_SVD_ > 0.0) {
                auto [U, S, V] = q_vv->svd_temps();
                q_vv->svd(U, S, V);

                int nsvd_ij = 0;
                std::vector<int> slice_indices;
                while (nsvd_ij < S->dim() && S->get(nsvd_ij) >= T_CUT_SVD_) {
                    U->scale_column(0, nsvd_ij, S->get(nsvd_ij));
                    slice_indices.push_back(nsvd_ij);
                    nsvd_ij += 1;
                }

                // U(Q_ij, r_ij) S(r_ij) V(r_ij, a_ij * e_ij)
                // U(Q_ij, s_ij) S(s_ij) V(s_ij, b_ij * f_ij)
                U = submatrix_cols(*U, slice_indices);
                auto B_rs = linalg::doublet(U, U, true, false);
                B_rs->power(0.5, 1.0e-14);
            
                q_vv = linalg::doublet(B_rs, submatrix_rows(*V, slice_indices));

                qvv_memory += naux_ij * npno_ij * npno_ij;
                qvv_svd_memory += nsvd_ij * npno_ij * npno_ij;

                n_svd_[ij] = nsvd_ij;
            } else {
                n_svd_[ij] = naux_ij;
            }

            Qab_ij_[ij].resize(q_vv->nrow());

            if (!write_qab_pno_) {
                for (int q_ij = 0; q_ij < q_vv->nrow(); q_ij++) {
                    Qab_ij_[ij][q_ij] = std::make_shared<Matrix>(npno_ij, npno_ij);
                    C_DCOPY(npno_ij * npno_ij, &(*q_vv)(q_ij, 0), 1, &(*Qab_ij_[ij][q_ij])(0, 0), 1);
                }
            } else {
                std::stringstream toc_entry;
                toc_entry << "QAB (PNO) " << ij;
                q_vv->set_name(toc_entry.str());
#pragma omp critical
                q_vv->save(psio_, PSIF_DLPNO_QAB_PNO, psi::Matrix::ThreeIndexLowerTriangle);
            }
        }
        */

        if (i_j_to_ij_strong_[i][j] != -1) {
            i_Qk_ij_[ij] = q_io;
            i_Qa_ij_[ji] = q_jv;
        }

        // Save DF integrals (only for strong pairs)
        if (i <= j && i_j_to_ij_strong_[i][j] != -1) {
            Qma_ij_[ij].resize(naux_ij);
            Qab_ij_[ij].resize(naux_ij);
            for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
                // Save transformed (Q_ij | m_ij a_ij) integrals
                Qma_ij_[ij][q_ij] = std::make_shared<Matrix>(nlmo_ij, npno_ij);
                C_DCOPY(nlmo_ij * npno_ij, &(*q_ov)(q_ij, 0), 1, &(*Qma_ij_[ij][q_ij])(0, 0), 1);

                // Save transformed (Q_ij | a_ij b_ij) integrals
                Qab_ij_[ij][q_ij] = std::make_shared<Matrix>(npno_ij, npno_ij);
                C_DCOPY(npno_ij * npno_ij, &(*q_vv)(q_ij, 0), 1, &(*Qab_ij_[ij][q_ij])(0, 0), 1);
            }
        }

        // L_iajb
        L_iajb_[ij] = K_iajb_[ij]->clone();
        L_iajb_[ij]->scale(2.0);
        L_iajb_[ij]->subtract(K_iajb_[ij]->transpose());

        // Lt_iajb
        M_iajb_[ij] = K_iajb_[ij]->clone();
        M_iajb_[ij]->scale(2.0);
        M_iajb_[ij]->subtract(J_ijab_[ij]);
    }

    // Antisymmetrize K_mbij integrals
#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];
        const int ji = ij_to_ji_[ij];

        // number of PNOs in the pair domain
        const int npno_ij = n_pno_[ij];
        if (npno_ij == 0) continue;

        L_bar_[ij] = K_bar_[ij]->clone();
        L_bar_[ij]->scale(2.0);
        L_bar_[ij]->subtract(K_bar_[ji]);
    }
}

SharedMatrix DLPNOCCSD::compute_Fmi(const std::vector<SharedMatrix>& tau_tilde) {
    timer_on("Compute Fmi");

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    // Equation 40, Term 1
    auto Fmi = F_lmo_->clone();

#pragma omp parallel for schedule(dynamic, 1)
    for (int m = 0; m < naocc; ++m) {
        for (int n = 0; n < naocc; ++n) {
            int mn = i_j_to_ij_[m][n];

            if (mn == -1 || n_pno_[mn] == 0) continue;

            for (int i_mn = 0; i_mn < lmopair_to_lmos_[mn].size(); i_mn++) {
                int i = lmopair_to_lmos_[mn][i_mn];
                int nn = i_j_to_ij_[n][n];

                // Equation 40, Term 3
                std::vector<int> i_mn_slice(1, i_mn);
                auto l_mn_temp = submatrix_rows(*L_bar_[mn], i_mn_slice);
                auto S_nn_mn = S_PNO(nn, mn);
                l_mn_temp = linalg::doublet(S_nn_mn, l_mn_temp, false, true);
                (*Fmi)(m,i) += l_mn_temp->vector_dot(T_ia_[n]);

                // Equation 40, Term 4
                int in = i_j_to_ij_[i][n];
                if (in != -1 && n_pno_[in] > 0) {
                    auto S_in_mn = S_PNO(in, mn);
                    l_mn_temp = linalg::triplet(S_in_mn, L_iajb_[mn], S_in_mn, false, false, true);
                    (*Fmi)(m,i) += l_mn_temp->vector_dot(tau_tilde[in]);
                }
            }
        }
    }

    timer_off("Compute Fmi");

    return Fmi;
}

std::vector<SharedMatrix> DLPNOCCSD::compute_Fbe(const std::vector<SharedMatrix>& tau_tilde) {
    timer_on("Compute Fbe");

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    std::vector<SharedMatrix> Fbe(n_lmo_pairs);

#pragma omp parallel for
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        int npno_ij = n_pno_[ij];
        if (npno_ij == 0) continue;

        bool is_strong_pair = (i_j_to_ij_strong_[i][j] != -1);
        if (!is_strong_pair) continue;

        // Equation 39, Term 1
        Fbe[ij] = std::make_shared<Matrix>("Fbe", npno_ij, npno_ij);
        Fbe[ij]->zero();
        Fbe[ij]->set_diagonal(e_pno_[ij]);

        for (int m_ij = 0; m_ij < lmopair_to_lmos_[ij].size(); m_ij++) {
            int m = lmopair_to_lmos_[ij][m_ij];
            int im = i_j_to_ij_[i][m], mm = i_j_to_ij_[m][m], mj = i_j_to_ij_[m][j], mi = i_j_to_ij_[m][i];

            if (mj != -1 && n_pno_[mj] != 0 && mi != -1 && n_pno_[mi] != 0) {
                // Contribution 1: +ij, mj
                auto S_ij_mj = S_PNO(ij, mj);
                auto S_mm_mj = S_PNO(mm, mj);
                auto T_m_temp = linalg::doublet(S_mm_mj, T_ia_[m], true, false);

                auto Fbe_mj = linalg::doublet(T_m_temp, L_tilde_[mj], true, false);
                Fbe_mj->reshape(n_pno_[mj], n_pno_[mj]);
                Fbe_mj->scale(0.5);

                Fbe[ij]->add(linalg::triplet(S_ij_mj, Fbe_mj, S_ij_mj, false, false, true));

                // Contribution 2: +ij, im
                auto S_ij_im = S_PNO(ij, im);
                auto S_mm_im = S_PNO(mm, im);
                T_m_temp = linalg::doublet(S_mm_im, T_ia_[m], true, false);

                auto Fbe_im = linalg::doublet(T_m_temp, L_tilde_[mi], true, false);
                Fbe_im->reshape(n_pno_[im], n_pno_[im]);
                Fbe_im->scale(0.5);

                Fbe[ij]->add(linalg::triplet(S_ij_im, Fbe_im, S_ij_im, false, false, true));

                // Contribution 3: -ij, mm
                /*
                auto S_ij_mm = S_PNO(ij, mm);
                auto Fbe_mm = linalg::doublet(T_ia_[m], L_tilde_[mm], true, false);
                Fbe_mm->reshape(n_pno_[mm], n_pno_[mm]);

                Fbe[ij]->subtract(linalg::triplet(S_ij_mm, Fbe_mm, S_ij_mm, false, false, true));
                */
            }

            for (int n_ij = 0; n_ij < lmopair_to_lmos_[ij].size(); n_ij++) {
                int n = lmopair_to_lmos_[ij][n_ij];
                int mn = i_j_to_ij_[m][n];

                if (mn != -1 && n_pno_[mn] != 0) {
                    auto S_ij_mn = S_PNO(ij, mn);
                    auto tau_L_temp = linalg::triplet(tau_tilde[mn], L_iajb_[mn], S_ij_mn, false, true, true);
                    Fbe[ij]->subtract(linalg::doublet(S_ij_mn, tau_L_temp, false, false));
                }
            }
        }

    }

    timer_off("Compute Fbe");

    return Fbe;
}

std::vector<SharedMatrix> DLPNOCCSD::compute_Fme() {
    timer_on("Compute Fme");

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    std::vector<SharedMatrix> Fme(n_lmo_pairs);

#pragma omp parallel for
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];

        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int npno_ij = n_pno_[ij];

        if (npno_ij == 0) continue;

        Fme[ij] = std::make_shared<Matrix>("Fme", nlmo_ij, npno_ij);
        Fme[ij]->zero();

        for (int m_ij = 0; m_ij < lmopair_to_lmos_[ij].size(); m_ij++) {
            int m = lmopair_to_lmos_[ij][m_ij];
            int mm = i_j_to_ij_[m][m], mj = i_j_to_ij_[m][j];

            for (int n_ij = 0; n_ij < lmopair_to_lmos_[ij].size(); n_ij++) {
                int n = lmopair_to_lmos_[ij][n_ij];
                int mn = i_j_to_ij_[m][n], nn = i_j_to_ij_[n][n];

                if (mn != -1 && n_pno_[mn] != 0) {
                    auto S_mn_nn = S_PNO(mn, nn);
                    auto T_n_temp = linalg::doublet(S_mn_nn, T_ia_[n], false, false);

                    auto S_mn_ij = S_PNO(mn, ij);
                    auto F_me_temp = linalg::triplet(S_mn_ij, L_iajb_[mn], T_n_temp, true, false, false);
                    C_DAXPY(npno_ij, 1.0, &(*F_me_temp)(0,0), 1, &(*Fme[ij])(m_ij, 0), 1);
                }
            }
        }
    }

    timer_off("Compute Fme");

    return Fme;
}

std::vector<SharedMatrix> DLPNOCCSD::compute_Wmnij(const std::vector<SharedMatrix>& tau) {
    timer_on("Compute Wmnij");
    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    std::vector<SharedMatrix> Wmnij(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        int m, n;
        std::tie(m, n) = ij_to_i_j_[mn];
        int nm = ij_to_ji_[mn];

        int npno_mn = n_pno_[mn];
        if (npno_mn == 0) continue;

        Wmnij[mn] = K_mnij_[mn]->clone();

        int nlmo_mn = lmopair_to_lmos_[mn].size();
        auto T_i_mn = std::make_shared<Matrix>(nlmo_mn, npno_mn);
        T_i_mn->zero();

        for (int i_mn = 0; i_mn < lmopair_to_lmos_[mn].size(); i_mn++) {
            int i = lmopair_to_lmos_[mn][i_mn];
            int ii = i_j_to_ij_[i][i];

            auto S_ii_mn = S_PNO(ii, mn);
            auto T_temp = linalg::doublet(S_ii_mn, T_ia_[i], true, false);
            C_DCOPY(npno_mn, &(*T_temp)(0,0), 1, &(*T_i_mn)(i_mn, 0), 1);
        }

        Wmnij[mn]->add(linalg::doublet(K_bar_[mn], T_i_mn, false, true));
        Wmnij[mn]->add(linalg::doublet(T_i_mn, K_bar_[nm], false, true));

        for (int ij_mn = 0; ij_mn < nlmo_mn * nlmo_mn; ++ij_mn) {
            int i_mn = ij_mn / nlmo_mn, j_mn = ij_mn % nlmo_mn;
            int i = lmopair_to_lmos_[mn][i_mn], j = lmopair_to_lmos_[mn][j_mn];
            int ij = i_j_to_ij_[i][j];
            
            if (ij != -1 && n_pno_[ij] != 0) {
                auto S_mn_ij = S_PNO(mn, ij);
                auto K_temp = linalg::triplet(S_mn_ij, K_iajb_[mn], S_mn_ij, true, false, false);
                (*Wmnij[mn])(i_mn, j_mn) += K_temp->vector_dot(tau[ij]);
            }
        }
    }

    timer_off("Compute Wmnij");

    return Wmnij;
}

std::vector<SharedMatrix> DLPNOCCSD::compute_Wmbej(const std::vector<SharedMatrix>& tau_bar) {
    timer_on("Compute Wmbej");
    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    std::vector<SharedMatrix> Wmbej(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int mj = 0; mj < n_lmo_pairs; ++mj) {
        int m, j;
        std::tie(m, j) = ij_to_i_j_[mj];
        int mm = i_j_to_ij_[m][m], jj = i_j_to_ij_[j][j], jm = ij_to_ji_[mj];
        int npno_mm = n_pno_[mm], npno_mj = n_pno_[mj];

        if (npno_mj == 0) continue;

        Wmbej[mj] = K_iajb_[jm]->clone();
        
        auto S_mm_mj = S_PNO(mm, mj);
        auto S_mj_jj = S_PNO(mj, jj);
        auto tia_temp = linalg::doublet(S_mj_jj, T_ia_[j], false, false);

        auto K_temp1 = std::make_shared<Matrix>(npno_mj, npno_mj);
        K_temp1 = K_tilde_chem_[mj]->clone();
        K_temp1->reshape(npno_mj * npno_mj, npno_mj);
        K_temp1 = linalg::doublet(K_temp1, tia_temp, false, false);
        K_temp1->reshape(npno_mj, npno_mj);
        // auto eye = std::make_shared<Matrix>(npno_mj, npno_mj);
        // eye->identity();
        // K_temp1 = linalg::doublet(eye, K_temp1, false, true);
        
        Wmbej[mj]->add(K_temp1->transpose());

        for (int n_mj = 0; n_mj < lmopair_to_lmos_[mj].size(); n_mj++) {
            int n = lmopair_to_lmos_[mj][n_mj];
            int nn = i_j_to_ij_[n][n], mn = i_j_to_ij_[m][n], jn = i_j_to_ij_[j][n], nj = i_j_to_ij_[n][j];

            auto S_nn_mj = S_PNO(nn, mj);
            auto t_n_temp = linalg::doublet(S_nn_mj, T_ia_[n], true, false);
            C_DGER(npno_mj, npno_mj, -1.0, &(*t_n_temp)(0, 0), 1, &(*K_bar_[jm])(n_mj, 0), 1, &(*Wmbej[mj])(0, 0), npno_mj);

            if (mn != -1 && n_pno_[mn] != 0 && jn != -1 && n_pno_[jn] != 0) {
                auto S_mn_jn = S_PNO(mn, jn);
                auto S_mj_mn = S_PNO(mj, mn);
                auto S_jn_mj = S_PNO(jn, mj);
                auto tau_temp = linalg::triplet(S_mn_jn, tau_bar[jn], S_jn_mj, false, false, false);
                auto K_mn_temp = linalg::doublet(S_mj_mn, K_iajb_[mn], false, false);
                Wmbej[mj]->subtract(linalg::doublet(tau_temp, K_mn_temp, true, true));

                auto T_nj_temp = linalg::triplet(S_mn_jn, T_iajb_[nj], S_jn_mj, false, false, false);
                auto L_mn_temp = linalg::doublet(S_mj_mn, L_iajb_[mn], false, false);
                auto TL_temp = linalg::doublet(T_nj_temp, L_mn_temp, true, true);
                TL_temp->scale(0.5);
                Wmbej[mj]->add(TL_temp);
            } // end if
        } // end n
    } // end mj

    timer_off("Compute Wmbej");

    return Wmbej;
}

std::vector<SharedMatrix> DLPNOCCSD::compute_Wmbje(const std::vector<SharedMatrix>& tau_bar) {

    timer_on("Compute Wmbje");

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    std::vector<SharedMatrix> Wmbje(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int mj = 0; mj < n_lmo_pairs; ++mj) {
        int m, j;
        std::tie(m, j) = ij_to_i_j_[mj];
        int mm = i_j_to_ij_[m][m], jj = i_j_to_ij_[j][j], jm = ij_to_ji_[mj];
        int npno_mm = n_pno_[mm], npno_mj = n_pno_[mj];

        if (npno_mj == 0) continue;

        Wmbje[mj] = J_ijab_[mj]->clone();
        Wmbje[mj]->scale(-1.0);
        
        auto S_mm_mj = S_PNO(mm, mj);
        auto S_mj_jj = S_PNO(mj, jj);
        auto tia_temp = linalg::doublet(S_mj_jj, T_ia_[j], false, false);

        auto K_temp1 = K_tilde_chem_[mj]->clone();
        K_temp1 = linalg::doublet(tia_temp, K_temp1, true, false);
        K_temp1->reshape(npno_mj, npno_mj);
        Wmbje[mj]->subtract(K_temp1);

        for (int n_mj = 0; n_mj < lmopair_to_lmos_[mj].size(); n_mj++) {
            int n = lmopair_to_lmos_[mj][n_mj];
            int nn = i_j_to_ij_[n][n], mn = i_j_to_ij_[m][n], jn = i_j_to_ij_[j][n], mj = i_j_to_ij_[m][j];

            auto S_nn_mj = S_PNO(nn, mj);
            auto t_n_temp = linalg::doublet(S_nn_mj, T_ia_[n], true, false);
            C_DGER(npno_mj, npno_mj, 1.0, &(*t_n_temp)(0, 0), 1, &(*K_bar_chem_[mj])(n_mj, 0), 1, &(*Wmbje[mj])(0, 0), npno_mj);

            if (mn != -1 && n_pno_[mn] != 0 && jn != -1 && n_pno_[jn] != 0) {
                auto tau_temp = linalg::triplet(S_PNO(mn, jn), tau_bar[jn], S_PNO(jn, mj), false, false, false);
                auto K_mn_temp = linalg::doublet(K_iajb_[mn], S_PNO(mn, mj), false, false);
                Wmbje[mj]->add(linalg::doublet(tau_temp, K_mn_temp, true, false));
            }
        }
    }

    timer_off("Compute Wmbje");

    return Wmbje;
}

void DLPNOCCSD::lccsd_iterations() {

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    outfile->Printf("\n  ==> Local CCSD <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    outfile->Printf("                      Corr. Energy    Delta E     Max R1     Max R2     Time (s)\n");

    // => Initialize Singles Amplitudes <= //

    T_ia_.resize(naocc);
#pragma omp parallel for
    for (int i = 0; i < naocc; ++i) {
        int ii = i_j_to_ij_[i][i];
        T_ia_[i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        T_ia_[i]->zero();
    }

    // => Initialize Dressed Doubles Amplitudes <= //

    std::vector<SharedMatrix> tau(n_lmo_pairs);
    std::vector<SharedMatrix> tau_tilde(n_lmo_pairs);
    std::vector<SharedMatrix> tau_bar(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        if (n_pno_[ij] == 0) continue;

        tau[ij] = T_iajb_[ij]->clone();
        tau_tilde[ij] = T_iajb_[ij]->clone();
        tau_bar[ij] = T_iajb_[ij]->clone();
        tau_bar[ij]->scale(0.5);
    }

    // => Initialize Residuals <= //

    std::vector<SharedMatrix> R_ia(naocc);
    std::vector<SharedMatrix> Rn_iajb(n_lmo_pairs);
    std::vector<SharedMatrix> R_iajb(n_lmo_pairs);

    int iteration = 0, max_iteration = options_.get_int("DLPNO_MAXITER");
    double e_curr = 0.0, e_prev = 0.0, r1_curr = 0.0, r2_curr = 0.0;
    bool e_converged = false, r_converged = false;

    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), "LCCSD DIIS", DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);

    double F_CUT = options_.get_double("F_CUT");

    while (!(e_converged && r_converged)) {
        // RMS of residual per single LMO, for assesing convergence
        std::vector<double> R_ia_rms(naocc, 0.0);
        // RMS of residual per LMO pair, for assessing convergence
        std::vector<double> R_iajb_rms(n_lmo_pairs, 0.0);

        std::time_t time_start = std::time(nullptr);

        // Build one-particle intermediates
        auto Fmi = compute_Fmi(tau_tilde);
        auto Fbe = compute_Fbe(tau_tilde);
        auto Fme = compute_Fme();

        // Build two-particle W intermediates
        auto Wmnij = compute_Wmnij(tau);
        auto Wmbej = compute_Wmbej(tau_bar);
        auto Wmbje = compute_Wmbje(tau_bar);

        // Calculate singles residuals from current amplitudes
#pragma omp parallel for
        for (int i = 0; i < naocc; i++) {
            int ii = i_j_to_ij_[i][i];
            int npno_ii = n_pno_[ii];

            // Madriaga Eq. 34, Term 2
            R_ia[i] = linalg::doublet(Fbe[ii], T_ia_[i], false, false);

            for (int m = 0; m < naocc; m++) {
                int im = i_j_to_ij_[i][m], mm = i_j_to_ij_[m][m], mi = i_j_to_ij_[m][i];

                // Madriaga Eq. 34, Term 3
                if (fabs(Fmi->get(m, i)) > F_CUT) {
                    auto S_ii_mm = S_PNO(ii, mm);
                    auto T_m_temp = linalg::doublet(S_ii_mm, T_ia_[m], false, false);
                    T_m_temp->scale(Fmi->get(m,i));
                    R_ia[i]->subtract(T_m_temp);
                }

                if (im != -1 && n_pno_[im] != 0) {
                    // Madriaga Eq. 34, Term 5
                    auto S_mm_im = S_PNO(mm, im);
                    auto temp_t1 = linalg::doublet(S_mm_im, T_ia_[m], true, false);
                    auto S_im_ii = S_PNO(im, ii);
                    R_ia[i]->add(linalg::triplet(S_im_ii, M_iajb_[im], temp_t1, true, false, false));

                    // Madriaga Eq. 34, Term 4
                    int m_im = lmopair_to_lmos_dense_[im][m];
                    std::vector<int> m_im_slice(1, m_im);
                    auto Fe_im = submatrix_rows(*Fme[im], m_im_slice);
                    R_ia[i]->add(linalg::triplet(S_im_ii, Tt_iajb_[im], Fe_im, true, false, true));

                    // Madriaga Eq. 34, Term 6
                    auto T_mi_mm = Tt_iajb_[mi]->clone();
                    T_mi_mm->reshape(n_pno_[mi] * n_pno_[mi], 1);
                    // (npno_ii, npno_mm) (npno_mm, npno_mm * npno_mm) (npno_mm * npno_mm, 1)
                    R_ia[i]->add(linalg::triplet(S_im_ii, K_tilde_phys_[mi], T_mi_mm, true, false, false));
                

                    // Madriaga Eq. 34, Term 7
                    for (int n_im = 0; n_im < lmopair_to_lmos_[im].size(); n_im++) {
                        int n = lmopair_to_lmos_[im][n_im];
                        int mn = i_j_to_ij_[m][n], nm = i_j_to_ij_[n][m];

                        if (mn != -1 && n_pno_[mn] != 0) {
                            int i_mn = lmopair_to_lmos_dense_[mn][i];
                            std::vector<int> i_mn_slice(1, i_mn);
                            auto L_temp = submatrix_rows(*L_bar_[mn], i_mn_slice);

                            auto S_ii_mn = S_PNO(ii, mn);
                            R_ia[i]->subtract(linalg::triplet(S_ii_mn, T_iajb_[mn], L_temp, false, false, true));
                        }
                    }
                }
            }
            R_ia_rms[i] = R_ia[i]->rms();
        }

        // Calculate residuals from current amplitudes
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j], ji = ij_to_ji_[ij];
            int npno_ij = n_pno_[ij], npno_ii = n_pno_[ii], npno_jj = n_pno_[jj];

            if (npno_ij == 0) continue;

            // Buffers for R2 (Save memory)
            SharedMatrix r2_temp;

            if (i_j_to_ij_strong_[i][j] == -1) { // If the pair is a weak pair, amplitudes are NOT updated, and hence are NOT considered
                Rn_iajb[ij] = std::make_shared<Matrix>(npno_ij, npno_ij);
                Rn_iajb[ij]->zero();
                continue;
            }

            // Madriaga Eq. 35, Term 1
            Rn_iajb[ij] = K_iajb_[ij]->clone();
            Rn_iajb[ij]->scale(0.5);

            // Madriaga Eq. 35, Term 2a
            Rn_iajb[ij]->add(linalg::doublet(T_iajb_[ij], Fbe[ij], false, true));

            // Madriaga Eq. 35, Term 5
            if (!write_qab_pno_) { // Read from Core
                int nsvd_ij = (i > j) ? Qab_ij_[ji].size() : Qab_ij_[ij].size();
                for (int q_ij = 0; q_ij < nsvd_ij; q_ij++) {
                    if (i > j) r2_temp = linalg::triplet(Qab_ij_[ji][q_ij], tau[ij], Qab_ij_[ji][q_ij]);
                    else r2_temp = linalg::triplet(Qab_ij_[ij][q_ij], tau[ij], Qab_ij_[ij][q_ij]);
                    r2_temp->scale(0.5);
                    Rn_iajb[ij]->add(r2_temp);
                }
            } else { // Read from Disk
                int nsvd_ij = (i > j) ? n_svd_[ji] : n_svd_[ij];
                int pair_no = (i > j) ? ji : ij;

                std::stringstream toc_entry;
                toc_entry << "QAB (PNO) " << pair_no;

                auto q_vv = std::make_shared<Matrix>(toc_entry.str(), nsvd_ij, npno_ij * npno_ij);
#pragma omp critical
                q_vv->load(psio_, PSIF_DLPNO_QAB_PNO, psi::Matrix::ThreeIndexLowerTriangle);

                for (int q_ij = 0; q_ij < nsvd_ij; q_ij++) {
                    auto qab_ij_temp = std::make_shared<Matrix>(npno_ij, npno_ij);
                    C_DCOPY(npno_ij * npno_ij, &(*q_vv)(q_ij, 0), 1, &(*qab_ij_temp)(0, 0), 1);

                    r2_temp = linalg::triplet(qab_ij_temp, tau[ij], qab_ij_temp);
                    r2_temp->scale(0.5);
                    Rn_iajb[ij]->add(r2_temp);
                }
            }

            // Madriaga Eq. 35, Term 12
            auto S_ij_ii = S_PNO(ij, ii);
            r2_temp = linalg::doublet(S_ij_ii, T_ia_[i], false, false);
            r2_temp = linalg::doublet(r2_temp, K_tilde_phys_[ji], true, false);
            r2_temp->reshape(npno_ij, npno_ij);
            auto eye = std::make_shared<Matrix>(npno_ij, npno_ij);
            eye->identity();
            Rn_iajb[ij]->add(linalg::doublet(r2_temp, eye, true, false));

            // Madriaga Eq. 35, Term 3
            for (int m = 0; m < naocc; m++) {
                int im = i_j_to_ij_[i][m];

                if (im != -1 && n_pno_[im] != 0) {
                    double tf_dot = 0.0;
                    int m_jj = lmopair_to_lmos_dense_[jj][m];
                    std::vector<int> m_jj_slice(1, m_jj);
                    if (m_jj != -1) {
                        tf_dot = T_ia_[j]->vector_dot(submatrix_rows(*Fme[jj], m_jj_slice)->transpose());
                    }
                    double scale_factor = (*Fmi)(m,j) + 0.5 * tf_dot;
                    if (fabs(scale_factor) >= F_CUT) {
                        auto S_im_ij = S_PNO(im, ij);
                        r2_temp = linalg::triplet(S_im_ij, T_iajb_[im], S_im_ij, true, false, false);
                        r2_temp->scale(scale_factor);
                        Rn_iajb[ij]->subtract(r2_temp);
                    }
                }
            }

            for (int m_ij = 0; m_ij < lmopair_to_lmos_[ij].size(); m_ij++) {
                int m = lmopair_to_lmos_[ij][m_ij];
                int im = i_j_to_ij_[i][m], mm = i_j_to_ij_[m][m], mj = i_j_to_ij_[m][j], mi = i_j_to_ij_[m][i];

                // Shared Intermediates
                auto S_ij_mm = S_PNO(ij, mm);
                auto temp_t1 = linalg::doublet(S_ij_mm, T_ia_[m], false, false);

                // Madriaga Eq. 35, Term 2b
                std::vector<int> m_ij_slice(1, m_ij);
                r2_temp = submatrix_rows(*Fme[ij], m_ij_slice);
                r2_temp = linalg::doublet(T_iajb_[ij], r2_temp, false, true);
                C_DGER(npno_ij, npno_ij, -0.5, &(*r2_temp)(0,0), 1, &(*temp_t1)(0,0), 1, &(*Rn_iajb[ij])(0,0), npno_ij);
                
                if (mj != -1 && n_pno_[mj] != 0) {
                    auto S_ij_mj = S_PNO(ij, mj);

                    // Madriaga Eq. 35, Term 10
                    auto S_ii_mj = S_PNO(ii, mj);
                    auto T_i_temp = linalg::doublet(S_ii_mj, T_ia_[i], true, false);
                    r2_temp = linalg::triplet(T_i_temp, K_iajb_[mj], S_ij_mj, true, false, true);
                    C_DGER(npno_ij, npno_ij, -1.0, &(*temp_t1)(0,0), 1, &(*r2_temp)(0,0), 1, &(*Rn_iajb[ij])(0,0), npno_ij);

                    // Madriaga Eq. 35, Term 11
                    r2_temp = linalg::triplet(S_ij_mj, J_ijab_[mj], T_i_temp, false, false, false);
                    C_DGER(npno_ij, npno_ij, -1.0, &(*r2_temp)(0,0), 1, &(*temp_t1)(0,0), 1, &(*Rn_iajb[ij])(0,0), npno_ij);
                }

                // Madriaga Eq. 35, Term 13
                r2_temp = submatrix_rows(*K_bar_[ij], m_ij_slice)->transpose();
                C_DGER(npno_ij, npno_ij, -1.0, &(*temp_t1)(0,0), 1, &(*r2_temp)(0,0), 1, &(*Rn_iajb[ij])(0,0), npno_ij);

                if (mi != -1 && mj != -1 && n_pno_[mi] != 0 && n_pno_[mj] != 0) {
                    auto S_im_ij = S_PNO(im, ij);
                    auto S_mj_mi = S_PNO(mj, mi);
                    auto S_ij_mj = S_PNO(ij, mj);
                    auto Wmbej_mj = linalg::triplet(S_ij_mj, Wmbej[mj], S_mj_mi, false, false, false);
                    auto Wmbje_mj = linalg::triplet(S_ij_mj, Wmbje[mj], S_mj_mi, false, false, false);
                    auto Wmbje_mi = linalg::triplet(S_im_ij, Wmbje[mi], S_mj_mi, true, false, true);

                    // Madriaga Eq. 35, Term 6 (Zmbij term)
                    // Contribution 1: mj
                    r2_temp = linalg::triplet(S_ij_mj, tau[ij], S_ij_mj, true, false, false);
                    r2_temp->reshape(n_pno_[mj] * n_pno_[mj], 1);
                    r2_temp = linalg::doublet(K_tilde_phys_[mj], r2_temp, false, false);
                    r2_temp = linalg::doublet(S_ij_mj, r2_temp, false, false);
                    C_DGER(npno_ij, npno_ij, -0.5, &(*temp_t1)(0,0), 1, &(*r2_temp)(0,0), 1, &(*Rn_iajb[ij])(0,0), npno_ij);
                    // Contribution 2: mi
                    r2_temp = linalg::triplet(S_im_ij, tau[ij], S_im_ij, false, false, true);
                    r2_temp->reshape(n_pno_[mi] * n_pno_[mi], 1);
                    r2_temp = linalg::doublet(K_tilde_phys_[mi], r2_temp, false, false);
                    r2_temp = linalg::doublet(S_im_ij, r2_temp, true, false);
                    C_DGER(npno_ij, npno_ij, -0.5, &(*temp_t1)(0,0), 1, &(*r2_temp)(0,0), 1, &(*Rn_iajb[ij])(0,0), npno_ij);

                    // Madriaga Eq. 35, Term 7
                    r2_temp = T_iajb_[im]->clone();
                    r2_temp->subtract(T_iajb_[im]->transpose());
                    Rn_iajb[ij]->add(linalg::triplet(S_im_ij, r2_temp, Wmbej_mj, true, false, true));

                    // Madriaga Eq. 35, Term 8
                    r2_temp = Wmbej_mj->clone();
                    r2_temp->add(Wmbje_mj);
                    Rn_iajb[ij]->add(linalg::triplet(S_im_ij, T_iajb_[im], r2_temp, true, false, true));

                    // Madriaga Eq. 35, Term 9
                    Rn_iajb[ij]->add(linalg::triplet(S_ij_mj, T_iajb_[mj], Wmbje_mi, false, false, true));
                }

                // Madriaga Eq. 35, Term 4
                for (int n_ij = 0; n_ij < lmopair_to_lmos_[ij].size(); n_ij++) {
                    int n = lmopair_to_lmos_[ij][n_ij];
                    int mn = i_j_to_ij_[m][n];

                    if (mn == -1 || n_pno_[mn] == 0) continue;

                    auto S_ij_mn = S_PNO(ij, mn);

                    int i_mn = lmopair_to_lmos_dense_[mn][i], j_mn = lmopair_to_lmos_dense_[mn][j];
                    if (i_mn != -1 && j_mn != -1) {
                        r2_temp = linalg::triplet(S_ij_mn, tau[mn], S_ij_mn, false, false, true);
                        r2_temp->scale(0.5 * Wmnij[mn]->get(i_mn, j_mn));
                        Rn_iajb[ij]->add(r2_temp);
                    }

                    /*
                    // Wmbej and Wmbje terms taken out to avoid inaccurate integral projections
                    int jn = i_j_to_ij_[j][n], nj = i_j_to_ij_[n][j];
                    if (jn != -1 && im != -1 && n_pno_[jn] != 0 && n_pno_[im] != 0) {
                        auto V_bar = linalg::triplet(K_iajb_[mn], S_PNO(mn, jn), tau_bar[jn]); // (e, b)
                        V_bar->scale(-1.0);
                        auto V_bar_tmp = linalg::triplet(L_iajb_[mn], S_PNO(mn, jn), T_iajb_[nj]); // (e, b)
                        V_bar_tmp->scale(0.5);
                        V_bar->add(V_bar_tmp);
                        V_bar = linalg::triplet(S_PNO(ij, jn), V_bar, S_PNO(mn, im), false, true, false); // (b, e)

                        auto V_tilde = linalg::triplet(tau_bar[jn], S_PNO(jn, mn), K_iajb_[mn], true, false, false); // (b, e)
                        V_tilde = linalg::triplet(S_PNO(ij, jn), V_tilde, S_PNO(mn, im), false, false, false); // (b, e)

                        auto V_sum = V_bar->clone();
                        V_sum->add(V_tilde);

                        auto T_im_temp = T_iajb_[im]->clone();
                        T_im_temp->subtract(T_iajb_[im]->transpose());
                        Rn_iajb[ij]->add(linalg::triplet(S_PNO(ij, im), T_im_temp, V_bar, false, false, true));
                        Rn_iajb[ij]->add(linalg::triplet(S_PNO(ij, im), T_iajb_[im], V_sum, false, false, true));
                    }

                    int in = i_j_to_ij_[i][n];
                    if (in != -1 && mj != -1 && n_pno_[in] != 0 && n_pno_[mj] != 0) {
                        auto V_tilde = linalg::triplet(tau_bar[in], S_PNO(in, mn), K_iajb_[mn], true, false, false); // (b, e)
                        V_tilde = linalg::triplet(S_PNO(ij, in), V_tilde, S_PNO(mn, mj), false, false, false); // (b, e)
                        Rn_iajb[ij]->add(linalg::triplet(S_PNO(ij, mj), T_iajb_[mj], V_tilde, false, false, true)); 
                    }
                    */
                }
            }
        }

        // Compute Doubles Residual from Non-Symmetrized Doubles Residual
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int ji = ij_to_ji_[ij];
            R_iajb[ij] = std::make_shared<Matrix>("R_iajb", n_pno_[ij], n_pno_[ij]);

            if (n_pno_[ij] == 0) continue;

            R_iajb[ij] = Rn_iajb[ij]->clone();
            R_iajb[ij]->add(Rn_iajb[ji]->transpose());

            R_iajb_rms[ij] = R_iajb[ij]->rms();
        }

        // Update Singles Amplitude
#pragma omp parallel for
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            for (int a_ij = 0; a_ij < n_pno_[ii]; ++a_ij) {
                (*T_ia_[i])(a_ij, 0) -= (*R_ia[i])(a_ij, 0) / (e_pno_[ii]->get(a_ij) - F_lmo_->get(i,i));
            }
        }

        // Update Doubles Amplitude
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];

            if (n_pno_[ij] == 0) continue;

            for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                    (*T_iajb_[ij])(a_ij, b_ij) -= (*R_iajb[ij])(a_ij, b_ij) / 
                                (e_pno_[ij]->get(a_ij) + e_pno_[ij]->get(b_ij) - F_lmo_->get(i,i) - F_lmo_->get(j,j));
                }
            }
        }

        // DIIS Extrapolation
        std::vector<SharedMatrix> T_vecs;
        T_vecs.reserve(T_ia_.size() + T_iajb_.size());
        T_vecs.insert(T_vecs.end(), T_ia_.begin(), T_ia_.end());
        T_vecs.insert(T_vecs.end(), T_iajb_.begin(), T_iajb_.end());

        std::vector<SharedMatrix> R_vecs;
        R_vecs.reserve(R_ia.size() + R_iajb.size());
        R_vecs.insert(R_vecs.end(), R_ia.begin(), R_ia.end());
        R_vecs.insert(R_vecs.end(), R_iajb.begin(), R_iajb.end());

        auto T_vecs_flat = flatten_mats(T_vecs);
        auto R_vecs_flat = flatten_mats(R_vecs);

        if (iteration == 0) {
            diis.set_error_vector_size(R_vecs_flat);
            diis.set_vector_size(T_vecs_flat);
        }

        diis.add_entry(R_vecs_flat.get(), T_vecs_flat.get());
        diis.extrapolate(T_vecs_flat.get());

        copy_flat_mats(T_vecs_flat, T_vecs);

        // Update Special Doubles Amplitudes
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ij++) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];

            if (n_pno_[ij] == 0) continue;

            Tt_iajb_[ij] = T_iajb_[ij]->clone();
            Tt_iajb_[ij]->scale(2.0);
            Tt_iajb_[ij]->subtract(T_iajb_[ij]->transpose());

            auto S_ii_ij = S_PNO(ii, ij);
            auto S_jj_ij = S_PNO(jj, ij);
            auto tia_temp = linalg::doublet(S_ii_ij, T_ia_[i], true, false);
            auto tjb_temp = linalg::doublet(S_jj_ij, T_ia_[j], true, false);

            for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                    double t1_cont = tia_temp->get(a_ij, 0) * tjb_temp->get(b_ij, 0);
                    double t2_cont = T_iajb_[ij]->get(a_ij, b_ij);

                    tau[ij]->set(a_ij, b_ij, t2_cont + t1_cont);
                    tau_tilde[ij]->set(a_ij, b_ij, t2_cont + 0.5 * t1_cont);
                    tau_bar[ij]->set(a_ij, b_ij, 0.5 * t2_cont + t1_cont);
                }
            }
        }

        // evaluate convergence using current amplitudes and residuals
        e_prev = e_curr;
        // Compute LCCSD energy
        e_curr = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : e_curr)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];

            if (n_pno_[ij] == 0) continue;
            // ONLY strong pair doubles contribute to CCSD energy
            e_curr += tau[ij]->vector_dot(L_iajb_[ij]);
            if (i_j_to_ij_strong_[i][j] == -1) {
                e_curr -= T_iajb_[ij]->vector_dot(L_iajb_[ij]);
            }
        }
        double r_curr1 = *max_element(R_ia_rms.begin(), R_ia_rms.end());
        double r_curr2 = *max_element(R_iajb_rms.begin(), R_iajb_rms.end());

        r_converged = (fabs(r_curr1) < options_.get_double("R_CONVERGENCE"));
        r_converged &= (fabs(r_curr2) < options_.get_double("R_CONVERGENCE"));
        e_converged = (fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE"));

        std::time_t time_stop = std::time(nullptr);

        outfile->Printf("  @LCCSD iter %3d: %16.12f %10.3e %10.3e %10.3e %8d\n", iteration, e_curr, e_curr - e_prev, r_curr1, r_curr2, (int)time_stop - (int)time_start);

        iteration++;

        if (iteration > max_iteration) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }
    }

    e_lccsd_ = e_curr;
}

void DLPNOCCSD::t1_ints() {

    timer_on("DLPNO-CCSD: T1 Ints");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int naux_ij = lmopair_to_ribfs_[ij].size();
        int npno_ij = n_pno_[ij];
        int pair_idx = (i > j) ? ij_to_ji_[ij] : ij;
        int i_ij = lmopair_to_lmos_dense_[ij][i], j_ij = lmopair_to_lmos_dense_[ij][j];

        if (i_j_to_ij_strong_[i][j] == -1) continue;

        i_Qk_t1_[ij] = i_Qk_ij_[ij]->clone();
        for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
            for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
                for (int a_ij = 0; a_ij < npno_ij; ++a_ij) {
                    (*i_Qk_t1_[ij])(q_ij, k_ij) += (*Qma_ij_[pair_idx][q_ij])(k_ij, a_ij) * (*T_n_ij_[ij])(i_ij, a_ij);
                }
            }
        }

        i_Qa_t1_[ij] = i_Qa_ij_[ij]->clone();
        i_Qa_t1_[ij]->subtract(linalg::doublet(i_Qk_ij_[ij], T_n_ij_[ij]));
        for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
            auto Qtemp = Qab_ij_[pair_idx][q_ij]->clone();
            Qtemp->subtract(linalg::doublet(T_n_ij_[ij], Qma_ij_[pair_idx][q_ij], true, false));

            for (int a_ij = 0; a_ij < npno_ij; ++a_ij) {
                for (int b_ij = 0; b_ij < npno_ij; ++b_ij) {
                    (*i_Qa_t1_[ij])(q_ij, a_ij) += (*Qtemp)(a_ij, b_ij) * (*T_n_ij_[ij])(i_ij, b_ij);
                } // end b_ij
            } // end a_ij
        } // end q_ij

        if (i <= j) {
            Qab_t1_[ij].resize(naux_ij);
            for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
                Qab_t1_[ij][q_ij] = Qab_ij_[ij][q_ij]->clone();
                Qab_t1_[ij][q_ij]->subtract(linalg::doublet(T_n_ij_[ij], Qma_ij_[ij][q_ij], true, false));
            }
        }
    }

    timer_off("DLPNO-CCSD: T1 Ints");
}

void DLPNOCCSD::t1_fock() {

    timer_on("DLPNO-CCSD: T1 Fock");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

    // => Step 1: Dressing over the contracted indices <= //

    SharedMatrix Fij_bar = F_lmo_->clone();
    std::vector<SharedMatrix> Fia_bar(n_lmo_pairs);
    std::vector<SharedMatrix> Fai_bar(naocc);
    std::vector<SharedMatrix> Fab_bar(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];
        int pair_idx = (i > j) ? ji : ij;

        int naux_ij = lmopair_to_ribfs_[ij].size();
        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int npno_ij = n_pno_[ij];

        // Partially dress Fij
        (*Fij_bar)(i, j) += 2.0 * T_n_ij_[ij]->vector_dot(K_bar_chem_[ij]);
        (*Fij_bar)(i, j) -= T_n_ij_[ij]->vector_dot(K_bar_[ji]);

        if (i > j || i_j_to_ij_strong_[i][j] == -1) continue;

        // Partially dress Fia and Fab
        Fia_bar[ij] = std::make_shared<Matrix>(nlmo_ij, npno_ij);

        for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
            auto Qma = Qma_ij_[pair_idx][q_ij]->clone();
            auto Qab = Qab_ij_[pair_idx][q_ij]->clone();
            double gamma = Qma->vector_dot(T_n_ij_[ij]);

            // J like contributions
            auto Jcont = Qma->clone();
            Jcont->scale(2.0 * gamma);
            Fia_bar[ij]->add(Jcont);

            // Jcont = Qab->clone();
            // Jcont->scale(2.0 * gamma);
            // Fab_bar[ij]->add(Jcont);

            // K like contributions
            auto Kcont = linalg::triplet(Qma, T_n_ij_[ij], Qma, false, true, false);
            Fia_bar[ij]->subtract(Kcont);

            // Kcont = linalg::triplet(Qab, T_n_ij_[ij], Qma, false, true, false);
            // Fab_bar[ij]->subtract(Kcont);
        }

        Fab_bar[ij] = std::make_shared<Matrix>(npno_ij, npno_ij);
        Fab_bar[ij]->set_diagonal(e_pno_[ij]);

        for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
            int k = lmopair_to_lmos_[ij][k_ij];
            int kk = i_j_to_ij_[k][k];

            auto L_temp = linalg::doublet(T_ia_[k], L_tilde_[kk], true, false);
            L_temp->reshape(n_pno_[kk], n_pno_[kk]);
            Fab_bar[ij]->add(linalg::triplet(S_PNO(ij, kk), L_temp, S_PNO(kk, ij)));
        }

        // Partially dress Fai
        if (i == j) {
            Fai_bar[i] = std::make_shared<Matrix>(npno_ij, 1);

            auto Qia = i_Qa_ij_[ij]->clone();
            auto Qik = i_Qk_ij_[ij]->clone();

            for (int q_i = 0; q_i < naux_ij; ++q_i) {
                auto Qma = Qma_ij_[pair_idx][q_i];
                double gamma = Qma->vector_dot(T_n_ij_[ij]);

                auto Qab = Qab_ij_[pair_idx][q_i];
                auto lambda = linalg::doublet(Qab, T_n_ij_[ij], false, true);

                for (int a_i = 0; a_i < npno_ij; ++a_i) {
                    // J like contribution
                    (*Fai_bar[i])(a_i, 0) += 2.0 * gamma * (*Qia)(q_i, a_i);
                    for (int k_i = 0; k_i < nlmo_ij; ++k_i) {
                        // K like contribution
                        (*Fai_bar[i])(a_i, 0) -= (*lambda)(a_i, k_i) * (*Qik)(q_i, k_i);
                    } // end k_i
                } // end a_i   
            } // end q_i
        } // end i == j

    } // end ij

    // => Step 2: Dressing over the free/non-contracted indices <= //

    Fkj_ = Fij_bar->clone();
    Fkc_.resize(n_lmo_pairs);
    Fai_.resize(naocc);
    Fab_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int i_ij = lmopair_to_lmos_dense_[ij][i], j_ij = lmopair_to_lmos_dense_[ij][j];
        int ji = ij_to_ji_[ij], jj = i_j_to_ij_[j][j];
        int pair_idx = (i > j) ? ji : ij;

        int naux_ij = lmopair_to_ribfs_[ij].size();
        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int npno_ij = n_pno_[ij];

        // Fully dress Fkj matrices
        int i_jj = lmopair_to_lmos_dense_[jj][i];
        for (int a_jj = 0; a_jj < n_pno_[jj]; ++a_jj) {
            (*Fkj_)(i, j) += (*Fia_bar[jj])(i_jj, a_jj) * (*T_ia_[j])(a_jj, 0);
        }

        // Fkc matrices (built separately since Fia intermediate not built for weak pairs)
        Fkc_[ij] = std::make_shared<Matrix>(1, npno_ij);

        for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
            int k = lmopair_to_lmos_[ij][k_ij];
            int ik = i_j_to_ij_[i][k], kk = i_j_to_ij_[k][k];

            // Fkc contributions
            auto T_k = linalg::doublet(S_PNO(ik, kk), T_ia_[k]);
            Fkc_[ij]->add(linalg::triplet(S_PNO(ij, ik), L_iajb_[ik], T_k)->transpose());
        }

        if (i_j_to_ij_strong_[i][j] == -1 || i > j) continue;

        // Fully dress Fab matrices
        Fab_[ij] = Fab_bar[ij]->clone();
        Fab_[ij]->subtract(linalg::doublet(T_n_ij_[ij], Fia_bar[ij], true, false));

        // Fully dress Fai matrices (the worst term)
        if (i == j) {
            Fai_[i] = Fai_bar[i]->clone();
            Fai_[i]->add(linalg::doublet(Fab_bar[ij], T_ia_[i], false, false));
            Fai_[i]->subtract(linalg::triplet(T_n_ij_[ij], Fia_bar[ij], T_ia_[i], true, false, false));

            for (int a_i = 0; a_i < npno_ij; ++a_i) {
                for (int k_i = 0; k_i < nlmo_ij; ++k_i) {
                    int k = lmopair_to_lmos_[ij][k_i];
                    (*Fai_[i])(a_i, 0) -= (*T_n_ij_[ij])(k_i, a_i) * (*Fij_bar)(k, i);
                } // end k_i
            } // end a_i
            
        } // end i == j
    }

    timer_off("DLPNO-CCSD: T1 Fock");
}

std::vector<SharedMatrix> DLPNOCCSD::compute_B_tilde() {
    timer_on("DLPNO-CCSD: B tilde");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

    std::vector<SharedMatrix> B_tilde(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        if (i_j_to_ij_strong_[i][j] == -1) continue;

        int naux_ij = lmopair_to_ribfs_[ij].size();
        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int pair_idx = (i > j) ? ji : ij;
        int i_ij = lmopair_to_lmos_dense_[ij][i], j_ij = lmopair_to_lmos_dense_[ij][j];

        B_tilde[ij] = linalg::doublet(i_Qk_t1_[ij], i_Qk_t1_[ji], true, false);

        for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
            B_tilde[ij]->add(linalg::triplet(Qma_ij_[pair_idx][q_ij], T_iajb_[ij], Qma_ij_[pair_idx][q_ij], false, false, true));
        }
    }

    timer_off("DLPNO-CCSD: B tilde");

    return B_tilde;
}

std::vector<std::vector<SharedMatrix>> DLPNOCCSD::compute_gamma() {
    
    timer_on("DLPNO-CCSD: Form gamma");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

    std::vector<std::vector<SharedMatrix>> gamma(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        if (i_j_to_ij_strong_[i][j] == -1) continue;

        int naux_ij = lmopair_to_ribfs_[ij].size();
        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int pair_idx = (i > j) ? ji : ij;
        int i_ij = lmopair_to_lmos_dense_[ij][i], j_ij = lmopair_to_lmos_dense_[ij][j];
        gamma[ij].resize(nlmo_ij);

        // Contract T1-dressed DF-integrals
        for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
            int k = lmopair_to_lmos_[ij][k_ij];
            int kj = i_j_to_ij_[k][j];
            
            gamma[ij][k_ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
            for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
                C_DAXPY(n_pno_[ij] * n_pno_[ij], (*i_Qk_t1_[ij])(q_ij, k_ij), Qab_t1_[pair_idx][q_ij]->get_pointer(), 1, gamma[ij][k_ij]->get_pointer(), 1);
            }
            gamma[ij][k_ij] = linalg::doublet(gamma[ij][k_ij], S_PNO(ij, kj));

            for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
                int l = lmopair_to_lmos_[ij][l_ij];
                int li = i_j_to_ij_[l][i], kl = i_j_to_ij_[k][l];
                if (kl == -1) continue;

                auto T_temp = linalg::doublet(S_PNO(ij, li), T_iajb_[li]);
                auto K_temp = linalg::triplet(S_PNO(li, kl), K_iajb_[kl], S_PNO(kl, kj));
                auto gam_temp = linalg::doublet(T_temp, K_temp);
                gam_temp->scale(0.5);
                gamma[ij][k_ij]->subtract(gam_temp);
            }
                
        }
    }

    timer_off("DLPNO-CCSD: Form gamma");

    return gamma;
}

std::vector<std::vector<SharedMatrix>> DLPNOCCSD::compute_delta() {

    timer_on("DLPNO-CCSD: Form delta");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

    std::vector<std::vector<SharedMatrix>> delta(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        if (i_j_to_ij_strong_[i][j] == -1) continue;

        int naux_ij = lmopair_to_ribfs_[ij].size();
        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int pair_idx = (i > j) ? ji : ij;
        int i_ij = lmopair_to_lmos_dense_[ij][i], j_ij = lmopair_to_lmos_dense_[ij][j];
        delta[ij].resize(nlmo_ij);

        // Contract T1-dressed DF-integrals
        for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
            int k = lmopair_to_lmos_[ij][k_ij];
            int ik = i_j_to_ij_[i][k], jk = i_j_to_ij_[j][k], kj = i_j_to_ij_[k][j];
            int naux_kj = lmopair_to_ribfs_[kj].size();
            int jk_idx = (j > k) ? kj : jk;
            int i_kj = lmopair_to_lmos_dense_[kj][i];

            delta[ij][k_ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
            for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
                C_DGER(n_pno_[ij], n_pno_[ij], 2.0, &(*i_Qa_t1_[ij])(q_ij, 0), 1, &(*Qma_ij_[pair_idx][q_ij])(k_ij, 0), 1, delta[ij][k_ij]->get_pointer(), n_pno_[ij]);
                C_DGER(n_pno_[ij], n_pno_[ij], -2.0, &(*i_Qa_ij_[ij])(q_ij, 0), 1, &(*Qma_ij_[pair_idx][q_ij])(k_ij, 0), 1, delta[ij][k_ij]->get_pointer(), n_pno_[ij]);
                C_DAXPY(n_pno_[ij] * n_pno_[ij], -(*i_Qk_t1_[ij])(q_ij, k_ij), Qab_t1_[pair_idx][q_ij]->get_pointer(), 1, delta[ij][k_ij]->get_pointer(), 1);
            }
            delta[ij][k_ij] = linalg::doublet(delta[ij][k_ij], S_PNO(ij, jk));

            // Error correction Term
            auto delta_err = linalg::triplet(S_PNO(ij, ik), K_iajb_[ik], S_PNO(ik, kj));
            delta_err->scale(2.0);
            delta[ij][k_ij]->add(delta_err);

            for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
                int l = lmopair_to_lmos_[ij][l_ij];
                int il = i_j_to_ij_[i][l], lk = i_j_to_ij_[l][k];
                if (lk == -1) continue;

                auto U_temp = linalg::doublet(S_PNO(ij, il), Tt_iajb_[il]);
                auto L_temp = linalg::triplet(S_PNO(il, lk), L_iajb_[lk], S_PNO(lk, jk));
                auto delt_temp = linalg::doublet(U_temp, L_temp);
                delt_temp->scale(0.5);
                delta[ij][k_ij]->add(delt_temp);
            }
                
        }
    }

    timer_off("DLPNO-CCSD: Form delta");

    return delta;
}

std::vector<SharedMatrix> DLPNOCCSD::compute_E_tilde() {
    
    timer_on("DLPNO-CCSD: E tilde");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

    std::vector<SharedMatrix> E_tilde(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        if (i > j || i_j_to_ij_strong_[i][j] == -1) continue;

        int nlmo_ij = lmopair_to_lmos_[ij].size();

        E_tilde[ij] = Fab_[ij]->clone();

        for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
            int k = lmopair_to_lmos_[ij][k_ij];

            for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
                int l = lmopair_to_lmos_[ij][l_ij];
                int kl = i_j_to_ij_[k][l];
                if (kl == -1) continue;

                auto E_temp = linalg::doublet(Tt_iajb_[kl], K_iajb_[kl], false, true);
                E_tilde[ij]->subtract(linalg::triplet(S_PNO(ij, kl), E_temp, S_PNO(kl, ij)));
            }
        }
    }

    timer_off("DLPNO-CCSD: E tilde");

    return E_tilde;
}

SharedMatrix DLPNOCCSD::compute_G_tilde() {

    timer_on("DLPNO-CCSD: G tilde");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

    SharedMatrix G_tilde = Fkj_->clone();

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        int nlmo_ij = lmopair_to_lmos_[ij].size();

        for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
            int l = lmopair_to_lmos_[ij][l_ij];
            int il = i_j_to_ij_[i][l], lj = i_j_to_ij_[l][j];

            auto U_lj = linalg::triplet(S_PNO(il, lj), Tt_iajb_[lj], S_PNO(lj, il));
            (*G_tilde)(i, j) += K_iajb_[il]->vector_dot(U_lj->transpose());
        }
    }

    timer_off("DLPNO-CCSD: G tilde");

    return G_tilde;
}

void DLPNOCCSD::t1_lccsd_iterations() {

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    // Thread and OMP Parallel info
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    outfile->Printf("\n  ==> Local CCSD (T1-transformed Hamiltonian) <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    outfile->Printf("                      Corr. Energy    Delta E     Max R1     Max R2     Time (s)\n");

    // => Initialize Residuals and Amplitudes <= //

    std::vector<SharedMatrix> R_ia(naocc);
    std::vector<SharedMatrix> Rn_iajb(n_lmo_pairs);
    std::vector<SharedMatrix> R_iajb(n_lmo_pairs);

    // => Initialize Singles Residuals and Amplitudes <= //

    T_ia_.resize(naocc);
#pragma omp parallel for
    for (int i = 0; i < naocc; ++i) {
        int ii = i_j_to_ij_[i][i];
        T_ia_[i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        R_ia[i] = std::make_shared<Matrix>(n_pno_[ii], 1);
    }

    // => Initialize Doubles Residuals and Amplitudes <= //

    std::vector<SharedMatrix> tau(n_lmo_pairs);
#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        if (n_pno_[ij] == 0) continue;

        tau[ij] = T_iajb_[ij]->clone();
        R_iajb[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
        Rn_iajb[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
    }

    // => Thread buffers <= //

    std::vector<std::vector<SharedMatrix>> R_ia_buffer(nthreads);
    for (int thread = 0; thread < nthreads; ++thread) {
        R_ia_buffer[thread].resize(naocc);
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            R_ia_buffer[thread][i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        }
    }

    int iteration = 0, max_iteration = options_.get_int("DLPNO_MAXITER");
    double e_curr = 0.0, e_prev = 0.0, r1_curr = 0.0, r2_curr = 0.0;
    bool e_converged = false, r_converged = false;

    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), "LCCSD DIIS", DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);

    double F_CUT = options_.get_double("F_CUT");

    i_Qk_t1_.resize(n_lmo_pairs);
    i_Qa_t1_.resize(n_lmo_pairs);
    Qab_t1_.resize(n_lmo_pairs);

    T_n_ij_.resize(n_lmo_pairs);

    while (!(e_converged && r_converged)) {
        // RMS of residual per single LMO, for assesing convergence
        std::vector<double> R_ia_rms(naocc, 0.0);
        // RMS of residual per LMO pair, for assessing convergence
        std::vector<double> R_iajb_rms(n_lmo_pairs, 0.0);

        std::time_t time_start = std::time(nullptr);

        // Step 1: Create T_n intermediate
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];

            int nlmo_ij = lmopair_to_lmos_[ij].size();
            int npno_ij = n_pno_[ij];
            
            T_n_ij_[ij] = std::make_shared<Matrix>(nlmo_ij, npno_ij);

            for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
                int n = lmopair_to_lmos_[ij][n_ij];
                int nn = i_j_to_ij_[n][n];
                auto T_n_temp = linalg::doublet(S_PNO(ij, nn), T_ia_[n], false, false);
                
                for (int a_ij = 0; a_ij < npno_ij; ++a_ij) {
                    (*T_n_ij_[ij])(n_ij, a_ij) = (*T_n_temp)(a_ij, 0);
                } // end a_ij
            } // end n_ij
        }

        t1_ints();
        t1_fock();

        auto B_tilde = compute_B_tilde();
        auto gamma = compute_gamma();
        auto delta = compute_delta();
        auto E_tilde = compute_E_tilde();
        auto G_tilde = compute_G_tilde();

        timer_on("DLPNO-CCSD: Compute R1");

        // Initialize R1 residuals, DePrince 2013 Equation 19, Term 1
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            R_ia[i]->copy(Fai_[i]);
        }

        // Zero out buffers
        for (int thread = 0; thread < nthreads; ++thread) {
            for (int i = 0; i < naocc; ++i) {
                R_ia_buffer[thread][i]->zero();
            }
        }

        // Compute residual for singles amplitude (A1 and C contributions)
#pragma omp parallel for schedule(dynamic, 1)
        for (int ik = 0; ik < n_lmo_pairs; ++ik) {
            auto &[i, k] = ij_to_i_j_[ik];
            int ki = ij_to_ji_[ik];

            int nlmo_ik = lmopair_to_lmos_[ik].size();
            int naux_ik = lmopair_to_ribfs_[ik].size();
            int npno_ik = n_pno_[ik];

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            int pair_idx = (i > k) ? ki : ik;

            int i_ik = lmopair_to_lmos_dense_[ik][i], k_ik = lmopair_to_lmos_dense_[ik][k];
            std::vector<int> k_ik_slice = std::vector<int>(1, k_ik);
            int ii = i_j_to_ij_[i][i];

            // A_{i}^{a} = u_{ki}^{cd}B^{Q}_{kc}B^{Q}_{ad} (DePrince 2013 Equation 20)
            auto K_kcad = K_tilde_phys_[ki]->clone();
            auto Uki = Tt_iajb_[ki]->clone();
            Uki->reshape(npno_ik * npno_ik, 1);
            R_ia_buffer[thread][i]->add(linalg::triplet(S_PNO(ii, ki), K_kcad, Uki));

            // C_{i}^{a} = F_{kc}U_{ik}^{ac}  (DePrince 2013 Equation 22)
            R_ia_buffer[thread][i]->add(linalg::triplet(S_PNO(ii, ik), Tt_iajb_[ik], Fkc_[ki], false, false, true));
        } // end ki

        // A2 and B contributions
#pragma omp parallel for schedule(dynamic, 1)
        for (int kl = 0; kl < n_lmo_pairs; ++kl) {
            auto &[k, l] = ij_to_i_j_[kl];

            int naux_kl = lmopair_to_ribfs_[kl].size();
            int nlmo_kl = lmopair_to_lmos_[kl].size();
            int npno_kl = n_pno_[kl];
            int pair_idx = (k > l) ? ij_to_ji_[kl] : kl;
            int k_kl = lmopair_to_lmos_dense_[kl][k], l_kl = lmopair_to_lmos_dense_[kl][l];

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            // B_{i}^{a} = -u_{kl}^{ac}B^{Q}_{ki}B^{Q}_{lc} (DePrince 2013 Equation 21)
            auto K_kilc = K_bar_[kl]->clone();
            K_kilc->add(linalg::doublet(T_n_ij_[kl], K_iajb_[kl]));
            auto B_ia = linalg::doublet(Tt_iajb_[kl], K_kilc, false, true);

            for (int i_kl = 0; i_kl < nlmo_kl; ++i_kl) {
                int i = lmopair_to_lmos_[kl][i_kl];
                int ii = i_j_to_ij_[i][i], ki = i_j_to_ij_[k][i];

                // A2 conribution
                std::vector<int> l_ii_slice(1, lmopair_to_lmos_dense_[ii][l]);
                auto U_ki = linalg::triplet(S_PNO(kl, ki), Tt_iajb_[ki], S_PNO(ki, kl));
                auto T_l = submatrix_rows(*T_n_ij_[ii], l_ii_slice);
                T_l->scale(K_iajb_[kl]->vector_dot(U_ki));
                R_ia_buffer[thread][i]->subtract(T_l->transpose());

                // B contribution
                std::vector<int> i_kl_slice(1, i_kl);
                R_ia_buffer[thread][i]->subtract(linalg::doublet(S_PNO(ii, kl), submatrix_cols(*B_ia, i_kl_slice), false, false));
            }
        }

        // Add R_ia buffers to R_ia
        for (int i = 0; i < naocc; ++i) {
            for (int thread = 0; thread < nthreads; ++thread) {
                R_ia[i]->add(R_ia_buffer[thread][i]);
            }
        }

        // Get rms of R_ia
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < naocc; ++i) {
            R_ia_rms[i] = R_ia[i]->rms();
        }

        timer_off("DLPNO-CCSD: Compute R1");

        timer_on("DLPNO-CCSD: Compute R2");

        // Zero out residuals
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            R_iajb[ij]->zero();
            Rn_iajb[ij]->zero();
        }

        // Compute residual for doubles amplitude
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];
            bool is_weak_pair = (i_j_to_ij_strong_[i][j] == -1);
            int ji = ij_to_ji_[ij];

            int nlmo_ij = lmopair_to_lmos_[ij].size();
            int naux_ij = lmopair_to_ribfs_[ij].size();
            int npno_ij = n_pno_[ij];

            // Skip if pair is weak or contains no pair natural orbitals
            if (is_weak_pair || npno_ij == 0) continue;

            int pair_idx = (i > j) ? ji : ij;

            // Useful information for integral slices
            int i_ij = lmopair_to_lmos_dense_[ij][i], j_ij = lmopair_to_lmos_dense_[ij][j];

            if (i <= j) {
                // R_{ij}^{ab} += (i a_ij | q_ij)' * (q_ij | j b_ij)' (Deprince Equation 10, Term 1)
                auto K_ij = linalg::doublet(i_Qa_t1_[ij], i_Qa_t1_[ji], true, false);
                R_iajb[ij]->add(K_ij);
                if (i != j) R_iajb[ji]->add(K_ij->transpose());

                // A_{ij}^{ab} = B^{Q}_{ac} * t_{ij}^{cd} * B^{Q}_{bd} (DePrince Equation 11)
                auto A_ij = std::make_shared<Matrix>(npno_ij, npno_ij);
                for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
                    A_ij->add(linalg::triplet(Qab_t1_[pair_idx][q_ij], T_iajb_[ij], Qab_t1_[pair_idx][q_ij], false, false, true));
                } // end q_ij
                R_iajb[ij]->add(A_ij);
                if (i != j) R_iajb[ji]->add(A_ij->transpose());

                // B_{ij}^{ab} = t_{kl}^{ab} * [B^{Q}_{ki} * B^{Q}_{lj} + t_{ij}^{cd} * B^{Q}_{kc} * t_{ij}^{cd} * B^{Q}_{ld}]
                // (DePrince Equation 12)
                auto B_ij = std::make_shared<Matrix>(npno_ij, npno_ij);
                for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
                    int k = lmopair_to_lmos_[ij][k_ij];
                    for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
                        int l = lmopair_to_lmos_[ij][l_ij];
                        int kl = i_j_to_ij_[k][l];
                        if (kl == -1 || n_pno_[kl] == 0) continue;

                        auto T_kl = linalg::triplet(S_PNO(ij, kl), T_iajb_[kl], S_PNO(kl, ij));
                        T_kl->scale((*B_tilde[ij])(k_ij, l_ij));
                        B_ij->add(T_kl);

                    } // end l_ij
                } // end k_ij
                R_iajb[ij]->add(B_ij);
                if (i != j) R_iajb[ji]->add(B_ij->transpose());
            } // end if

            // C_{ij}^{ab} = -t_{kj}^{bc}[B^{Q}_{ki}B^{Q}_{ac} - 0.5t_{li}^{ad}(B^{Q}_{kd}B^{Q}_{lc})] 
            // (DePrince Equation 13)
            auto C_ij = std::make_shared<Matrix>(npno_ij, npno_ij);
            for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
                int k = lmopair_to_lmos_[ij][k_ij];
                int ki = i_j_to_ij_[k][i], kj = i_j_to_ij_[k][j];

                C_ij->subtract(linalg::triplet(gamma[ij][k_ij], T_iajb_[kj], S_PNO(kj, ij), false, true, false));
            }
            // Add all the C terms to the non-symmetrized R buffer
            auto C_ij_total = C_ij->clone();
            C_ij_total->scale(0.5);
            C_ij_total->add(C_ij->transpose());
            Rn_iajb[ij]->add(C_ij_total);

            // D_{ij}^{ab} = u_{jk}^{bc}(L_{aikc} + 0.5[u_{il}^{ad}L_{ldkc}]) (DePrince Equation 14)
            auto D_ij = R_iajb[ij]->clone();
            D_ij->zero();
            for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
                int k = lmopair_to_lmos_[ij][k_ij];
                int ik = i_j_to_ij_[i][k], jk = i_j_to_ij_[j][k];

                auto D_temp = linalg::triplet(delta[ij][k_ij], Tt_iajb_[jk], S_PNO(jk, ij), false, true, false);
                D_temp->scale(0.5);
                D_ij->add(D_temp);
            }
            Rn_iajb[ij]->add(D_ij);

            // E_{ij}^{ab} = t_{ij}^{ac} (Fbc - U_{kl}^{bd}[B^{Q}_{ld}B^{Q}_{kc}]) (DePrince Equation 15)
            SharedMatrix E_ij = linalg::doublet(T_iajb_[ij], E_tilde[pair_idx], false, true);
            Rn_iajb[ij]->add(E_ij);

            // G_{ij}^{ab} = -t_{ik}^{ab} (Fkj + U_{lj}^{cd}[B^{Q}_{kd}B^{Q}_{lc}]) (DePrince Equation 16)
            auto G_ij = R_iajb[ij]->clone();
            G_ij->zero();

            for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
                int k = lmopair_to_lmos_[ij][k_ij];
                int ik = i_j_to_ij_[i][k], kj = i_j_to_ij_[k][j];
                if (n_pno_[ik] == 0) continue;

                auto T_ik = linalg::triplet(S_PNO(ij, ik), T_iajb_[ik], S_PNO(ik, ij), false, false, false);
                T_ik->scale((*G_tilde)(k, j));
                G_ij->subtract(T_ik);
            }
            Rn_iajb[ij]->add(G_ij);
        } // end ij

        // Symmetrize residual for doubles amplitude
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            int ji = ij_to_ji_[ij];

            if (n_pno_[ij] != 0 && i_j_to_ij_strong_[i][j] != -1) {
                R_iajb[ij]->add(Rn_iajb[ij]);
                R_iajb[ij]->add(Rn_iajb[ji]->transpose());
            } else {
                R_iajb[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
            }
        }

        timer_off("DLPNO-CCSD: Compute R2");

        // Update Singles Amplitude
#pragma omp parallel for
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            int i_ii = lmopair_to_lmos_dense_[ii][i];
            for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                (*T_ia_[i])(a_ii, 0) -= (*R_ia[i])(a_ii, 0) / ((*Fab_[ii])(a_ii, a_ii) - (*Fkj_)(i,i));
                // (e_pno_[ii]->get(a_ii) - F_lmo_->get(i,i));
            }
        }

        // Update Doubles Amplitude
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];
            int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j], ji = ij_to_ji_[ij];
            if (n_pno_[ij] == 0 || i_j_to_ij_strong_[i][j] == -1) continue;

            int pair_idx = (i > j) ? ji : ij;
            int i_ij = lmopair_to_lmos_dense_[ij][i], j_ij = lmopair_to_lmos_dense_[ij][j];

            for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                    (*T_iajb_[ij])(a_ij, b_ij) -= (*R_iajb[ij])(a_ij, b_ij) / 
                                    ((*Fab_[pair_idx])(a_ij, a_ij) + (*Fab_[pair_idx])(b_ij, b_ij) - (*Fkj_)(i,i) - (*Fkj_)(j,j));
                                    // (e_pno_[ij]->get(a_ij) + e_pno_[ij]->get(b_ij) - F_lmo_->get(i,i) - F_lmo_->get(j,j));
                }
            }
            R_iajb_rms[ij] = R_iajb[ij]->rms();
        }

        // DIIS Extrapolation
        std::vector<SharedMatrix> T_vecs;
        T_vecs.reserve(T_ia_.size() + T_iajb_.size());
        T_vecs.insert(T_vecs.end(), T_ia_.begin(), T_ia_.end());
        T_vecs.insert(T_vecs.end(), T_iajb_.begin(), T_iajb_.end());

        std::vector<SharedMatrix> R_vecs;
        R_vecs.reserve(R_ia.size() + R_iajb.size());
        R_vecs.insert(R_vecs.end(), R_ia.begin(), R_ia.end());
        R_vecs.insert(R_vecs.end(), R_iajb.begin(), R_iajb.end());

        auto T_vecs_flat = flatten_mats(T_vecs);
        auto R_vecs_flat = flatten_mats(R_vecs);

        if (iteration == 0) {
            diis.set_error_vector_size(R_vecs_flat);
            diis.set_vector_size(T_vecs_flat);
        }

        diis.add_entry(R_vecs_flat.get(), T_vecs_flat.get());
        diis.extrapolate(T_vecs_flat.get());

        copy_flat_mats(T_vecs_flat, T_vecs);

        // Update Special Doubles Amplitudes
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ij++) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];

            if (n_pno_[ij] == 0) continue;

            Tt_iajb_[ij] = T_iajb_[ij]->clone();
            Tt_iajb_[ij]->scale(2.0);
            Tt_iajb_[ij]->subtract(T_iajb_[ij]->transpose());

            auto S_ii_ij = S_PNO(ii, ij);
            auto S_jj_ij = S_PNO(jj, ij);
            auto tia_temp = linalg::doublet(S_ii_ij, T_ia_[i], true, false);
            auto tjb_temp = linalg::doublet(S_jj_ij, T_ia_[j], true, false);

            for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                    double t1_cont = tia_temp->get(a_ij, 0) * tjb_temp->get(b_ij, 0);
                    double t2_cont = T_iajb_[ij]->get(a_ij, b_ij);

                    tau[ij]->set(a_ij, b_ij, t2_cont + t1_cont);
                }
            }
        }

        // evaluate convergence using current amplitudes and residuals
        e_prev = e_curr;
        // Compute LCCSD energy
        e_curr = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : e_curr)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];

            if (n_pno_[ij] == 0) continue;
            // ONLY strong pair doubles contribute to CCSD energy
            if (i_j_to_ij_strong_[i][j] == -1) continue;
            e_curr += tau[ij]->vector_dot(L_iajb_[ij]);
        }
        double r_curr1 = *max_element(R_ia_rms.begin(), R_ia_rms.end());
        double r_curr2 = *max_element(R_iajb_rms.begin(), R_iajb_rms.end());

        r_converged = (fabs(r_curr1) < options_.get_double("R_CONVERGENCE"));
        r_converged &= (fabs(r_curr2) < options_.get_double("R_CONVERGENCE"));
        e_converged = (fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE"));

        std::time_t time_stop = std::time(nullptr);

        outfile->Printf("  @LCCSD iter %3d: %16.12f %10.3e %10.3e %10.3e %8d\n", iteration, e_curr, e_curr - e_prev, r_curr1, r_curr2, (int)time_stop - (int)time_start);

        iteration++;

        if (iteration > max_iteration) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }
    }

    e_lccsd_ = e_curr;
}

double DLPNOCCSD::compute_energy() {

    timer_on("DLPNO-CCSD");

    print_header();

    timer_on("Setup Orbitals");
    setup_orbitals();
    timer_off("Setup Orbitals");

    timer_on("Overlap Ints");
    compute_overlap_ints();
    timer_off("Overlap Ints");

    timer_on("Dipole Ints");
    compute_dipole_ints();
    timer_off("Dipole Ints");

    timer_on("Compute Metric");
    compute_metric();
    timer_off("Compute Metric");

    // Adjust parameters for "crude" prescreening
    T_CUT_MKN_ *= 100;
    T_CUT_DO_ *= 2;

    outfile->Printf("  Starting Crude Prescreening...\n");
    outfile->Printf("    T_CUT_MKN set to %6.3e\n", T_CUT_MKN_);
    outfile->Printf("    T_CUT_DO  set to %6.3e\n\n", T_CUT_DO_);

    timer_on("Sparsity");
    prep_sparsity(true, false);
    timer_off("Sparsity");

    timer_on("Crude DF Ints");
    compute_qia();
    timer_off("Crude DF Ints");

    timer_on("Crude Pair Prescreening");
    pair_prescreening<true>();
    timer_off("Crude Pair Prescreening");

    // Reset Sparsity After
    T_CUT_MKN_ *= 0.01;
    T_CUT_DO_ *= 0.5;

    outfile->Printf("  Starting Refined Prescreening...\n");
    outfile->Printf("    T_CUT_MKN reset to %6.3e\n", T_CUT_MKN_);
    outfile->Printf("    T_CUT_DO  reset to %6.3e\n\n", T_CUT_DO_);

    timer_on("Sparsity");
    prep_sparsity(false, false);
    timer_off("Sparsity");

    timer_on("Refined DF Ints");
    compute_qia();
    timer_off("Refined DF Ints");

    timer_on("Refined Pair Prescreening");
    pair_prescreening<false>();
    timer_off("Refined Pair Prescreening");

    timer_on("Sparsity");
    prep_sparsity(false, true);
    timer_off("Sparsity");

    timer_on("PNO-LMP2 Iterations");
    pno_lmp2_iterations();
    timer_off("PNO-LMP2 Iterations");

    timer_on("PNO Overlaps");
    compute_pno_overlaps();
    timer_off("PNO Overlaps");

    timer_on("DF Ints");
    print_integral_sparsity();
    compute_qij();
    compute_qab();
    timer_off("DF Ints");

    timer_on("CC Integrals");
    estimate_memory();
    compute_cc_integrals();
    timer_off("CC Integrals");

    timer_on("LCCSD");
    if (options_.get_bool("DLPNO_T1_HAMILTONIAN")) {
        t1_lccsd_iterations();
    } else {
        lccsd_iterations();
    }
    timer_off("LCCSD");

    if (write_qab_pao_) {
        if (algorithm_ == CCSD) {
            // Integrals no longer needed
            psio_->close(PSIF_DLPNO_QAB_PAO, 0);
        } else {
            // Integrals may still be needed for post-CCSD calculations
            psio_->close(PSIF_DLPNO_QAB_PAO, 1);
        }
    }

    if (write_qab_pno_) {
        // Bye bye (Q_ij | a_ij b_ij) integrals. You won't be missed
        psio_->close(PSIF_DLPNO_QAB_PNO, 0);
    }

    print_results();

    timer_off("DLPNO-CCSD");

    double e_scf = reference_wavefunction_->energy();
    double e_ccsd_corr = e_lccsd_ + de_lmp2_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;
    double e_ccsd_total = e_scf + e_ccsd_corr;

    set_scalar_variable("CCSD CORRELATION ENERGY", e_ccsd_corr);
    set_scalar_variable("CURRENT CORRELATION ENERGY", e_ccsd_corr);
    set_scalar_variable("CCSD TOTAL ENERGY", e_ccsd_total);
    set_scalar_variable("CURRENT ENERGY", e_ccsd_total);

    return e_ccsd_total;
}

void DLPNOCCSD::print_integral_sparsity() {
    // statistics for number of (MN|K) shell triplets we need to compute

    int nbf = basisset_->nbf();
    int nshell = basisset_->nshell();
    int naux = ribasis_->nbf();
    int naocc = nalpha_ - nfrzc();

    size_t triplets = 0;          // computed (MN|K) triplets with no screening
    size_t triplets_lmo = 0;      // computed (MN|K) triplets with only LMO screening
    size_t triplets_pao = 0;      // computed (MN|K) triplets with only PAO screening
    size_t triplets_lmo_lmo = 0;  // computed (MN|K) triplets with LMO and LMO screening
    size_t triplets_lmo_pao = 0;  // computed (MN|K) triplets with LMO and PAO screening
    size_t triplets_pao_pao = 0;  // computed (MN|K) triplets with PAO and PAO screening

    for (size_t atom = 0; atom < riatom_to_shells1_.size(); atom++) {
        size_t nshellri_atom = atom_to_rishell_[atom].size();
        triplets += nshell * nshell * nshellri_atom;
        triplets_lmo += riatom_to_shells1_[atom].size() * nshell * nshellri_atom;
        triplets_pao += nshell * riatom_to_shells2_[atom].size() * nshellri_atom;
        triplets_lmo_lmo += riatom_to_shells1_[atom].size() * riatom_to_shells1_[atom].size() * nshellri_atom;
        triplets_lmo_pao += riatom_to_shells1_[atom].size() * riatom_to_shells2_[atom].size() * nshellri_atom;
        triplets_pao_pao += riatom_to_shells2_[atom].size() * riatom_to_shells2_[atom].size() * nshellri_atom;
    }
    size_t screened_total = 3 * triplets - triplets_lmo_lmo - triplets_lmo_pao - triplets_pao_pao;
    size_t screened_lmo = triplets - triplets_lmo;
    size_t screened_pao = triplets - triplets_pao;

    // statistics for the number of (iu|Q) integrals we're left with after the transformation

    size_t total_integrals = (size_t)naocc * nbf * naux + naocc * naocc * naux + nbf * nbf * naux;
    size_t actual_integrals = 0;

    qij_memory_ = 0;
    qia_memory_ = 0;
    qab_memory_ = 0;

    for (size_t atom = 0; atom < riatom_to_shells1_.size(); atom++) {
        qij_memory_ +=
            riatom_to_lmos_ext_[atom].size() * riatom_to_lmos_ext_[atom].size() * atom_to_ribf_[atom].size();
        qia_memory_ +=
            riatom_to_lmos_ext_[atom].size() * riatom_to_paos_ext_[atom].size() * atom_to_ribf_[atom].size();
        qab_memory_ +=
            riatom_to_paos_ext_[atom].size() * riatom_to_paos_ext_[atom].size() * atom_to_ribf_[atom].size();
    }

    actual_integrals = qij_memory_ + qia_memory_ + qab_memory_;

    // number of doubles * (2^3 bytes / double) * (1 GiB / 2^30 bytes)
    double total_memory = total_integrals * pow(2.0, -27);
    double actual_memory = actual_integrals * pow(2.0, -27);
    double screened_memory = total_memory - actual_memory;

    outfile->Printf("\n");
    outfile->Printf("    Coefficient sparsity in AO -> LMO transform: %6.2f %% \n", screened_lmo * 100.0 / triplets);
    outfile->Printf("    Coefficient sparsity in AO -> PAO transform: %6.2f %% \n", screened_pao * 100.0 / triplets);
    outfile->Printf("    Coefficient sparsity in combined transforms: %6.2f %% \n", screened_total * 100.0 / (3.0 * triplets));
    outfile->Printf("\n");
    outfile->Printf("    Storing transformed LMO/LMO, LMO/PAO, and PAO/PAO integrals in sparse format.\n");
    outfile->Printf("    Required memory: %.3f GiB (%.2f %% reduction from dense format) \n", actual_memory,
                    screened_memory * 100.0 / total_memory);
}

void DLPNOCCSD::print_header() {
    outfile->Printf("   --------------------------------------------\n");
    outfile->Printf("                    DLPNO-CCSD                 \n");
    outfile->Printf("                   by Andy Jiang               \n");
    outfile->Printf("   --------------------------------------------\n\n");
    outfile->Printf("  DLPNO convergence set to %s.\n\n", options_.get_str("PNO_CONVERGENCE").c_str());
    outfile->Printf("  Detailed DLPNO thresholds and cutoffs:\n");
    outfile->Printf("    T_CUT_DO     = %6.3e \n", T_CUT_DO_);
    outfile->Printf("    T_CUT_PNO    = %6.3e \n", T_CUT_PNO_);
    outfile->Printf("    T_CUT_PAIRS  = %6.3e \n", T_CUT_PAIRS_);
    outfile->Printf("    T_CUT_MKN    = %6.3e \n", T_CUT_MKN_);
    outfile->Printf("    T_CUT_SVD    = %6.3e \n", T_CUT_SVD_);
    outfile->Printf("    DIAG_SCALE   = %6.3e \n", T_CUT_PNO_DIAG_SCALE_);
    outfile->Printf("    T_CUT_DO_ij  = %6.3e \n", options_.get_double("T_CUT_DO_ij"));
    outfile->Printf("    T_CUT_PRE    = %6.3e \n", T_CUT_PRE_);
    outfile->Printf("    T_CUT_DO_PRE = %6.3e \n", options_.get_double("T_CUT_DO_PRE"));
    outfile->Printf("    T_CUT_CLMO   = %6.3e \n", options_.get_double("T_CUT_CLMO"));
    outfile->Printf("    T_CUT_CPAO   = %6.3e \n", options_.get_double("T_CUT_CPAO"));
    outfile->Printf("    S_CUT        = %6.3e \n", options_.get_double("S_CUT"));
    outfile->Printf("    F_CUT        = %6.3e \n", options_.get_double("F_CUT"));
    outfile->Printf("    PRESCREENING = %6s   \n", options_.get_str("PRESCREENING_ALGORITHM").c_str());
    outfile->Printf("\n");
}

void DLPNOCCSD::print_results() {
    int naocc = i_j_to_ij_.size();
    double t1diag = 0.0;
#pragma omp parallel for reduction(+ : t1diag)
    for (int i = 0; i < naocc; ++i) {
        t1diag += T_ia_[i]->vector_dot(T_ia_[i]);
    }
    t1diag = std::sqrt(t1diag / (2.0 * naocc));
    outfile->Printf("\n  T1 Diagnostic: %8.8f \n", t1diag);
    if (t1diag > 0.02) {
        outfile->Printf("    WARNING: T1 Diagnostic is greater than 0.02, CCSD results may be unreliable!\n");
    }
    set_scalar_variable("CC T1 DIAGNOSTIC", t1diag);

    outfile->Printf("  \n");
    outfile->Printf("  Total DLPNO-CCSD Correlation Energy: %16.12f \n", e_lccsd_ + de_lmp2_weak_ + de_lmp2_eliminated_ + de_pno_total_ + de_dipole_);
    outfile->Printf("    CCSD Correlation Energy:           %16.12f \n", e_lccsd_);
    outfile->Printf("    Eliminated Pair MP2 Correction     %16.12f \n", de_lmp2_eliminated_);
    outfile->Printf("    Weak Pair MP2 Correction:          %16.12f \n", de_lmp2_weak_);
    outfile->Printf("    Dipole Pair Correction:            %16.12f \n", de_dipole_);
    outfile->Printf("    PNO Truncation Correction:         %16.12f \n", de_pno_total_);
    outfile->Printf("\n\n  @Total DLPNO-CCSD Energy: %16.12f \n", variables_["SCF TOTAL ENERGY"] + e_lccsd_ + de_lmp2_eliminated_ + de_lmp2_weak_ + de_pno_total_ + de_dipole_);
}

}  // namespace dlpno
}  // namespace psi