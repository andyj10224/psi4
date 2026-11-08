/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2024 The Psi4 Developers.
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

/*
 * Equation references and index conventions
 * -----------------------------------------
 * "Toth Eq." refers to E. C. Toth, A. Jiang, and H. F. Schaefer III,
 * J. Chem. Theory Comput. 22, 7667-7681 (2026), DOI: 10.1021/acs.jctc.6c00758.
 * "Jiang Eq." refers to A. Jiang et al., J. Chem. Phys. 161, 082502 (2024),
 * DOI: 10.1063/5.0219963. The latter supplies the right-hand DLPNO-CCSD
 * intermediates reused by the Lambda equations.
 *
 * Occupied indices i, j, ... label LMOs. A compound integer such as ij is the
 * ordered-pair table index i_j_to_ij_[i][j], while a virtual label a_{ij}
 * belongs to that pair's PNO space. S_PNO(target, source) projects a tensor
 * from the source PNO space into the target PNO space. Triples tensors use
 * row a and flattened column (b * nvirtual + c).
 */

void DLPNOCCSD::estimate_lambda_memory() {

    int n_lmo_pairs = ij_to_i_j_.size();

    outfile->Printf(" ==> Lambda DLPNO-CCSD Memory Estimate <== \n\n");

    size_t delta_imae_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : delta_imae_size)
    for (int im = 0; im < n_lmo_pairs; ++im) {
        auto &[i, m] = ij_to_i_j_[im];
        int ii = i_j_to_ij_[i][i], mm = i_j_to_ij_[m][m];

        delta_imae_size += n_pno_[ii] * n_pno_[mm];
    } // end im

    // Memory Estimate for K_{ma_{ii}}^{e_{mi} f_{mi}} intermediate
    size_t K_maef_dt_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : K_maef_dt_size)
    for (int mi = 0; mi < n_lmo_pairs; ++mi) {
        auto &[m, i] = ij_to_i_j_[mi];
        int ii = i_j_to_ij_[i][i];

        int nlmo_mi = lmopair_to_lmos_[mi].size();
        K_maef_dt_size += n_pno_[ii] * n_pno_[mi] * n_pno_[mi];
    } // end mi

    // Memory Estimate for K_{e_{mn} i}^{m n} intermediate
    size_t K_eimn_dt_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : K_eimn_dt_size)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        auto &[m, n] = ij_to_i_j_[mn];
        
        int nlmo_mn = lmopair_to_lmos_[mn].size();
        K_eimn_dt_size += nlmo_mn * n_pno_[mn];
    } // end mn

    size_t M_kace_bar_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : M_kace_bar_size)
    for (int ki = 0; ki < n_lmo_pairs; ++ki) {
        auto &[k, i] = ij_to_i_j_[ki];
        int ii = i_j_to_ij_[i][i];
        
        M_kace_bar_size += n_pno_[ki] * n_pno_[ki] * n_pno_[ii];
    } // end ki

    size_t M_mkic_bar_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : M_mkic_bar_size)
    for (int mk = 0; mk < n_lmo_pairs; ++mk) {
        auto &[m, k] = ij_to_i_j_[mk];
        
        int nlmo_mk = lmopair_to_lmos_[mk].size();
        M_mkic_bar_size += nlmo_mk * n_pno_[mk];
    } // end mk

    size_t F_knia_hat_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : F_knia_hat_size)
    for (int kn = 0; kn < n_lmo_pairs; ++kn) {
        auto &[k, n] = ij_to_i_j_[kn];
        
        int nlmo_kn = lmopair_to_lmos_[kn].size();

        for (int i_kn = 0; i_kn < nlmo_kn; ++i_kn) {
            int i = lmopair_to_lmos_[kn][i_kn];
            int ii = i_j_to_ij_[i][i];

            F_knia_hat_size += n_pno_[ii];
        } // end i_kn
    } // end kn

    size_t L_ieab_bar_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : L_ieab_bar_size)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int jj = i_j_to_ij_[j][j];

        L_ieab_bar_size += n_pno_[jj] * n_pno_[ij] * n_pno_[ij];
    } // end ij
    
    size_t K_mbij_bar_size = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : K_mbij_bar_size)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        int nlmo_ij = lmopair_to_lmos_[ij].size();

        K_mbij_bar_size += nlmo_ij * n_pno_[ij];
    } // end ij

    // 1 GB = 1000^3 = 10^9 Bytes
    const double DOUBLES_TO_GB = pow(10.0, -9) * sizeof(double);
    size_t total_size = delta_imae_size + K_maef_dt_size + K_eimn_dt_size + 2 * M_kace_bar_size + 2 * M_mkic_bar_size + F_knia_hat_size + L_ieab_bar_size + K_mbij_bar_size;

    outfile->Printf("     delta_{im}^{a_{ii} e_{mm}}   : %8.3f [GB]\n", delta_imae_size * DOUBLES_TO_GB);
    outfile->Printf("    (a_{ii}, b_{ij}, c_{ij})-like : %8.3f [GB]\n", (K_maef_dt_size + 2 * M_kace_bar_size + L_ieab_bar_size) * DOUBLES_TO_GB);
    outfile->Printf("    (k_{ij}, a_{ij})-like         : %8.3f [GB]\n", (K_eimn_dt_size + 2 * M_mkic_bar_size + K_mbij_bar_size) * DOUBLES_TO_GB);
    outfile->Printf("    F_knia_hat                    : %8.3f [GB]\n", F_knia_hat_size * DOUBLES_TO_GB);
    outfile->Printf("    Total Memory Required         : %8.3f [GB]\n\n", total_size * DOUBLES_TO_GB);

} // end function

void DLPNOCCSD::form_goo() {
    // Lambda-density contractions rho^OO and rho^VV (Toth Eqs. 53 and 54).
    // They are rebuilt from the current Lambda amplitudes every iteration and
    // are reused by both Lambda residuals and the unrelaxed CCSD OPDM.
    // Number of active occupied orbitals
    int naocc = nalpha_ - nfrzc();
    // Number of surviving pairs after DLPNO screening
    int n_lmo_pairs = ij_to_i_j_.size();

    // \rho^{OO}_{nk} = \sum_{m, e, f} \lambda_{mn}^{e_{mn} f_{mn}} [S(e_{mn}, e_{mk}) T_{mk}^{e_{mk} f_{mk}} S(f_{mk}, f_{mn})]
    rho_oo_ = std::make_shared<Matrix>("rho_oo", naocc, naocc);

#pragma omp parallel for schedule(dynamic, 1)
    for (int nk = 0; nk < n_lmo_pairs; ++nk) {
        auto &[n, k] = ij_to_i_j_[nk];

        for (int m_nk = 0; m_nk < lmopair_to_lmos_[nk].size(); ++m_nk) {
            int m = lmopair_to_lmos_[nk][m_nk];
            int mn = i_j_to_ij_[m][n], mk = i_j_to_ij_[m][k];

            // Transform amplitude from domain of mk to mn [S(e_{mn}, e_{mk}) T_{mk}^{e_{mk} f_{mk}} S(f_{mk}, f_{mn})]
            auto T_mk_to_mn = linalg::triplet(S_PNO(mn, mk), T_iajb_[mk], S_PNO(mk, mn)); // (e_{mn}, f_{mn})

            // \rho^{OO}_{nk} += \lambda_{mn}^{e_{mn} f_{mn}} T_mk_to_mn(e_{mn} f_{mn})
            double val = lambda_iajb_[mn]->vector_dot(T_mk_to_mn) + rho_oo_->get(n, k);
            rho_oo_->set(n, k, val);
        } // end m_nk
    } // end nk

    // \rho^{VV}_{f_{mn} c{mn}} = \sum_{e_{mn}} \lambda_{mn}^{e_{mn} f_{mn}} T_{mn}^{e_{mn} c_{mn}}
    rho_vv_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        auto &[m, n] = ij_to_i_j_[mn];

        rho_vv_[mn] = linalg::doublet(lambda_iajb_[mn], T_iajb_[mn], true, false);
    } // end
}

void DLPNOCCSD::compute_lambda_intermediates() {

    // Build the reusable tensors entering the stationary Lambda equations.
    // Published equation map: Toth Eqs. 33--34, 57, 59--64, 73--74,
    // 103, and 105. beta/gamma/delta and the doubly transformed Fock
    // matrices are shared with the right-hand implementation (Jiang
    // Eqs. 82--86). Keeping this map here makes the residual routines below
    // readable without relying on the order in which storage is allocated.

    outfile->Printf("   ==> Computing Lambda Intermediates <== \n\n");

    // Number of active occupied orbitals
    int naocc = nalpha_ - nfrzc();
    // Number of surviving pairs after DLPNO screening
    int n_lmo_pairs = ij_to_i_j_.size();

    outfile->Printf("   T1-dressing integrals and Fock matrices from converged T1...");

    std::time_t time_start = std::time(nullptr);

    prepare_ccsd_residual();

    std::time_t time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Computing beta, gamma, and delta from converged T2...");

    time_start = std::time(nullptr);
    
    beta_ = compute_beta();
    gamma_ = compute_gamma();
    delta_ = compute_delta();

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming K_maef_dt...");

    time_start = std::time(nullptr);
    
    // Toth Eq. 59 (\widetilde{\widetilde{K}}_{m a_{ii}}^{e_{mi} f_{mi}})
    K_maef_dt_.resize(n_lmo_pairs);
    
#pragma omp parallel for
    for (int mi = 0; mi < n_lmo_pairs; ++mi) {
        auto &[m, i] = ij_to_i_j_[mi];
        int im = ij_to_ji_[mi];
        int ii = i_j_to_ij_[i][i];

        int naux_mi = lmopair_to_ribfs_[mi].size();
        int nlmo_mi = lmopair_to_lmos_[mi].size();
        int npno_mi = n_pno_[mi];

        auto q_vo_t1 = i_Qa_t1_[mi];
        auto q_oo_t1 = i_Qk_t1_[mi];
        auto q_ov = QIA_PNO(mi);
        auto q_vv = QAB_PNO(mi);

        K_maef_dt_[mi] = std::make_shared<Matrix>(npno_mi, npno_mi * npno_mi); // (a_{mi}, e_{mi} f_{mi}) -> (a_{ii}, e_{mi} f_{mi}) later
        K_maef_dt_[mi]->zero();

        // Toth Eq. 59a: +\widetilde{B}^{Q_mi}_{e_mi m}\widetilde{B}^{Q_mi}_{f_mi a_mi}S^{a_mi}_{a_ii}

        for (int q_mi = 0; q_mi < naux_mi; ++q_mi) {
            auto q_vv_t1 = q_vv[q_mi]->clone();
            q_vv_t1->subtract(linalg::doublet(T_n_ij_[mi], q_ov[q_mi], true, false)); // (k_{mi}, f_{mi}) (k_{mi}, a_{mi})

            for (int a_mi = 0; a_mi < n_pno_[mi]; ++a_mi) {
                for (int e_mi = 0; e_mi < n_pno_[mi]; ++e_mi) {
                    for (int f_mi = 0; f_mi < n_pno_[mi]; ++f_mi) {
                        double val = K_maef_dt_[mi]->get(a_mi, e_mi * n_pno_[mi] + f_mi) + q_vv_t1->get(f_mi, a_mi) * q_vo_t1->get(q_mi, e_mi);
                        K_maef_dt_[mi]->set(a_mi, e_mi * n_pno_[mi] + f_mi, val);
                        // (*K_maef_dt_[mi])(a_mi, e_mi * n_pno_[mi] + f_mi) += (*q_vv_t1)(f_mi, a_mi) * (*q_vo_t1)(q_mi, e_mi);
                    } // end f_mi
                } // end e_mi
            } // end a_ii
        } // end q_mi

        K_maef_dt_[mi] = linalg::doublet(S_PNO(ii, mi), K_maef_dt_[mi]); // (a_{mi}, e_{mi} f_{mi}) -> (a_{ii}, e_{mi} f_{mi})

        for (int k_mi = 0; k_mi < nlmo_mi; ++k_mi) {
            int k = lmopair_to_lmos_[mi][k_mi];
            int k_ii = lmopair_to_lmos_dense_[ii][k];
            int mk = i_j_to_ij_[m][k];

            // Toth Eq. 59c: -(S T_mk S)^{e_mi f_mi} \overline{F}_{k_ii a_ii}

            auto T_mk_to_mi = linalg::triplet(S_PNO(mi, mk), T_iajb_[mk], S_PNO(mk, mi));

            for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                for (int e_mi = 0; e_mi < n_pno_[mi]; ++e_mi) {
                    for (int f_mi = 0; f_mi < n_pno_[mi]; ++f_mi) {
                        double val = K_maef_dt_[mi]->get(a_ii, e_mi * npno_mi + f_mi) - Fkc_bar_[ii]->get(k_ii, a_ii) * T_mk_to_mi->get(e_mi, f_mi);
                        K_maef_dt_[mi]->set(a_ii, e_mi * npno_mi + f_mi, val);
                        // (*K_maef_dt_[mi])(a_ii, e_mi * npno_mi + f_mi) -= (*Fkc_bar_[ii])(k_ii, a_ii) * (*T_mk_to_mi)(e_mi, f_mi) * (*Fkc_bar_[ii])(k_ii, a_ii);
                    } // end f_mi
                } // end e_mi
            } // end a_ii

            for (int l_mi = 0; l_mi < nlmo_mi; ++l_mi) {
                int l = lmopair_to_lmos_[mi][l_mi];
                int kl = i_j_to_ij_[k][l];
                if (kl == -1) continue; // checks to make sure kl is a pair

                // Toth Eq. 59b: +(S T_kl S)^{e_mi f_mi}\widetilde{B}^{Q_mi}_{k_mi m}
                // B^{Q_mi}_{l_mi a_mi}S^{a_mi}_{a_ii}
                auto T_kl_to_mi = linalg::triplet(S_PNO(mi, kl), T_iajb_[kl], S_PNO(kl, mi));

                // \widetilde{B}^{Q_{mi}}_{k_{mi} m} B^{Q_{mi}}_{l_{mi} a_{mi}}
                auto g_tilde_kmla_slice = std::make_shared<Matrix>(n_pno_[mi], 1);
                g_tilde_kmla_slice->zero();

                for (int q_mi = 0; q_mi < naux_mi; ++q_mi) {
                    // (S^{a_{ii}}_{a_{mi}}) * B^{Q_{mi}}_{l_{mi}a_{mi}} -> (a_{ii}, 1)
                    for (int a_mi = 0; a_mi < n_pno_[mi]; ++a_mi) {
                        double val = g_tilde_kmla_slice->get(a_mi, 0) + q_ov[q_mi]->get(l_mi, a_mi) * q_oo_t1->get(q_mi, k_mi);
                        g_tilde_kmla_slice->set(a_mi, 0, val);
                    } // end a_mi
                } // end q_mi

                g_tilde_kmla_slice = linalg::doublet(S_PNO(ii, mi), g_tilde_kmla_slice); // (a_{mi}, 1) -> (a_{ii}, 1)

                for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                    for (int e_mi = 0; e_mi < n_pno_[mi]; ++e_mi) {
                        for (int f_mi = 0; f_mi < n_pno_[mi]; ++f_mi) {
                                double val = K_maef_dt_[mi]->get(a_ii, e_mi * npno_mi + f_mi) + T_kl_to_mi->get(e_mi, f_mi) * g_tilde_kmla_slice->get(a_ii, 0);
                                K_maef_dt_[mi]->set(a_ii, e_mi * npno_mi + f_mi, val);
                        } // end f_mi
                    } // end e_mi
                } // end a_ii

            } // end l_mi
        } // end k_mi
    } // end mi

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming K_eimn_dt...");

    time_start = std::time(nullptr);

    // Toth Eq. 60: \widetilde{\widetilde{K}}_{e_mn i}^{mn}

    K_eimn_dt_.resize(n_lmo_pairs);
#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        auto &[m, n] = ij_to_i_j_[mn];
        int nm = ij_to_ji_[mn];

        int naux_mn = lmopair_to_ribfs_[mn].size();
        int nlmo_mn = lmopair_to_lmos_[mn].size();
        int npno_mn = n_pno_[mn];

        auto q_vv = QAB_PNO(mn);
        auto q_ov = QIA_PNO(mn);
        auto q_vo_t1 = i_Qa_t1_[mn];
        auto q_oo_t1 = i_Qk_t1_[nm];

        // Toth Eq. 60a: +\widetilde{B}^{Q_mn}_{e_mn m}\widetilde{B}^{Q_mn}_{i_mn n}
        K_eimn_dt_[mn] = linalg::doublet(q_vo_t1, q_oo_t1, true, false); // (Q, e) (Q, i) -> (e, i)

        for (int q_mn = 0; q_mn < naux_mn; ++q_mn) {
            auto q_vv_t1 = q_vv[q_mn]->clone();
            q_vv_t1->subtract(linalg::doublet(T_n_ij_[mn], q_ov[q_mn], true, false)); // (k_{mi}, f_{mi}) (k_{mi}, a_{mi})

            // Toth Eq. 60b: +\widetilde{B}^{Q_mn}_{e_mn c_mn}T_mn^{c_mn d_mn}B^{Q_mn}_{i_mn d_mn}
            K_eimn_dt_[mn]->add(linalg::triplet(q_vv_t1, T_iajb_[mn], q_ov[q_mn], false, false, true)); // (e, c) (c, d) (i, d) -> (e, i)
        } // end q_mn

        // Toth Eq. 60c: +T_mn^{e_mn c_mn}\overline{F}_{i_mn c_mn}
        int mn_idx = (m < n) ? mn : nm;
        K_eimn_dt_[mn]->add(linalg::doublet(T_iajb_[mn], Fkc_bar_[mn_idx], false, true)); // (e, c) (i, c) -> (e, i)
    } // end mn

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming M_kace_bar_...");

    time_start = std::time(nullptr);

    // Toth Eq. 61: \overline{M}_{k a_ii}^{c_ki e_ki}
    M_kace_bar_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ki = 0; ki < n_lmo_pairs; ++ki) {
        auto &[k, i] = ij_to_i_j_[ki];
        int ii = i_j_to_ij_[i][i];

        int nlmo_ki = lmopair_to_lmos_[ki].size();
        
        M_kace_bar_[ki] = std::make_shared<Matrix>(n_pno_[ki], n_pno_[ki] * n_pno_[ki]);
        
        for (int a_ki = 0; a_ki < n_pno_[ki]; ++a_ki) {
            for (int c_ki = 0; c_ki < n_pno_[ki]; ++c_ki) {
                for (int e_ki = 0; e_ki < n_pno_[ki]; ++e_ki) {
                    double val = 2.0 * K_ivvv_[ki]->get(c_ki, a_ki * n_pno_[ki] + e_ki) - K_ivvv_[ki]->get(a_ki, c_ki * n_pno_[ki] + e_ki);
                    M_kace_bar_[ki]->set(a_ki, c_ki * n_pno_[ki] + e_ki, val);
                    // (*M_kace_bar_[ki])(a_ki, c_ki * n_pno_[ki] + e_ki) = 2.0 * (*K_ivvv_[ki])(c_ki, a_ki * n_pno_[ki] + e_ki)
                            // - (*K_ivvv_[ki])(a_ki, c_ki * n_pno_[ki] + e_ki);
                } // end e_ki
            } // end c_i
        } // end a_ki

        M_kace_bar_[ki] = linalg::doublet(S_PNO(ii, ki), M_kace_bar_[ki]);

        for (int l_ki = 0; l_ki < nlmo_ki; ++l_ki) {
            int l = lmopair_to_lmos_[ki][l_ki];
            int lk = i_j_to_ij_[l][k], ll = i_j_to_ij_[l][l];

            auto L_lk_to_ii_ki = linalg::triplet(S_PNO(ii, lk), L_iajb_[lk], S_PNO(lk, ki));
            auto T_l_ki = linalg::doublet(S_PNO(ki, ll), T_ia_[l]);

            for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                for (int c_ki = 0; c_ki < n_pno_[ki]; ++c_ki) {
                    for (int e_ki = 0; e_ki < n_pno_[ki]; ++e_ki) {
                        double val = -L_lk_to_ii_ki->get(a_ii, c_ki) * T_l_ki->get(e_ki, 0)
                            + M_kace_bar_[ki]->get(a_ii, c_ki * n_pno_[ki] + e_ki);
                        M_kace_bar_[ki]->set(a_ii, c_ki * n_pno_[ki] + e_ki, val);
                        // (*M_kace_bar_[ki])(a_ii, c_ki * n_pno_[ki] + e_ki) -= (*L_lk_to_ii_ki)(a_ii, c_ki) * (*T_l_ki)(e_ki, 0);
                    } // end e_ki
                } // end c_ki
            } // end a_ki
        } // end l_ki

    } // end ki

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming M_mkic_bar...");

    time_start = std::time(nullptr);

    // Toth Eq. 62: \overline{M}_{mk}^{i c_mk}
    M_mkic_bar_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int mk = 0; mk < n_lmo_pairs; ++mk) {
        auto &[m, k] = ij_to_i_j_[mk];
        int mm = i_j_to_ij_[m][m];

        int nlmo_mk = lmopair_to_lmos_[mk].size();
        int npno_mk = n_pno_[mk];

        M_mkic_bar_[mk] = K_mibj_[mk]->clone();
        M_mkic_bar_[mk]->scale(2.0);
        M_mkic_bar_[mk]->subtract(J_ijmb_[mk]);

        for (int i_mk = 0; i_mk < nlmo_mk; ++i_mk) {
            int i = lmopair_to_lmos_[mk][i_mk];
            int ik = i_j_to_ij_[i][k];

            auto T_m_ik = linalg::doublet(S_PNO(ik, mm), T_ia_[m]); // (f_ik, 1)

            auto temp_m = linalg::triplet(T_m_ik, L_iajb_[ik], S_PNO(ik, mk), true, false, false);

            for (int c_mk = 0; c_mk < n_pno_[mk]; ++c_mk) {
                double val = temp_m->get(0, c_mk) + M_mkic_bar_[mk]->get(i_mk, c_mk);
                M_mkic_bar_[mk]->set(i_mk, c_mk, val);
                // (*M_mkic_bar_[mk])(i_mk, c_mk) += (*temp_m)(0, c_mk);
            }
        } // end i_mk
    } // end mk

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming J_kmic_bar...");

    time_start = std::time(nullptr);

    // Toth Eq. 63: \overline{J}_{km}^{i c_km}
    // \overline{J}_{km}^{ic} = (km | i c_{km}) + \widetilde{T}_{m}^{f_{ki}} (k f_{ki} | i c_{ki})
    //  S_{c_{ki}}^{c_{km}}
    J_kmic_bar_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int km = 0; km < n_lmo_pairs; ++km) {
        auto &[k, m] = ij_to_i_j_[km];
        int mm = i_j_to_ij_[m][m];

        J_kmic_bar_[km] = J_ijmb_[km]->clone();

        for (int i_km = 0; i_km < lmopair_to_lmos_[km].size(); ++i_km) {
            int i = lmopair_to_lmos_[km][i_km];
            int ki = i_j_to_ij_[k][i];

            auto T_m_ki = linalg::doublet(S_PNO(ki, mm), T_ia_[m]);
            auto J_kmic_temp = linalg::triplet(T_m_ki, K_iajb_[ki], S_PNO(ki, km), true, false, false); // a column vector of dimension (1, n_pno_[km])

            for (int c_km = 0; c_km < n_pno_[km]; ++c_km) {
                double val = J_kmic_temp->get(0, c_km) + J_kmic_bar_[km]->get(i_km, c_km);
                J_kmic_bar_[km]->set(i_km, c_km, val);
                // (*J_kmic_bar_[km])(i_km, c_km) += (*J_kmic_temp)(0, c_km);
            } // end c_km
        } // end i_km
    } // end km

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming J_kaec_bar...");
    time_start = std::time(nullptr);

    // Toth Eq. 64: \overline{J}_{k a_ii}^{e_ki c_ki}
    // \overline{J}_{ka_{ii}}^{e_{ki}c_{ki}} = S^{a_{ii}}_{a_{ki}} (ka_{ki}|e_{ki}c_{ki}) -
    // S_{a_{ii}}^{a_{kl}} (k a_{kl} | l c_{kl}) S_{c_{kl}}^{c_{ki}} \widetilde{T}_{l}^{e_{ki}}
    J_kaec_bar_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ki = 0; ki < n_lmo_pairs; ++ki) {
        auto &[k, i] = ij_to_i_j_[ki];
        int ii = i_j_to_ij_[i][i];

        J_kaec_bar_[ki] = linalg::doublet(S_PNO(ii, ki), K_ivvv_[ki]); // (a, e * c)

        for (int l_ki = 0; l_ki < lmopair_to_lmos_[ki].size(); ++l_ki) {
            int l = lmopair_to_lmos_[ki][l_ki];
            int kl = i_j_to_ij_[k][l], ll = i_j_to_ij_[l][l];

            // S_{a_{ii}}^{a_{kl}} (k a_{kl} | l c_{kl}) S_{c_{kl}}^{c_{ki}}
            auto K_kl_to_ii_ki = linalg::triplet(S_PNO(ii, kl), K_iajb_[kl], S_PNO(kl, ki));
            auto T_l_ki = linalg::doublet(S_PNO(ki, ll), T_ia_[l]); // \widetilde{T}_{l}^{e_{ki}} (Jiang Eq. 70?)

            for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                for (int e_ki = 0; e_ki < n_pno_[ki]; ++e_ki) {
                    for (int c_ki = 0; c_ki < n_pno_[ki]; ++c_ki) {
                        double val = J_kaec_bar_[ki]->get(a_ii, e_ki * n_pno_[ki] + c_ki) - K_kl_to_ii_ki->get(a_ii, c_ki) * T_l_ki->get(e_ki, 0);
                        J_kaec_bar_[ki]->set(a_ii, e_ki * n_pno_[ki] + c_ki, val);
                    } // end c_ki
                } // end e_ki
            } // end a_ii

        } // end l_ki
    } // end for ki

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    // Toth Eq. 33. This explicit three-virtual intermediate and its matching
    // residual contraction are retained below as documentation but disabled;
    // the active factorized Eq. 68a contraction avoids storing it.
    /*
    F_fcia_hat_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        auto &[m, n] = ij_to_i_j_[mn];

        int naux_mn = lmopair_to_ribfs_[mn].size();
        int nlmo_mn = lmopair_to_lmos_[mn].size();
        int npno_mn = n_pno_[mn];

        F_fcia_hat_[mn].resize(nlmo_mn);

        auto qia_mn = QIA_PNO(mn); // naux_mn * (nlmo_mn, npno_mn)
        auto qab_mn = QAB_PNO(mn); // naux_mn * (npno_mn, npno_mn)

        SharedMatrix F_fcia_temp = std::make_shared<Matrix>("F_fcia_temp", nlmo_mn * npno_mn, npno_mn * npno_mn);

        // Eq 33a and b
        // TODO: Add tilde to B^{Q}_{fc} term
        for (int q_mn = 0; q_mn < naux_mn; ++q_mn) {
            auto q_vv = qab_mn[q_mn]->clone(); // (npno_mn, npno_mn)
            auto q_ov = qia_mn[q_mn]->clone(); // (nlmo_mn, npno_mn)

            // This performs a "T1-dressing" on Qvv, 
            // B^{Q_{mn}}_{f_{mn}c_{mn}} -= \widetilde{T}_{k_{mn}}^{f_{mn}} B^{Q_{mn}}_{k_{mn}c_{mn}}
            q_vv->subtract(linalg::doublet(T_n_ij_[mn], q_ov, true, false));

            for (int i_mn = 0; i_mn < nlmo_mn; ++i_mn) {
                for (int a_mn = 0; a_mn < npno_mn; ++a_mn) {
                    for (int f_mn = 0; f_mn < npno_mn; ++f_mn) {
                        for (int c_mn = 0; c_mn < npno_mn; ++c_mn) {
                            double val = 2.0 * q_vv->get(f_mn, c_mn) * q_ov->get(i_mn, a_mn) - q_vv->get(f_mn, a_mn) * q_ov->get(i_mn, c_mn);
                            F_fcia_temp->set(i_mn * npno_mn + a_mn, f_mn * npno_mn + c_mn, 
                                                F_fcia_temp->get(i_mn * npno_mn + a_mn, f_mn * npno_mn + c_mn) + val);
                        } // end c_mn
                    } // end f_mn
                } // end a_mn
            } // end i_mn
        }

        // Change the dimensions of F_fcia_temp to be (i_mn, a_mn * f_mn * c_mn)
        F_fcia_temp->reshape(nlmo_mn, npno_mn * npno_mn * npno_mn); // (i_mn, a_mn * f_mn * c_mn)

        // Package it up into F_fcia_hat_[mn][i_mn]
        for (int i_mn = 0; i_mn < nlmo_mn; ++i_mn) {
            int i = lmopair_to_lmos_[mn][i_mn];
            int ii = i_j_to_ij_[i][i];

            // This is the slice of F_fcia_temp corresponding to the current i_mn
            F_fcia_hat_[mn][i_mn] = submatrix_rows(*F_fcia_temp, std::vector<int>(1, i_mn)); // (1, a_mn * f_mn * c_mn)
            F_fcia_hat_[mn][i_mn]->reshape(npno_mn, npno_mn * npno_mn); // (a_mn, f_mn * c_mn)
            F_fcia_hat_[mn][i_mn] = linalg::doublet(S_PNO(ii, mn), F_fcia_hat_[mn][i_mn]); // (a_ii, f_mn * c_mn)
        } // end i_mn
    } // end mn
     */

    outfile->Printf("   Forming F_knia_hat...");
    time_start = std::time(nullptr);

    // Toth Eq. 34a
    F_knia_hat_.resize(n_lmo_pairs);
#pragma omp parallel for schedule(dynamic, 1)
    for (int kn = 0; kn < n_lmo_pairs; ++kn) {
        auto &[k, n] = ij_to_i_j_[kn];
        int nk = ij_to_ji_[kn];

        int naux_kn = lmopair_to_ribfs_[kn].size();
        int nlmo_kn = lmopair_to_lmos_[kn].size();
        int npno_kn = n_pno_[kn];

        F_knia_hat_[kn].resize(nlmo_kn);

        auto F_knia_temp = J_ijmb_[kn]->clone(); // (i_kn, a_kn)
        F_knia_temp->scale(2.0);
        F_knia_temp->subtract(K_mibj_[nk]); // (i_kn, a_kn)

        for (int i_kn = 0; i_kn < nlmo_kn; ++i_kn) {
            int i = lmopair_to_lmos_[kn][i_kn];
            int ii = i_j_to_ij_[i][i];

            F_knia_hat_[kn][i_kn] = submatrix_rows(*F_knia_temp, std::vector<int>(1, i_kn)); // (1, a_kn)
            F_knia_hat_[kn][i_kn]->reshape(npno_kn, 1); // (a_kn, 1)
            F_knia_hat_[kn][i_kn] = linalg::doublet(S_PNO(ii, kn), F_knia_hat_[kn][i_kn]); // (a_ii, 1)
        } // end i_kn
    } // end kn

    // Toth Eq. 34b
#pragma omp parallel for schedule(dynamic, 1)
    for (int n = 0; n < naocc; ++n) {
        int nn = i_j_to_ij_[n][n];

        int naux_nn = lmopair_to_ribfs_[nn].size(); // Number of auxiliary functions in domain of nn
        int nlmo_nn = lmopair_to_lmos_[nn].size(); // Number of LMOs in domain of pair nn
        int npno_nn = n_pno_[nn];

        auto qov_nn = QIA_PNO(nn); // naux_nn * (nlmo_nn, npno_nn)

        for (int q_nn = 0; q_nn < naux_nn; ++q_nn) {
            auto qov = qov_nn[q_nn]->clone(); // (nlmo_nn, npno_nn)
            auto qov_contracted = linalg::doublet(qov, T_ia_[n], false, false); // (nlmo_nn, npno_nn) * (npno_nn, 1) -> (nlmo_nn, 1)

            for (int k_nn = 0; k_nn < nlmo_nn; ++k_nn) {
                int k = lmopair_to_lmos_[nn][k_nn];
                int kn = i_j_to_ij_[k][n];
                for (int i_nn = 0; i_nn < nlmo_nn; ++i_nn) {
                    int i = lmopair_to_lmos_[nn][i_nn];
                    int ii = i_j_to_ij_[i][i];
                    int i_kn = lmopair_to_lmos_dense_[kn][i]; // Index of i in the domain of kn
                    if (i_kn == -1) continue; // If i is not in the domain of kn, skip

                    auto F_knia_temp = std::make_shared<Matrix>("F_knia_temp", npno_nn, 1);

                    for (int a_nn = 0; a_nn < npno_nn; ++a_nn) {
                        double val = 2.0 * qov_contracted->get(k_nn, 0) * qov->get(i_nn, a_nn) - 
                            qov_contracted->get(i_nn, 0) * qov->get(k_nn, a_nn);
                        F_knia_temp->set(a_nn, 0, F_knia_temp->get(a_nn, 0) + val);
                    }

                    F_knia_hat_[kn][i_kn]->add(linalg::doublet(S_PNO(ii, nn), F_knia_temp)); // (a_ii, 1)
                } // end i_nn
            } // end k_nn
        } // q_nn

    } // end n

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming L_ieab_bar_ and K_ijmb_bar_...");
    time_start = std::time(nullptr);

    L_ieab_bar_.resize(n_lmo_pairs);
    K_ijmb_bar_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int jj = i_j_to_ij_[j][j];

        int naux_ij = lmopair_to_ribfs_[ij].size();
        int nlmo_ij = lmopair_to_lmos_[ij].size();
        int npno_ij = n_pno_[ij];

        // Toth Eq. 73: S_{e_ij}^{e_jj} \overline{L}_{i e_ij}^{a_ij b_ij}
        // - \widetilde{T}_{l}^{e_{jj}} [S_{a_{il}}^{a_{ij}}L_{il}^{a_{il}b_{il}}]S_{b_{il}}^{b_{ij}}
        L_ieab_bar_[ij] = std::make_shared<Matrix>("L_ieab_bar", npno_ij, npno_ij * npno_ij); // (e_ij, a_ij * b_ij)

        for (int a_ij = 0; a_ij < npno_ij; ++a_ij) {
            for (int e_ij = 0; e_ij < npno_ij; ++e_ij) {
                for (int b_ij = 0; b_ij < npno_ij; ++b_ij) {
                    // L_{ie_{ij}}^{a_{ij} b_{ij}} = 2 (i a_{ij} | e_{ij} b_{ij}) - (i b_{ij} | e_{ij} a_{ij})
                    double val = 2.0 * K_ivvv_[ij]->get(a_ij, e_ij * n_pno_[ij] + b_ij) - K_ivvv_[ij]->get(b_ij, e_ij * n_pno_[ij] + a_ij);
                    L_ieab_bar_[ij]->set(e_ij, a_ij * npno_ij + b_ij, val);
                } // end b_ij
            } // end e_ij
        } // end a_ij

        L_ieab_bar_[ij] = linalg::doublet(S_PNO(jj, ij), L_ieab_bar_[ij]); // (e_ij, a_ij * b_ij) -> (e_jj, a_ij * b_ij)

        for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
            int l = lmopair_to_lmos_[ij][l_ij];
            int il = i_j_to_ij_[i][l], ll = i_j_to_ij_[l][l];

            auto T_l_j = linalg::doublet(S_PNO(jj, ll), T_ia_[l]); // (e_ll, 1) -> (e_jj, 1)
            auto L_il_to_ij = linalg::triplet(S_PNO(ij, il), L_iajb_[il], S_PNO(il, ij)); // (a_ij, b_ij)
            
            for (int e_jj = 0; e_jj < n_pno_[jj]; ++e_jj) {
                for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                    for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                        double val = (L_ieab_bar_[ij])->get(e_jj, a_ij * n_pno_[ij] + b_ij) - T_l_j->get(e_jj, 0) * L_il_to_ij->get(a_ij, b_ij);
                        L_ieab_bar_[ij]->set(e_jj, a_ij * n_pno_[ij] + b_ij, val);
                    } // end b_ij
                } // end a_ij
            } // end e_jj
        } // end l_ij

        // Toth Eq. 74: \overline{K}_{ij}^{m b_ij} = K_{ij}^{m b_ij}
        // + \widetilde{T}_m^{e_ij} K_{ij}^{e_ij b_ij}
        K_ijmb_bar_[ij] = K_mibj_[ij]->clone();
        K_ijmb_bar_[ij]->add(linalg::doublet(T_n_ij_[ij], K_iajb_[ij])); // (m, e) (e, b) -> (m, b)
    } // end ij

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming delta_imae_tilde ...");
    time_start = std::time(nullptr);

    // Toth Eq. 57: \widetilde{\delta}_{im}^{a_ii e_mm}. The following
    // contractions form its integral, singles-dressed, and doubles-dressed terms.
    delta_imae_tilde_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int im = 0; im < n_lmo_pairs; ++im) {
        auto &[i, m] = ij_to_i_j_[im];
        int ii = i_j_to_ij_[i][i], mm = i_j_to_ij_[m][m];
        int i_mm = lmopair_to_lmos_dense_[mm][i]; // Index of LMO (occupied) i in the domain of mm

        int nlmo_im = lmopair_to_lmos_[im].size();

        // The relavent density-fitted two-electron integrals needed
        auto Qov = QIA_PNO(mm); // (Q_{mm} | i_{mm} a_{mm})
        auto Qvv = QAB_PNO(mm); // (Q_{mm} | a_{mm} e_{mm})
        auto Qme = i_Qa_ij_[mm]; // \tilde(Q_{mm} | m e_{mm})
        auto Qmi = i_Qk_ij_[mm]; // \tilde(Q_{mm} | m i_{mm})

        // Toth Eq. 57, bare-integral contribution:
        // S(a_ii, a_mm) M_im^{a_mm e_mm}.
        
        auto M_imae_tilde = std::make_shared<Matrix>(n_pno_[mm], n_pno_[mm]);
        M_imae_tilde->zero();
        
        // M_{im}^{a_{mm}e_{mm}} = 2(ia_{mm}|me_{mm}) - (im|a_{mm}e_{mm})
        for (int q_mm = 0; q_mm < lmopair_to_ribfs_[mm].size(); ++q_mm) {
            for (int a_mm = 0; a_mm < n_pno_[mm]; ++a_mm) {
                for (int e_mm = 0; e_mm < n_pno_[mm]; ++e_mm) {
                    double val = 2.0 * Qov[q_mm]->get(i_mm, a_mm) * Qme->get(q_mm, e_mm)
                                    - Qmi->get(q_mm, i_mm) * Qvv[q_mm]->get(a_mm, e_mm);
                    M_imae_tilde->set(a_mm, e_mm, M_imae_tilde->get(a_mm, e_mm) + val);
                } // end e_mm
            } // end a_mm
        } // end q_mm

        M_imae_tilde = linalg::doublet(S_PNO(ii, mm), M_imae_tilde);

        // Toth Eq. 57, first singles-dressed contribution:
        // S(a_ii, a_mm) M_{i f_mm}^{a_mm e_mm} T_m^{f_mm}.

        // M_{if_{mm}}^{a_{mm}e_{mm}} = 2 (ia_{mm} | f_{mm} e_{mm}) - (if_{mm} | a_{mm} e_{mm}) (24c)
        auto M_ifae = std::make_shared<Matrix>(n_pno_[mm], n_pno_[mm] * n_pno_[mm]);

        for (int q_mm = 0; q_mm < lmopair_to_ribfs_[mm].size(); ++q_mm) {
            for (int f_mm = 0; f_mm < n_pno_[mm]; ++f_mm) {
                for (int a_mm = 0; a_mm < n_pno_[mm]; ++a_mm) {
                    for (int e_mm = 0; e_mm < n_pno_[mm]; ++e_mm) {
                        double val = 2.0 * Qov[q_mm]->get(i_mm, a_mm) * Qvv[q_mm]->get(f_mm, e_mm)
                                        - Qov[q_mm]->get(i_mm, f_mm) * Qvv[q_mm]->get(a_mm, e_mm);
                        M_ifae->set(f_mm, a_mm * n_pno_[mm] + e_mm, M_ifae->get(f_mm, a_mm * n_pno_[mm] + e_mm) + val);
                    } // end e_mm
                } // end a_mm
            } // end f_mm
        } // end q_mm
        M_ifae = linalg::doublet(T_ia_[m], M_ifae, true, false); // (f_mm, 1) x (f_mm, a_mm * e_mm) -> (1, a_mm * e_mm) 
        M_ifae->reshape(n_pno_[mm], n_pno_[mm]); // (1, a_mm * e_mm) -> (a_mm, e_mm)
        M_imae_tilde->add(linalg::doublet(S_PNO(ii, mm), M_ifae)); // (a_ii, a_mm) x (a_mm, e_mm) -> (a_ii, e_mm)

        // Toth Eq. 57, bilinear-singles contribution:
        // -\widetilde{T}_k^{e_mm} L_{ik}^{a_mm f_mm} S^{a_mm}_{a_ii} T_m^{f_mm}.

        // L_{ik}^{a_{mm}f_{mm}} = 2(ia_{mm} | kf_{mm})  - (if_{mm} | ka_{mm})
        for (int k_im = 0; k_im < nlmo_im; ++k_im) {
            int k = lmopair_to_lmos_[im][k_im];
            int mk = i_j_to_ij_[m][k], kk = i_j_to_ij_[k][k];
            int i_mk = lmopair_to_lmos_dense_[mk][i];
            int k_mm = lmopair_to_lmos_dense_[mm][k];

            // \widetilde{T}^{e_{mm}}_k
            auto T_k_m = submatrix_rows(*T_n_ij_[mm], std::vector<int>(1, k_mm)); // (1, e_mm)

            auto L_ikaf = std::make_shared<Matrix>(n_pno_[mm], n_pno_[mm]);
            L_ikaf->zero();

            for (int q_mm = 0; q_mm < lmopair_to_ribfs_[mm].size(); ++q_mm) {
                for (int a_mm = 0; a_mm < n_pno_[mm]; ++a_mm) {
                    for (int f_mm = 0; f_mm < n_pno_[mm]; ++f_mm) {
                        double val = 2.0 * Qov[q_mm]->get(i_mm, a_mm) * Qov[q_mm]->get(k_mm, f_mm)
                            - Qov[q_mm]->get(i_mm, f_mm) * Qov[q_mm]->get(k_mm, a_mm);
                        
                        L_ikaf->set(a_mm, f_mm, L_ikaf->get(a_mm, f_mm) + val);
                    } // end f_mm
                } // end a_mm
            } // end q_mm

            // S_{a_{ii}}^{a_{mm}} L^{a_{mm}f_{mm}}_{ik}  T^{f_{mm}}_m
            L_ikaf = linalg::triplet(S_PNO(ii, mm), L_ikaf, T_ia_[m]); // (a_ii, a_mm) x (a_mm, f_mm) x (f_mm, 1) -> (a_ii, 1)

            for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                for (int e_mm = 0; e_mm < n_pno_[mm]; ++e_mm) {
                    double val = L_ikaf->get(a_ii, 0) * T_k_m->get(0, e_mm);
                    M_imae_tilde->set(a_ii, e_mm, M_imae_tilde->get(a_ii, e_mm) - val);
                } // end e_mm
            } // end a_ii
        } // end k_im

        // Toth Eq. 57, doubles-dressed contribution:
        // S^{a_ii}_{a_ik} L_{ik}^{a_ik c_ik} S^{c_ik}_{c_km}
        // U_{km}^{c_km e_km} S^{e_km}_{e_mm}.

        // Compute delta_imae_tilde
        delta_imae_tilde_[im] = M_imae_tilde->clone();

        for (int k_im = 0; k_im < lmopair_to_lmos_[im].size(); ++k_im) {
            int k = lmopair_to_lmos_[im][k_im];
            int ik = i_j_to_ij_[i][k], km = i_j_to_ij_[k][m];
            
            auto delta_imae_tilde_temp = linalg::triplet(L_iajb_[ik], S_PNO(ik, km), Tt_iajb_[km]);
            delta_imae_tilde_[im]->add(linalg::triplet(S_PNO(ii, ik), delta_imae_tilde_temp, S_PNO(km, mm)));
        }
        
    } // end im

#pragma omp parallel for schedule(dynamic, 1)
    for (int mk = 0; mk < n_lmo_pairs; ++mk) {
        auto &[m, k] = ij_to_i_j_[mk];
        int mm = i_j_to_ij_[m][m], kk = i_j_to_ij_[k][k];
        int nlmo_mk = lmopair_to_lmos_[mk].size();

        // Toth Eq. 57, second singles-dressed contribution:
        // -\widetilde{T}_k^{e_mm} N_{mk}^{i a_mk} S(a_mk, a_ii).

        // T_k_m = \widetilde{T}_{k}^{e_{mm}}
        auto T_k_m = linalg::doublet(S_PNO(mm, kk), T_ia_[k]); // (e_{mm}, 1)

        auto N_mkia = J_ijmb_[mk]->clone(); // (i_{mk}, a_{mk})
        N_mkia->scale(2.0);
        N_mkia->subtract(K_mibj_[mk]); // (i_{mk}, a_{mk})

        // N_{mk}^{ia_{mk}} = 2 (mk | ia_{mk}) - (mi | ka_{mk})
        for (int i_mk = 0; i_mk < nlmo_mk; ++i_mk) {
            int i = lmopair_to_lmos_[mk][i_mk];
            int ii = i_j_to_ij_[i][i], im = i_j_to_ij_[i][m];
            
            auto N_mkia_slice = submatrix_rows(*N_mkia, std::vector<int>(1, i_mk))->transpose();
            N_mkia_slice = linalg::doublet(S_PNO(ii, mk), N_mkia_slice);

            for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                for (int e_mm = 0; e_mm < n_pno_[mm]; ++e_mm) {
#pragma omp atomic
                    (*delta_imae_tilde_[im])(a_ii, e_mm) -= N_mkia_slice->get(a_ii, 0) * T_k_m->get(e_mm, 0);
                } // end e_mm
            } // end a_ii
        } // end k_im
    }

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming F_vv_double_tilde...");
    time_start = std::time(nullptr);

    // Reuse the right-hand doubly transformed virtual Fock matrix (Jiang Eq. 85).
    F_vv_double_tilde_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        if (i > j) continue; // Only compute for upper triangle pairs, since F_vv_double_tilde_[ij] = F_vv_double_tilde_[ji]

        int nlmo_ij = lmopair_to_lmos_[ij].size();

        F_vv_double_tilde_[ij] = Fab_[ij]->clone();

        for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
            int k = lmopair_to_lmos_[ij][k_ij];
    
            for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
                int l = lmopair_to_lmos_[ij][l_ij];
                int kl = i_j_to_ij_[k][l];
                if (kl == -1) continue;

                auto F_vv_double_tilde_temp = linalg::doublet(Tt_iajb_[kl], K_iajb_[kl], false, true);
                F_vv_double_tilde_[ij]->subtract(linalg::triplet(S_PNO(ij, kl), F_vv_double_tilde_temp, S_PNO(kl, ij)));

            } // end l_ij
        } // end k_ij
    } // end ij

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming F_im_double_tilde...");
    time_start = std::time(nullptr);
    
    // Reuse the right-hand doubly transformed occupied Fock matrix (Jiang Eq. 86).
    F_im_double_tilde_ = Fkj_->clone();

#pragma omp parallel for schedule(dynamic, 1)
    for (int im = 0; im < n_lmo_pairs; ++im) {
        auto &[i, m] = ij_to_i_j_[im];

        int nlmo_im = lmopair_to_lmos_[im].size();
        
        for (int l_im = 0; l_im < nlmo_im; ++l_im) {
            int l = lmopair_to_lmos_[im][l_im];
            int li = i_j_to_ij_[l][i], lm = i_j_to_ij_[l][m];

            auto U_lm_to_li = linalg::triplet(S_PNO(li, lm), Tt_iajb_[lm], S_PNO(li, lm), false, false, true);
            double val = F_im_double_tilde_->get(i, m) + U_lm_to_li->vector_dot(K_iajb_[li]);
            F_im_double_tilde_->set(i, m, val);
        } // end l_im
    } // end im

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);

    outfile->Printf("   Forming gamma_double_tilde and delta_double_tilde...");
    time_start = std::time(nullptr);

    // Higher-order doubles-residual intermediates (Toth Eqs. 103 and 105).
    gamma_double_tilde_.resize(n_lmo_pairs);
    delta_double_tilde_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        int nlmo_ij = lmopair_to_lmos_[ij].size();

        gamma_double_tilde_[ij].resize(nlmo_ij);
        delta_double_tilde_[ij].resize(nlmo_ij);

        for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
            int n = lmopair_to_lmos_[ij][n_ij];
            int in = i_j_to_ij_[i][n], jn = i_j_to_ij_[j][n], nj = i_j_to_ij_[n][j];
            int i_nj = lmopair_to_lmos_dense_[nj][i];

            gamma_double_tilde_[ij][n_ij] = std::make_shared<Matrix>(n_pno_[jn], n_pno_[ij]);
            gamma_double_tilde_[ij][n_ij]->zero();

            for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
                int k = lmopair_to_lmos_[ij][k_ij];
                int kn = i_j_to_ij_[k][n], ik = i_j_to_ij_[i][k], kj = i_j_to_ij_[k][j];
                if (kn == -1) continue;

                auto T_kn = linalg::triplet(S_PNO(jn, kn), T_iajb_[kn], S_PNO(kn, ik));
                auto K_ik = linalg::doublet(K_iajb_[ik], S_PNO(ik, ij));

                gamma_double_tilde_[ij][n_ij]->add(linalg::doublet(T_kn, K_ik));
                
            } // end k_ij

        } // end n_ij

        for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
            int n = lmopair_to_lmos_[ij][n_ij];
            int nj = i_j_to_ij_[n][j], in = i_j_to_ij_[i][n], ni = i_j_to_ij_[n][i];
            int j_ni = lmopair_to_lmos_dense_[ni][j];

            delta_double_tilde_[ij][n_ij] = std::make_shared<Matrix>(n_pno_[ni], n_pno_[ij]);
            delta_double_tilde_[ij][n_ij]->zero();
            
            for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
                int k = lmopair_to_lmos_[ij][k_ij];
                int nk = i_j_to_ij_[n][k], ik = i_j_to_ij_[i][k], kj = i_j_to_ij_[k][j];
                if (nk == -1) continue;

                auto U_nk = linalg::triplet(S_PNO(ni, nk), Tt_iajb_[nk], S_PNO(nk, kj));
                auto L_kj = linalg::doublet(L_iajb_[kj], S_PNO(kj, ij));

                delta_double_tilde_[ij][n_ij]->add(linalg::doublet(U_nk, L_kj));
            } // end k_ij
        }
    } // end ij

    time_stop = std::time(nullptr);
    outfile->Printf("   %3d seconds\n\n", (int) time_stop - (int) time_start);
}

void DLPNOCCSD::compute_L_ia(std::vector<SharedMatrix>& L_ia, std::vector<std::vector<SharedMatrix>> &L_ia_buffer) {

    // Assemble l_i^{a_ii} = dL_CCSD/dt_i^{a_ii}, the Lambda-singles
    // stationarity residual. The leading transformed-Fock/delta terms are
    // Toth Eq. 58; the remaining contractions are Toth Eqs. 67--72.

    timer_on("DLPNO-CCSD Lambda : Compute L1");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();

    // Thread and OMP Parallel Info
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    // Zero out buffers
    for (int thread = 0; thread < nthreads; ++thread) {
        for (int i = 0; i < naocc; ++i) {
            L_ia_buffer[thread][i]->zero();
        }
    }

    // \lambda_i^{e_ii}\widetilde{\widetilde{F}}_{e_ii a_ii} (Toth Eq. 58c)
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < naocc; ++i) {
        int ii = i_j_to_ij_[i][i];

        L_ia[i] = linalg::doublet(F_vv_double_tilde_[ii], lambda_ia_[i], true, false); // (e, a) x (e, 1) -> (a, 1)
    } // end for

    // + 2.0 * (S^{a_{ii}}_{a_{in}} L_{in}^{a_{in}f_{in}} S^{f_{in}}_{f_{nn}}) T_{n}^{f_{nn}}
#pragma omp parallel for schedule(dynamic, 1)
    for (int in = 0; in < n_lmo_pairs; ++in) {
        auto &[i, n] = ij_to_i_j_[in];
        int ii = i_j_to_ij_[i][i], nn = i_j_to_ij_[n][n];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        auto L1_temp = linalg::triplet(L_iajb_[in], S_PNO(in, nn), T_ia_[n]);
        L1_temp = linalg::doublet(S_PNO(ii, in), L1_temp);
        L1_temp->scale(2.0);

        L_ia_buffer[thread][i]->add(L1_temp);
    } // end in

#pragma omp parallel for schedule(dynamic, 1)
    for (int im = 0; im < n_lmo_pairs; ++im) {
        auto &[i, m] = ij_to_i_j_[im];
        int ii = i_j_to_ij_[i][i], mm = i_j_to_ij_[m][m], mi = i_j_to_ij_[m][i];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // \widetilde{\delta}_{im}^{a_ii e_mm}\lambda_m^{e_mm} (Toth Eq. 58a)
        L_ia_buffer[thread][i]->add(linalg::doublet(delta_imae_tilde_[im], lambda_ia_[m]));

        // -\widetilde{\widetilde{F}}_{im}\lambda_m^{a_mm}S_{a_mm}^{a_ii} (Toth Eq. 58b)
        auto lambda_m_to_i = linalg::doublet(S_PNO(ii, mm), lambda_ia_[m]);
        lambda_m_to_i->scale(F_im_double_tilde_->get(i, m));
        L_ia_buffer[thread][i]->subtract(lambda_m_to_i);
        
        // l_i^{a_ii} += \widetilde{\widetilde{K}}_{m a_ii}^{e_mi f_mi}
        //                  \widetilde{\lambda}_{mi}^{e_mi f_mi} (Toth Eq. 67a)
        auto lambda_mi_slice = lambda_iajb_[mi]->clone();
        lambda_mi_slice->reshape(n_pno_[mi] * n_pno_[mi], 1);
        L_ia_buffer[thread][i]->add(linalg::doublet(K_maef_dt_[mi], lambda_mi_slice)); // (a_{ii}, e_{mi} * f_{mi}) (e_{mi} * f_{mi}, 1)
    } // end im
    
#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        auto &[m, n] = ij_to_i_j_[mn];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        int nlmo_mn = lmopair_to_lmos_[mn].size();
	    int naux_mn = lmopair_to_ribfs_[mn].size();

        auto qia_mn = QIA_PNO(mn); // naux_mn * (nlmo_mn, npno_mn)
        auto qab_mn = QAB_PNO(mn); // naux_mn * (npno_mn, npno_mn)

        // l_i^{a_ii} -= \widetilde{\widetilde{K}}_{e_mn i}^{mn}
        //                  \widetilde{\lambda}_{mn}^{e_mn a_mn}S_{a_mn}^{a_ii} (Toth Eq. 67b)
        auto L1_temp = linalg::doublet(K_eimn_dt_[mn], lambda_iajb_[mn], true, false); // (e_{mn}, i_{mn}) x (e_{mn}, a_{mn}) -> (i_{mn}, a_{mn})

        for (int i_mn = 0; i_mn < lmopair_to_lmos_[mn].size(); ++i_mn) {
            int i = lmopair_to_lmos_[mn][i_mn];
            int ii = i_j_to_ij_[i][i];
            auto L1_temp_slice = submatrix_rows(*L1_temp, std::vector<int>(1, i_mn));
            L_ia_buffer[thread][i]->subtract(linalg::doublet(S_PNO(ii, mn), L1_temp_slice, false, true)); // (a_{ii}, a_{mn}) x (1, a_{mn}) -> (a_{ii}, 1)
        }

        // rho^VV contribution to l_i^{a_ii} (Toth Eq. 68a).
        for (int q_mn = 0; q_mn < naux_mn; ++q_mn) {
            auto q_vv = qab_mn[q_mn]->clone(); // (npno_mn, npno_mn)
            auto q_ov = qia_mn[q_mn]->clone(); // (nlmo_mn, npno_mn)

            // This performs a "T1-dressing" on Qvv, 
            // B^{Q_{mn}}_{f_{mn}c_{mn}} -= \widetilde{T}_{k_{mn}}^{f_{mn}} B^{Q_{mn}}_{k_{mn}c_{mn}}
            q_vv->subtract(linalg::doublet(T_n_ij_[mn], q_ov, true, false));

            auto Gvv_temp = q_ov->clone();
            Gvv_temp->scale(2.0 * rho_vv_[mn]->vector_dot(q_vv));
            Gvv_temp->subtract(linalg::triplet(q_ov, rho_vv_[mn], q_vv, false, true, false)); // (i, c) (f, c) (f, a) -> (i, a)

            for (int i_mn = 0; i_mn < lmopair_to_lmos_[mn].size(); ++i_mn) {
                int i = lmopair_to_lmos_[mn][i_mn];
                int ii = i_j_to_ij_[i][i];

                auto Gvv_slice = submatrix_rows(*Gvv_temp, std::vector<int>(1, i_mn)); // (1, a)
                L_ia_buffer[thread][i]->add(linalg::doublet(S_PNO(ii, mn), Gvv_slice, false, true));
            } // end i_mn
        } // end q_mn

        // -rho^OO_nm \hat{F}_{mn}^{i a_ii} contribution (Toth Eq. 68b).
        for (int i_mn = 0; i_mn < lmopair_to_lmos_[mn].size(); ++i_mn) {
            int i = lmopair_to_lmos_[mn][i_mn];

            /*
            auto Gvv_temp = linalg::doublet(F_fcia_hat_[mn][i_mn], Gvv_slice, false, false); // (a_ii, f_mn * c_mn) (f_mn * c_mn, 1) -> (a_ii, 1)
            L_ia_buffer[thread][i]->add(Gvv_temp);
            */

            auto F_mnia_slice = F_knia_hat_[mn][i_mn]->clone(); // (a_ii, 1)
            F_mnia_slice->scale(rho_oo_->get(n, m)); // Needs to be (n, m) not mn
            L_ia_buffer[thread][i]->subtract(F_mnia_slice);
        }
    } // end mn

    /* \textcolor{blue}{\begin{equation}
        l^{a_{ii}}_{i} \mathrel{+}= [S^{a_{ii}}_{a_{km}}S^{a_{km}}_{a_{mn}}\overline{\lambda}^{a_{mn}f_{mn}}_{mn}S^{f_{kn}}_{f_{mn}}]T^{f_{kn}c_{kn}}_{kn}S^{c_{km}}_{c_{kn}}\overline{J}^{ic_{km}}_{km} (Toth Eq. 69)
        \end{equation}}
    */
#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        auto &[m, n] = ij_to_i_j_[mn];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int k_mn = 0; k_mn < lmopair_to_lmos_[mn].size(); ++k_mn) {
            int k = lmopair_to_lmos_[mn][k_mn];
            int km = i_j_to_ij_[k][m], kn = i_j_to_ij_[k][n];
            auto gus = linalg::doublet(lambda_iajb_bar_[mn], S_PNO(mn, kn));
            auto L1_temp = linalg::triplet(gus, T_iajb_[kn], S_PNO(kn, km));

            // Contract with the \overline{J} intermediate from Toth Eq. 63.
            L1_temp = linalg::doublet(L1_temp, J_kmic_bar_[km], false, true); // (a, c) * (i, c)
            
            for (int i_km = 0; i_km < lmopair_to_lmos_[km].size(); ++i_km) {
                int i = lmopair_to_lmos_[km][i_km];
                int ii = i_j_to_ij_[i][i];
                auto L1_temp_slice = submatrix_cols(*L1_temp, std::vector<int>(1, i_km));
                L_ia_buffer[thread][i]->add(linalg::doublet(S_PNO(ii, mn), L1_temp_slice));
            } // end for

        } // end for
    } // end for

    // l_i^{a_ii} -= (S \overline{\lambda}_{in} S) T_kn S \overline{J}_{k a_ii}
    // (Toth Eq. 70; \overline{J} is defined in Toth Eq. 64).
#pragma omp parallel for schedule(dynamic, 1) 
    for (int in = 0; in < n_lmo_pairs; ++in) {
        auto &[i, n] = ij_to_i_j_[in];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int k_in = 0; k_in < lmopair_to_lmos_[in].size(); ++k_in) {
            int k = lmopair_to_lmos_[in][k_in];
            int kn = i_j_to_ij_[k][n], in = i_j_to_ij_[i][n], ki = i_j_to_ij_[k][i];
            auto L1_temp = linalg::triplet(S_PNO(ki, in), lambda_iajb_bar_[in], S_PNO(in, kn));
            L1_temp = linalg::triplet(L1_temp, T_iajb_[kn], S_PNO(kn, ki));

            L1_temp->reshape(n_pno_[ki] * n_pno_[ki], 1);
            L1_temp = linalg::doublet(J_kaec_bar_[ki], L1_temp); // (a, e * c) (e * c, 1) => (a, 1)
            L_ia_buffer[thread][i]->subtract(L1_temp);
        } // end k_in
    } // end in
        
    /*
    \textcolor{blue}{\begin{equation}
        l^{a_{ii}}_i \mathrel{+}= \frac{1}{2} \widetilde{\lambda}_{in}^{e_{in}f_{in}} [S^{e_{in}}_{e_{ik}}(S^{c_{nk}}_{c_{ik}}u^{f_{nk}c_{nk}}_{nk}S^{f_{in}}_{f_{nk}})]\overline{M}^{c_{ki}e_{ki}}_{ka_{ii}} (Toth Eq. 71)
    \end{equation}}
    */
#pragma omp parallel for schedule(dynamic, 1)
    for (int in = 0; in < n_lmo_pairs; ++in) {
        auto &[i, n] = ij_to_i_j_[in];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int k_in = 0; k_in < lmopair_to_lmos_[in].size(); ++k_in) {
            int k = lmopair_to_lmos_[in][k_in];
            int nk = i_j_to_ij_[n][k], in = i_j_to_ij_[i][n], ki = i_j_to_ij_[k][i];
            auto L1_temp = linalg::triplet(S_PNO(ki, in), lambda_iajb_[in], S_PNO(in, nk)); // (e_{ki}, e_{in}) (e_{in}, f_{in}) (f_{in}, f_{nk}) -> (e_{ki}, f_{nk})
            L1_temp = linalg::triplet(L1_temp, Tt_iajb_[nk], S_PNO(nk, ki));

            L1_temp->transpose_this(); // (e, c) -> (c, e)
            L1_temp->reshape(n_pno_[ki] * n_pno_[ki], 1);
            L1_temp = linalg::doublet(M_kace_bar_[ki], L1_temp); // (a, c * e) (c * e, 1) => (a, 1) (steak sauce)
            L1_temp->scale(0.5);
            L_ia_buffer[thread][i]->add(L1_temp);
        } // end k_in
    } // end in

    /*
        \textcolor{blue}{\begin{equation}
            l^{a_{ii}}_i \mathrel{-}= \frac{1}{2}[S^{a_{ii}}_{a_{mk}}(S^{a_{mk}}_{a_{mn}}\widetilde{\lambda}^{a_{mn}f_{mn}}_{mn}S^{f_{mn}}_{f_{nk}})]u^{f_{nk}c_{nk}}_{nk}S^{c_{nk}}_{c_{mk}}\overline{M}^{ic_{mk}}_{mk} (Toth Eq. 72)
        \end{equation}}
    */
#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        auto &[m, n] = ij_to_i_j_[mn];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int k_mn = 0; k_mn < lmopair_to_lmos_[mn].size(); ++k_mn) {
            int k = lmopair_to_lmos_[mn][k_mn];
            int nk = i_j_to_ij_[n][k], mk = i_j_to_ij_[m][k];
            auto L1_temp = linalg::doublet(lambda_iajb_[mn], S_PNO(mn, nk)); // (a, f)
            L1_temp = linalg::triplet(L1_temp, Tt_iajb_[nk], S_PNO(nk, mk)); // (a, f) (f, c) -> (a, c)

            for (int i_mk = 0; i_mk < lmopair_to_lmos_[mk].size(); ++i_mk) {
                int i = lmopair_to_lmos_[mk][i_mk];
                int ii = i_j_to_ij_[i][i];
                auto M_temp = submatrix_rows(*M_mkic_bar_[mk], std::vector<int>(1, i_mk));
                M_temp = linalg::triplet(S_PNO(ii, mn), L1_temp, M_temp, false, false, true);
                M_temp->scale(-0.5);
                L_ia_buffer[thread][i]->add(M_temp);
            } // end i_mk
        } // end k_mn
    } // end mn

    // Add L_ia_buffers to L_ia
    for (int i = 0; i < naocc; ++i) {
        for (int thread = 0; thread < nthreads; ++thread) {
            L_ia[i]->add(L_ia_buffer[thread][i]);
        } // end thread
    } // end int i

    timer_off("DLPNO-CCSD Lambda : Compute L1");
}

std::vector<SharedMatrix> DLPNOCCSD::compute_alpha_ijkl() {
    timer_on("DLPNO-CCSD Lambda : alpha ijkl");

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    std::vector<SharedMatrix> alpha_ijkl(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];

        if (i > j) continue;

        int nlmo_ij = lmopair_to_lmos_[ij].size();
        alpha_ijkl[ij] = std::make_shared<Matrix>("alpha_ijkl", nlmo_ij, nlmo_ij);

        // alpha_{ij}^{kl} = \lambda_{ij}^{e_ij f_ij}
        // (S_{e_kl}^{e_ij} T_{kl}^{e_kl f_kl} S_{f_kl}^{f_ij}) (Toth Eq. 81)
        for (int k_ij = 0; k_ij < nlmo_ij; ++k_ij) {
            int k = lmopair_to_lmos_[ij][k_ij];
            for (int l_ij = 0; l_ij < nlmo_ij; ++l_ij) {
                int l = lmopair_to_lmos_[ij][l_ij];
                int kl = i_j_to_ij_[k][l];
                if (kl == -1) continue;
                auto T_kl_to_ij = linalg::triplet(S_PNO(ij, kl), T_iajb_[kl], S_PNO(kl, ij));
                double val = T_kl_to_ij->vector_dot(lambda_iajb_[ij]);
                alpha_ijkl[ij]->set(k_ij, l_ij, val);
            } // end l_ij
        } // end k_ij
    }

    timer_off("DLPNO-CCSD Lambda : alpha ijkl");

    return alpha_ijkl;
}

void DLPNOCCSD::compute_L_iajb(std::vector<SharedMatrix>& L_iajb, std::vector<SharedMatrix>& Ln_iajb) {

    // Assemble the ordered-pair Lambda-doubles stationarity residual. Ln_ij
    // holds the nonsymmetric contribution in Toth Eq. 75; the final loop adds
    // Ln_ij + (Ln_ji)^T. The expensive pair-symmetric contractions below are
    // the grouped contributions in Toth Eqs. 86--91.

    timer_on("DLPNO-CCSD Lambda : Compute L2");

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    // Thread and OMP Parallel info
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    // Zero out residuals
#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        L_iajb[ij]->zero();
        Ln_iajb[ij]->zero();
    } // end for

    auto alpha_ijkl = compute_alpha_ijkl();

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij], ii = i_j_to_ij_[i][i];

        const int npao_ij = lmopair_to_paos_[ij].size();
        const int naux_ij = lmopair_to_ribfs_[ij].size();
        const int nlmo_ij = lmopair_to_lmos_[ij].size();
        const int npno_ij = n_pno_[ij];

        Ln_iajb[ij] = L_iajb_[ij]->clone();

        // Pair-symmetric terms need be formed only for one orientation; their
        // transpose is accumulated into the reverse ordered pair at each step.
        if (i <= j) {
            // Necessary three-center integrals
            auto qma_ij = QIA_PNO(ij); // naux_ij * (nlmo_ij, npno_ij)
            auto qab_ij = QAB_PNO(ij); // naux_ij * (npno_ij, npno_ij)

            // Four-index/DF contractions collected in Toth Eq. 86.
            for (int q_ij = 0; q_ij < naux_ij; ++q_ij) {
                // Form the T1-dressed Qab intermediate on the fly; it is used once.
                // \widetilde{B}^{Q}_{ab} = B^{Q}_{ab} - t_{k}^{a} B^{Q}_{kb} (Jiang Eq. 93)
                auto Qab_t1 = qab_ij[q_ij]->clone(); // (a, b)
                Qab_t1->subtract(linalg::doublet(T_n_ij_[ij], qma_ij[q_ij], true, false)); // (k, a) (k, b) -> (a, b)

                auto L_temp = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
                L_temp->zero();
                // l_ij^{a_ij b_ij} += \widetilde{\lambda}_ij^{e_ij f_ij}
                //     \widetilde{B}^{Q_ij}_{e_ij a_ij}\widetilde{B}^{Q_ij}_{f_ij b_ij} (Toth Eq. 86a).
                L_temp->add(linalg::triplet(Qab_t1, lambda_iajb_[ij], Qab_t1, true, false, false)); // (e, a) (e, f) (f, b)
                // l_ij^{a_ij b_ij} += B^{Q_ij}_{k_ij a_ij}B^{Q_ij}_{l_ij b_ij}
                //     alpha_ij^{k_ij l_ij} (Toth Eq. 86b; alpha is Eq. 81).
                L_temp->add(linalg::triplet(qma_ij[q_ij], alpha_ijkl[ij], qma_ij[q_ij], true, false, false)); // (k, a) (k, l) (l, b)

                L_iajb[ij]->add(L_temp);
                if (i != j) L_iajb[ji]->add(L_temp->transpose());
            } // end q_ij

            // Projected Lambda-doubles/beta contraction (Toth Eq. 87;
            // beta is the right-hand intermediate of Jiang Eq. 82).
            for (int m_ij = 0; m_ij < nlmo_ij; ++m_ij) {
                int m = lmopair_to_lmos_[ij][m_ij];
                for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
                    int n = lmopair_to_lmos_[ij][n_ij];
                    int mn = i_j_to_ij_[m][n];
                    if (mn == -1) continue;
                    int i_mn = lmopair_to_lmos_dense_[mn][i], j_mn = lmopair_to_lmos_dense_[mn][j];

                    auto lambda_mn_to_ij = linalg::triplet(S_PNO(ij, mn), lambda_iajb_[mn], S_PNO(mn, ij));
                    lambda_mn_to_ij->scale(beta_[mn]->get(i_mn, j_mn));
                    L_iajb[ij]->add(lambda_mn_to_ij);
                    if (i != j) L_iajb[ji]->add(lambda_mn_to_ij->transpose());
                } // end n_ij
            } // end m_ij

            // Virtual-Fock and rho^VV contributions (Toth Eq. 90; rho^VV is
            // defined in Eq. 54 and projected from every overlapping pair).
            auto E_temp = linalg::doublet(lambda_iajb_[ij], F_vv_double_tilde_[ij], false, false); // (a, f) (f, b) -> (a, b)
            E_temp->add(linalg::doublet(F_vv_double_tilde_[ij], lambda_iajb_[ij], true, false)); // (f, a) (f, b) -> (a, b)

            auto rho_vv_cont = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
            rho_vv_cont->zero();
            
            for (int m_ij = 0; m_ij < nlmo_ij; ++m_ij) {
                int m = lmopair_to_lmos_[ij][m_ij];
                for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
                    int n = lmopair_to_lmos_[ij][n_ij];
                    int mn = i_j_to_ij_[m][n];
                    if (mn == -1) continue;

                    rho_vv_cont->add(linalg::triplet(S_PNO(ij, mn), rho_vv_[mn], S_PNO(mn, ij)));
                } // end n_ij
            } // end m_ij
            E_temp->subtract(linalg::doublet(rho_vv_cont, L_iajb_[ij])); // (a, c) (c, b) -> (a, b)
            E_temp->subtract(linalg::doublet(L_iajb_[ij], rho_vv_cont, false, true)); // (a, c) (b, c) -> (a, b)

            L_iajb[ij]->add(E_temp);
            if (i != j) L_iajb[ji]->add(E_temp->transpose());
        } // end i <= j

        // Lambda-singles contribution to the nonsymmetric residual in Toth
        // Eq. 75, using \overline{L} from Eq. 73.
        auto L_temp = linalg::doublet(lambda_ia_[j], L_ieab_bar_[ij], true, false); // (e, 1) (e, a * b) -> (1, a * b)
        L_temp->reshape(n_pno_[ij], n_pno_[ij]);
        Ln_iajb[ij]->add(L_temp);

        // Lambda-singles/\overline{K} contribution to Toth Eq. 75;
        // \overline{K}_ij^{m b_ij} is defined in Eq. 74.
        for (int m_ij = 0; m_ij < nlmo_ij; ++m_ij) {
            int m = lmopair_to_lmos_[ij][m_ij];
            int mm = i_j_to_ij_[m][m];

            auto lambda_m_ij = linalg::doublet(S_PNO(ij, mm), lambda_ia_[m]);
            for (int a_ij = 0; a_ij < npno_ij; ++a_ij) {
                for (int b_ij = 0; b_ij < npno_ij; ++b_ij) {
                    double val = Ln_iajb[ij]->get(a_ij, b_ij) - 2.0 * lambda_m_ij->get(a_ij, 0) * K_ijmb_bar_[ij]->get(m_ij, b_ij)
                                    + lambda_m_ij->get(b_ij, 0) * K_ijmb_bar_[ij]->get(m_ij, a_ij);
                    Ln_iajb[ij]->set(a_ij, b_ij, val);
                } // end b_ij
            } // end a_ij
        } // end m_ij

        // Lambda-singles/Fock contribution to the nonsymmetric residual in
        // Toth Eq. 75; the explicit 2-P_ab action is applied elementwise.
        auto lambda_i = linalg::doublet(S_PNO(ij, ii), lambda_ia_[i]);
        for (int a_ij = 0; a_ij < npno_ij; ++a_ij) {
            for (int b_ij = 0; b_ij < npno_ij; ++b_ij) {
                double val = Ln_iajb[ij]->get(a_ij, b_ij) + 2.0 * lambda_i->get(a_ij, 0) * Fkc_[ji]->get(b_ij, 0)
                                - lambda_i->get(b_ij, 0) * Fkc_[ji]->get(a_ij, 0);
                Ln_iajb[ij]->set(a_ij, b_ij, val);
            } // end b_ij
        } // end a_ij

        for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
            int n = lmopair_to_lmos_[ij][n_ij];
            int in = i_j_to_ij_[i][n], jn = i_j_to_ij_[j][n], nj = i_j_to_ij_[n][j];
            int i_nj = lmopair_to_lmos_dense_[nj][i];

            // Barred-Lambda/gamma contraction (Toth Eq. 88a).
            auto lambda_bar_temp = linalg::doublet(S_PNO(ij, jn), lambda_iajb_bar_[jn]);
            auto gamma_temp = linalg::triplet(S_PNO(nj, in), gamma_[in], S_PNO(in, ij));
            // Add the non-projected integral contribution before contracting
            // with the barred Lambda doubles.
            gamma_temp->add(J_ikac_non_proj_[nj][i_nj]);
            gamma_temp = linalg::doublet(lambda_bar_temp, gamma_temp);
            Ln_iajb[ij]->subtract(gamma_temp);

            // Higher-order correction to the same contraction (Toth Eq. 88b),
            // preassembled in gamma_double_tilde_ (Eq. 103).
            gamma_temp = gamma_double_tilde_[ij][n_ij]->clone();
            gamma_temp->scale(0.5);
            Ln_iajb[ij]->add(linalg::triplet(S_PNO(ij, jn), lambda_iajb_bar_[jn], gamma_temp));
        } // end n_ij
        
        for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
            int n = lmopair_to_lmos_[ij][n_ij];
            int nj = i_j_to_ij_[n][j], in = i_j_to_ij_[i][n], ni = i_j_to_ij_[n][i];
            int j_ni = lmopair_to_lmos_dense_[ni][j];

            // Lambda-doubles/delta contribution (Toth Eq. 89a); the following
            // additions explicitly apply its 2-P_ab spin adaptation.
            auto delta_temp = linalg::doublet(S_PNO(ij, in), lambda_iajb_[in]);
            /// [ij][k_ij] => (i a_{ij} | k c_{kj}) 
            /// [ij][k_ij] => (i k | a_{ij} c_{kj})

            /// [ni][j_ni] => (n a_{ni} | j c_{ji})

            /// (n f_{ni} | j b_{ij})
            // Clone because the following spin adaptation must not mutate the
            // stored non-projected integral block.
            auto M_iakc_temp = K_iakc_non_proj_[ni][j_ni]->clone();
            M_iakc_temp->scale(2.0);
            /// (n j | f_{ni} b_{ij})
            M_iakc_temp->subtract(J_ikac_non_proj_[ni][j_ni]);
            M_iakc_temp->add(linalg::triplet(S_PNO(ni, nj), delta_[nj], S_PNO(nj, ij))); // (f_{ni}, b_{ij})
            delta_temp = linalg::doublet(delta_temp, M_iakc_temp);
            
            Ln_iajb[ij]->add(delta_temp);
            delta_temp->scale(-0.5);
            Ln_iajb[ij]->add(delta_temp->transpose());

            // Higher-order partner, Toth Eq. 89b, using the Eq. 105
            // delta_double_tilde_ intermediate.
            delta_temp = delta_double_tilde_[ij][n_ij]->clone();
            delta_temp->scale(0.5);
            
            delta_temp = linalg::triplet(S_PNO(ij, in), lambda_iajb_[in], delta_temp);
            Ln_iajb[ij]->add(delta_temp);
            delta_temp->scale(-0.5);
            Ln_iajb[ij]->add(delta_temp->transpose());
        }

        // Occupied-Fock and rho^OO contributions (Toth Eq. 91; rho^OO is
        // defined in Eq. 53). Both source-pair tensors are projected to ij.
        for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
            int n = lmopair_to_lmos_[ij][n_ij];
            int in = i_j_to_ij_[i][n];
            auto lambda_in_to_ij = linalg::triplet(S_PNO(ij, in), lambda_iajb_[in], S_PNO(in, ij));
            lambda_in_to_ij->scale(F_im_double_tilde_->get(j, n));
            Ln_iajb[ij]->subtract(lambda_in_to_ij);

            auto L_in_to_ij = linalg::triplet(S_PNO(ij, in), L_iajb_[in], S_PNO(in, ij));
            L_in_to_ij->scale(rho_oo_->get(j, n));
            Ln_iajb[ij]->subtract(L_in_to_ij);
        } // end n_ij
    } // end ij

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];
        
        L_iajb[ij]->add(Ln_iajb[ij]);
        L_iajb[ij]->add(Ln_iajb[ji]->transpose());
    }

    timer_off("DLPNO-CCSD Lambda : Compute L2");
}

void DLPNOCCSD::reset_lambda_state() {
    gradient_lambda_fixed_.reset();
    lambda_solved_ = false;

    lambda_ia_.clear();
    lambda_iajb_.clear();
    lambda_iajb_bar_.clear();
    delta_imae_tilde_.clear();
    F_im_double_tilde_.reset();
    F_vv_double_tilde_.clear();
    beta_.clear();
    gamma_.clear();
    delta_.clear();
    gamma_double_tilde_.clear();
    delta_double_tilde_.clear();
    rho_oo_.reset();
    rho_vv_.clear();
    F_fcia_hat_.clear();
    F_knia_hat_.clear();
    K_maef_dt_.clear();
    K_eimn_dt_.clear();
    M_kace_bar_.clear();
    M_mkic_bar_.clear();
    J_kmic_bar_.clear();
    J_kaec_bar_.clear();
    L_ieab_bar_.clear();
    K_ijmb_bar_.clear();
    Doo_.reset();
    Dov_.clear();
    Dvv_singles_.clear();
    Dvv_pair_.clear();

    if (has_array_variable("DLPNO-CCSD AO OPDM")) {
        del_array_variable("DLPNO-CCSD AO OPDM");
    }
}

void DLPNOCCSD::solve_lambda(bool include_singles) {

    // Solve dL_CCSD/dT = 0 for the left amplitudes after the right-hand CCSD
    // amplitudes have converged. For converged Brueckner orbitals, Lambda1 is
    // omitted because the orbital stationarity condition replaces the singles
    // equation; canonical/reference-orbital and property calculations include it.

    reset_lambda_state();
    estimate_lambda_memory();
    compute_lambda_intermediates();

    int n_lmo_pairs = ij_to_i_j_.size();
    int naocc = nalpha_ - nfrzc();

    // Thread and OMP Parallel info
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    outfile->Printf("\n  ==> Lambda DLPNO-CCSD <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    outfile->Printf("                            Corr. Energy    Delta E     Max L1     Max L2     Time (s)\n");

    // => Initialize Residuals and Amplitudes <= //

    std::vector<SharedMatrix> L_ia(naocc);
    std::vector<SharedMatrix> L_ia_prev(naocc);
    std::vector<SharedMatrix> Ln_iajb(n_lmo_pairs);
    std::vector<SharedMatrix> L_iajb(n_lmo_pairs);
    std::vector<SharedMatrix> L_iajb_prev(n_lmo_pairs);

    // => Initialize Singles and Doubles Residuals and Amplitudes <= //
    lambda_ia_.resize(naocc);
    lambda_iajb_.resize(n_lmo_pairs);
    lambda_iajb_bar_.resize(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < naocc; ++i) {
        int ii = i_j_to_ij_[i][i];
        lambda_ia_[i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        L_ia[i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        L_ia_prev[i] = std::make_shared<Matrix>(n_pno_[ii], 1);
    }

    // => Initialize Doubles Residuals and Amplitudes <= //
    
#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        lambda_iajb_[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
        lambda_iajb_bar_[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
        L_iajb[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
        Ln_iajb[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
        L_iajb_prev[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
    }

    // => Thread buffers <= //

    std::vector<std::vector<SharedMatrix>> L_ia_buffer(nthreads);
    for (int thread = 0; thread < nthreads; ++thread) {
        L_ia_buffer[thread].resize(naocc);
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            L_ia_buffer[thread][i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        } // end i
    } // end thread

    int iteration = 1, max_iteration = options_.get_int("DLPNO_MAXITER");
    double damping = options_.get_double("DLPNO_LAMBDA_DAMPING");
    double e_curr = 0.0, e_prev = 0.0;
    bool e_converged = false, l_converged = false;

    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), "LCCSD DIIS", DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);

    while (!(e_converged && l_converged)) {
        // RMS of residual per single LMO, for assessing convergence
        std::vector<double> L_ia_rms(naocc, 0.0);
        // RMS of residual per LMO pair, for assessing convergence
        std::vector<double> L_iajb_rms(n_lmo_pairs, 0.0);

        std::time_t time_start = std::time(nullptr);

        // Density intermediates entering both residuals (Toth Eqs. 53--54).
        form_goo();

        // Lambda-doubles residual (Toth Eqs. 75 and 86--91).
        compute_L_iajb(L_iajb, Ln_iajb);

        // Track the largest pair RMS as the Lambda2 convergence metric.
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            L_iajb_rms[ij] = L_iajb[ij]->rms();
        }

        // Jacobi update of Lambda2 with the diagonal semicanonical PNO
        // denominator; damping is applied before DIIS extrapolation.
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];

            // Dynamic Damping
            /*
            double m = (iteration > 10) ? -L_iajb[ij]->vector_dot(K_iajb_[ij]) / L_iajb_prev[ij]->vector_dot(K_iajb_[ij]) : -0.3;
            double alpha = (m > 0.0) ? 1.0 : 1.0 / (1.0 - m);
            */

            for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                    double val = lambda_iajb_[ij]->get(a_ij, b_ij) - (1.0 - damping) * L_iajb[ij]->get(a_ij, b_ij) /
                                    (e_pno_[ij]->get(a_ij) + e_pno_[ij]->get(b_ij) - F_lmo_->get(i,i) - F_lmo_->get(j,j));
                    lambda_iajb_[ij]->set(a_ij, b_ij, val);
                }
            }
            L_iajb_prev[ij] = L_iajb[ij]->clone();
            // L_iajb[ij]->scale(alpha);
        }

        if (include_singles) {
            // Lambda2 has changed, so rebuild Eqs. 53--54 before Lambda1.
            form_goo();

            // Lambda-singles residual (Toth Eqs. 58 and 67--72).
            compute_L_ia(L_ia, L_ia_buffer);

            // Track the largest occupied-domain RMS as the Lambda1 metric.
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < naocc; ++i) {
                L_ia_rms[i] = L_ia[i]->rms();
            }

            // Jacobi update of Lambda1 with its diagonal PNO denominator.
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < naocc; ++i) {
                int ii = i_j_to_ij_[i][i];

                // Dynamic Damping
                /*
                double m = (iteration > 10) ? -L_ia[i]->vector_dot(linalg::doublet(K_iajb_[ii], L_ia[i])) 
                                            / L_ia_prev[i]->vector_dot(linalg::doublet(K_iajb_[ii], L_ia_prev[i])) : -0.3;
                double alpha = (m > 0.0) ? 1.0 : 1.0 / (1.0 - m);
                */

                for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                    double val = lambda_ia_[i]->get(a_ii, 0) - (1.0 - damping) * L_ia[i]->get(a_ii, 0) / (e_pno_[ii]->get(a_ii) - F_lmo_->get(i,i));
                    lambda_ia_[i]->set(a_ii, 0, val);
                }
                L_ia_prev[i] = L_ia[i]->clone();
                // L_ia[i]->scale(alpha);
            }
        }

        // Pulay DIIS: the Lambda residuals are error vectors and the Lambda
        // amplitudes are the vectors extrapolated toward the stationary point.
        std::vector<SharedMatrix> lambda_vecs;
        lambda_vecs.reserve(lambda_ia_.size() + lambda_iajb_.size());
        lambda_vecs.insert(lambda_vecs.end(), lambda_ia_.begin(), lambda_ia_.end());
        lambda_vecs.insert(lambda_vecs.end(), lambda_iajb_.begin(), lambda_iajb_.end());

        std::vector<SharedMatrix> L_vecs;
        L_vecs.reserve(L_ia.size() + L_iajb.size());
        L_vecs.insert(L_vecs.end(), L_ia.begin(), L_ia.end());
        L_vecs.insert(L_vecs.end(), L_iajb.begin(), L_iajb.end());

        auto lambda_vecs_flat = flatten_mats(lambda_vecs);
        auto L_vecs_flat = flatten_mats(L_vecs);

        if (iteration == 1) {
            diis.set_error_vector_size(L_vecs_flat);
            diis.set_vector_size(lambda_vecs_flat);
        }
        
        diis.add_entry(L_vecs_flat.get(), lambda_vecs_flat.get());
        diis.extrapolate(lambda_vecs_flat.get());

        copy_flat_mats(lambda_vecs_flat, lambda_vecs);
        
        // Rebuild the spin-adapted Lambda2 combination (Toth Eq. 55) and use
        // its conventional pseudoenergy as an inexpensive iteration diagnostic;
        // this value is not reported as a physical correlation energy.
        e_curr = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : e_curr)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            lambda_iajb_bar_[ij] = lambda_iajb_[ij]->clone();
            lambda_iajb_bar_[ij]->scale(0.5);
            lambda_iajb_bar_[ij]->add(lambda_iajb_[ij]->transpose());

            e_curr += 0.5 * K_iajb_[ij]->vector_dot(lambda_iajb_[ij]);
        }
        
        double l_curr1 = *max_element(L_ia_rms.begin(), L_ia_rms.end());
        double l_curr2 = *max_element(L_iajb_rms.begin(), L_iajb_rms.end());

        l_converged = (fabs(l_curr1) < options_.get_double("R_CONVERGENCE"));
        l_converged &= (fabs(l_curr2) < options_.get_double("R_CONVERGENCE"));
        e_converged = (fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE"));

        std::time_t time_stop = std::time(nullptr);

        outfile->Printf("  @Lambda-CCSD iter %3d: %16.12f %10.3e %10.3e %10.3e %8d\n", iteration, e_curr, e_curr - e_prev, l_curr1, l_curr2, (int)time_stop - (int)time_start);

        iteration++;

        if (iteration > max_iteration + 1) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }

        // For next iteration
        e_prev = e_curr;
    } // end iter

    lambda_solved_ = true;
}

void DLPNOCCSD::compute_opdm() {

    // Form the local blocks of the unrelaxed, spin-summed CCSD one-particle
    // density from the converged right and left amplitudes (Toth Appendix A,
    // Eqs. A2--A5). compute_ao_opdm() transforms and accumulates these blocks.

    int naocc = i_j_to_ij_.size();
    int n_lmo_pairs = ij_to_i_j_.size();

    form_goo();

    // Occupied-occupied correlation block (Toth Eq. A2).
    Doo_ = rho_oo_->clone();
    Doo_->scale(-1.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];

        auto Doo_temp = linalg::triplet(lambda_ia_[i], S_PNO(ii, jj), T_ia_[j], true, false, false);
        (*Doo_)(i, j) -= Doo_temp->get(0, 0);
    } // end ij

    // Zero out buffers

    // Thread and OMP Parallel Info
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    std::vector<std::vector<SharedMatrix>> D_ov_buffer(nthreads);
    for (int thread = 0; thread < nthreads; ++thread) {
        D_ov_buffer[thread].resize(naocc);
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            D_ov_buffer[thread][i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        }
    }

    // Occupied-virtual block, starting with the first term of Toth Eq. A3.
    Dov_.resize(naocc);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < naocc; ++i) {
        Dov_[i] = T_ia_[i]->clone();
        Dov_[i]->scale(2.0);
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto &[i, j] = ij_to_i_j_[ij];
        int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
            
        // Toth Eq. A3, second term
        auto D_ov_temp = linalg::triplet(S_PNO(ii, ij), Tt_iajb_[ij], S_PNO(ij, jj));
        D_ov_buffer[thread][i]->add(linalg::doublet(D_ov_temp, lambda_ia_[j]));

        // Toth Eq. A3, third term
        auto T_i_to_j = linalg::doublet(S_PNO(ii, jj), T_ia_[j]);
        T_i_to_j->scale(rho_oo_->get(j, i));
        D_ov_buffer[thread][i]->subtract(T_i_to_j);
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int mn = 0; mn < n_lmo_pairs; ++mn) {
        int nlmo_mn = lmopair_to_lmos_[mn].size();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int i_mn = 0; i_mn < nlmo_mn; ++i_mn) {
            int i = lmopair_to_lmos_[mn][i_mn];
            int ii = i_j_to_ij_[i][i];

            // Toth Eq. A3, fourth term
            // (e_{ii}, 1), (e_{ii}, e_{mn}), (e_{mn}, a_{mn})
            auto rho_vv_temp = linalg::triplet(T_ia_[i], S_PNO(ii, mn), rho_vv_[mn], true, false, false); 
            D_ov_buffer[thread][i]->subtract(linalg::doublet(S_PNO(ii, mn), rho_vv_temp, false, true));
        } // end i_mn
    } // end mn

    for (int i = 0; i < naocc; ++i) {
        for (int thread = 0; thread < nthreads; ++thread) {
            Dov_[i]->add(D_ov_buffer[thread][i]);
        } // end thread
    } // end int i

    Dvv_pair_.resize(n_lmo_pairs);
    // Pair-resolved virtual-virtual block rho^VV_ij (Toth Eq. A4).
#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        Dvv_pair_[ij] = rho_vv_[ij]->clone();
    }

    Dvv_singles_.resize(naocc);
    // Singles contribution to the virtual-virtual block (Toth Eq. A5).
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < naocc; ++i) {
        int ii = i_j_to_ij_[i][i];

        Dvv_singles_[i] = std::make_shared<Matrix>(n_pno_[ii], n_pno_[ii]);

        for (int a = 0; a < n_pno_[ii]; ++a) {
            for (int b = 0; b < n_pno_[ii]; ++b) {
                Dvv_singles_[i]->set(a, b, T_ia_[i]->get(a, 0) * lambda_ia_[i]->get(b, 0));
            } // end b
        } // end a
    } // end i
    
} // end function

SharedMatrix DLPNOCCSD::compute_ao_opdm() {
    if (!lambda_solved_) {
        throw PSIEXCEPTION("The DLPNO-CCSD Lambda equations must be solved before forming the OPDM.");
    }

    compute_opdm();

    const int naocc = nalpha_ - nfrzc();
    const int n_lmo_pairs = ij_to_i_j_.size();

    // Assemble the AO density whose one-electron contraction gives Toth
    // Eq. A1. Begin with the spin-summed density of the *current* occupied orbitals.
    // For Brueckner calculations C_lmo_ has been rotated away from the SCF
    // reference, so rebuilding this block from reference C_occ would give an
    // inconsistent OEProp density.
    auto D_ao = linalg::doublet(C_lmo_, C_lmo_, false, true);
    D_ao->scale(2.0);
    auto C_core = reference_wavefunction_->Ca_subset("AO", "FROZEN_OCC");
    if (C_core->ncol() > 0) {
        auto D_core = linalg::doublet(C_core, C_core, false, true);
        D_core->scale(2.0);
        D_ao->add(D_core);
    }

    // Correlated occupied-occupied block (Toth Eq. A2).
    D_ao->add(linalg::triplet(C_lmo_, Doo_, C_lmo_, false, false, true));

    // Correlated pair virtual-virtual blocks (Toth Eq. A4).
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        auto C_pno = submatrix_cols(*C_pao_, lmopair_to_paos_[ij]);
        C_pno = linalg::doublet(C_pno, X_pno_[ij]);
        D_ao->add(linalg::triplet(C_pno, Dvv_pair_[ij], C_pno, false, false, true));
    }

    for (int i = 0; i < naocc; ++i) {
        const int ii = i_j_to_ij_[i][i];
        auto C_i = submatrix_cols(*C_lmo_, std::vector<int>{i});
        auto C_pno = submatrix_cols(*C_pao_, lmopair_to_paos_[ii]);
        C_pno = linalg::doublet(C_pno, X_pno_[ii]);

        // Correlated occupied-virtual block (Toth Eq. A3), plus Lambda1 as
        // the opposite-direction block. Hermitization supplies the transpose
        // without changing contractions with Hermitian one-electron operators.
        auto Dov_total = Dov_[i]->clone();
        Dov_total->add(lambda_ia_[i]);
        D_ao->add(linalg::triplet(C_i, Dov_total, C_pno, false, true, true));

        // Correlated singles virtual-virtual block (Toth Eq. A5).
        D_ao->add(linalg::triplet(C_pno, Dvv_singles_[i], C_pno, false, false, true));
    }

    D_ao->hermitivitize();
    D_ao->set_name("DLPNO-CCSD spin-summed AO OPDM");
    set_array_variable("DLPNO-CCSD AO OPDM", D_ao);
    return D_ao;
}

} // end dlpno
} // end psi

// BEGIN Exact residual transpose response
#include "ccsd_reverse.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <algorithm>
#include <cmath>
#include <chrono>

namespace psi {
namespace dlpno {
using ccsd_detail::add_row;
using ccsd_detail::row_as_column;
using ccsd_detail::zero_blocks;

void DLPNOCCSD::estimate_gradient_lambda_memory() {
    // CCSD resident state, reverse intermediates, residuals/amplitudes/DIIS
    // and the largest local transpose workspace. No amplitude Jacobian is stored.
    const size_t no = T_ia_.size();
    size_t words = ccsd_baseline_memory_doubles_ + triples_local_response_words_ + no * no;
    size_t amp = 0, scratch = 0;
    for (auto& t : T_ia_) amp += t->nrow();
    size_t response = 0, df_scratch = 0;
    for (int p = 0; p < T_iajb_.size(); ++p) {
        const size_t n = n_pno_[p], m = lmopair_to_lmos_[p].size(), q = lmopair_to_ribfs_[p].size();
        const size_t nv = lmopair_to_paos_[p].size();
        response += 3 * n * n + 3 * m * n + n * n * n + q * (m + n) + nv * n;
        size_t widest = n;
        for (const auto& block : J_ikac_non_proj_[p]) {
            response += 2 * size_t(block->nrow()) * block->ncol();
            widest = std::max(widest, size_t(block->ncol()));
        }
        if (ij_to_i_j_[p].first <= ij_to_i_j_[p].second) response += q * (m * n + n * n);
        // Largest pair-local raw factors, eigensystem and simultaneous pullbacks.
        df_scratch = std::max(df_scratch, 10 * q * q + 6 * q * (m * n + n * n + n * widest + m + 2 * n + 1));
        // Saved selection data, parent coefficient covectors and the largest local
        // PAO exchange/DF workspace for either truncation correction.
        if (initial_pno_selection_[p].parent || final_pno_selection_[p].parent) {
            for (const auto* selection : {&initial_pno_selection_[p], &final_pno_selection_[p]}) {
                if (!selection->parent) continue;
                size_t v = selection->amplitudes->nrow(), k = selection->canonical_rotation->ncol();
                response += nv * v + 3 * v * v + k * k + 2 * v;
            }
            response += nv * nv;
            df_scratch = std::max(df_scratch, 4 * q * q + 8 * q * nv + (debug_ ? 40 : 25) * nv * nv);
        }
        amp += n * n;
        words += 2 * m * m + 5 * n * n + 2 * m * n + n + q * (n + m);
        scratch = std::max(scratch,
                           n * n * n + 8 * n * n + (write_qia_pno_ ? q * m * n : 0) + (write_qab_pno_ ? q * n * n : 0));
    }
    words += (11 + 2 * options_.get_int("DIIS_MAX_VECS")) * amp + scratch;
    ccsd_response_words_ = words + response + df_scratch;
    outfile->Printf("    CCSD response memory estimate before sparse overlaps: %.3f GiB\n",
                    ccsd_response_words_ * 8.0 / (1ULL << 30));
    if (toggle_memory_ && ccsd_response_words_ * sizeof(double) > 0.9 * memory_)
        throw PSIEXCEPTION("Insufficient memory for the RHF gradient integral response.");
    outfile->Printf("    RHF gradient Lambda memory estimate: %.3f GiB\n", words * 8.0 / (1ULL << 30));
    if (toggle_memory_ && words * sizeof(double) > 0.9 * memory_)
        throw PSIEXCEPTION("Insufficient memory for the RHF gradient Lambda solve.");
}

void DLPNOCCSD::gradient_lambda_residual(const std::vector<SharedMatrix>& z1, const std::vector<SharedMatrix>& z2,
                                         std::vector<SharedMatrix>& l1, std::vector<SharedMatrix>& l2, bool include_source,
                                         CCAdjoints* workspace) {
    // Differentiate E_CC + lambda1 . R1 + 1/2 sum_ij lambda2_ij . R2_ij.
    // Reverse the right-hand contractions in their original pair domains.
    std::unique_ptr<CCAdjoints> owned;
    if (!workspace) owned = std::make_unique<CCAdjoints>(*this, z1, z2);
    auto& g = workspace ? *workspace : *owned;
    if (workspace) {
        if (&g.z1 != &z1 || &g.z2 != &z2)
            throw PSIEXCEPTION("Lambda transpose workspace belongs to different amplitude vectors.");
        g.zero();
    }
    if (include_source) {
        for (int i = 0; i < g.t1.size(); ++i) {
            const int ii = i_j_to_ij_[i][i];
            auto f = submatrix_rows_and_cols(*F_lmo_pao_, {i}, lmopair_to_paos_[ii]);
            g.t1[i]->axpy(2., linalg::doublet(X_pno_[ii], f, true, true));
        }
        for (int p = 0; p < g.t2.size(); ++p) {
            const auto [i, j] = ij_to_i_j_[p];
            auto si = S_PNO(p, i_j_to_ij_[i][i]), sj = S_PNO(p, i_j_to_ij_[j][j]);
            auto ti = linalg::doublet(si, T_ia_[i]), tj = linalg::doublet(sj, T_ia_[j]);
            g.t1[i]->add(linalg::triplet(si, L_iajb_[p], tj, true, false, false));
            g.t1[j]->add(linalg::triplet(sj, L_iajb_[p], ti, true, true, false));
            g.t2[p]->add(L_iajb_[p]);
        }
    }
    reverse_R1(g);
    reverse_R2(g);
    reverse_pair_intermediates(g);
    reverse_t1_transform(g);

    for (int i = 0; i < l1.size(); ++i) {
        l1[i] = g.t1[i];
        if (include_source) l1[i]->add(triples_source1_[i]);
    }
    for (int p = 0; p < l2.size(); ++p) {
        g.t2[p]->axpy(2., g.u[p]);
        g.t2[p]->subtract(g.u[p]->transpose());
    }
    for (int p = 0; p < l2.size(); ++p) {
        l2[p] = g.t2[p]->clone();
        l2[p]->add(g.t2[ij_to_ji_[p]]->transpose());
        if (include_source) l2[p]->add(triples_source2_[p]);
    }
}

std::vector<SharedMatrix> DLPNOCCSD::ordered_ccsd_blocks(const std::vector<SharedMatrix>& x1,
                                                        const PairBlocks& unique) const {
    if (!gradient_model_ || beta_.size() != T_iajb_.size() || x1.size() != T_ia_.size())
        throw PSIEXCEPTION("Expected singles on the RHF gradient-model occupied space.");
    auto validate = [](const SharedMatrix& x, const SharedMatrix& model) {
        if (!x || x->nirrep() != 1 || x->nrow() != model->nrow() || x->ncol() != model->ncol())
            throw PSIEXCEPTION("CCSD amplitude direction has incompatible dimensions.");
    };
    for (size_t i = 0; i < x1.size(); ++i) validate(x1[i], T_ia_[i]);
    std::vector<SharedMatrix> x2(T_iajb_.size());
    size_t count = 0;
    for (size_t p = 0; p < x2.size(); ++p) {
        auto [i,j] = ij_to_i_j_[p];
        if (i > j) continue;
        ++count;
        const auto& block = unique.at({i,j});
        validate(block, T_iajb_[p]);
        if (i == j) {
            auto antisymmetric = block->transpose();
            antisymmetric->subtract(block);
            if (antisymmetric->absmax() > 1.e-12 * std::max(1., block->absmax()))
                throw PSIEXCEPTION("Diagonal CCSD amplitude directions must be symmetric.");
        }
        x2[p] = block;
        if (i < j) x2[ij_to_ji_[p]] = block->transpose();
    }
    if (unique.size() != count) throw PSIEXCEPTION("Expected one doubles block per retained unique pair.");
    return x2;
}

namespace {
std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> unique_ccsd_state(
    const std::vector<SharedMatrix>& singles, const std::vector<SharedMatrix>& doubles,
    const std::vector<std::pair<int,int>>& pairs) {
    std::vector<SharedMatrix> one;
    DLPNOCCSD::PairBlocks two;
    for (const auto& block : singles) one.push_back(block->clone());
    for (size_t p = 0; p < pairs.size(); ++p) {
        const auto ij = pairs[p];
        if (ij.first <= ij.second) two[ij] = doubles[p]->clone();
    }
    return {one,two};
}
}  // namespace

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> DLPNOCCSD::triples_ccsd_source() const {
    if (!gradient_model_ || triples_source1_.size() != T_ia_.size() || triples_source2_.size() != ij_to_i_j_.size())
        throw PSIEXCEPTION("Expected the converged ordinary triples CCSD source.");
    return unique_ccsd_state(triples_source1_,triples_source2_,ij_to_i_j_);
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> DLPNOCCSD::gradient_ccsd_lambda() const {
    if (!gradient_model_ || !lambda_solved_ || lambda_ia_.size() != T_ia_.size() ||
        lambda_iajb_.size() != ij_to_i_j_.size())
        throw PSIEXCEPTION("Expected the converged RHF gradient-model CCSD(T) multipliers.");
    return unique_ccsd_state(lambda_ia_,lambda_iajb_,ij_to_i_j_);
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> DLPNOCCSD::ccsd_residual_at(
    const std::vector<SharedMatrix>& singles, const PairBlocks& doubles) {
    auto ordered = ordered_ccsd_blocks(singles, doubles);
    const auto saved1 = T_ia_, saved2 = T_iajb_, saved_u = Tt_iajb_;
    auto restore = [&]() {
        T_ia_ = saved1;
        T_iajb_ = saved2;
        Tt_iajb_ = saved_u;
        prepare_ccsd_residual();
    };
    auto r1 = ccsd_detail::zero_blocks(singles), r2 = ccsd_detail::zero_blocks(ordered);
    auto rn = ccsd_detail::zero_blocks(ordered);
    std::vector<std::vector<SharedMatrix>> buffers(Process::environment.get_n_threads());
    for (auto& buffer : buffers) buffer = ccsd_detail::zero_blocks(singles);
    try {
        T_ia_ = singles;
        T_iajb_ = ordered;
        for (size_t p = 0; p < ordered.size(); ++p) {
            Tt_iajb_[p] = ordered[p]->clone();
            Tt_iajb_[p]->scale(2.);
            Tt_iajb_[p]->subtract(ordered[p]->transpose());
        }
        prepare_ccsd_residual();
        compute_R_ia(r1,buffers);
        compute_R_iajb(r2,rn);
    } catch (...) {
        restore();
        throw;
    }
    restore();
    PairBlocks result;
    for (size_t p = 0; p < r2.size(); ++p) {
        const auto ij = ij_to_i_j_[p];
        if (ij.first <= ij.second) result[ij] = r2[p];
    }
    return {r1,result};
}

std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> DLPNOCCSD::ccsd_amplitude_action(
    const std::vector<SharedMatrix>& x1, const PairBlocks& unique, bool transpose) {
    const auto x2 = ordered_ccsd_blocks(x1,unique);
    auto y1 = ccsd_detail::zero_blocks(T_ia_);
    auto y2 = ccsd_detail::zero_blocks(T_iajb_);
    if (transpose) {
        gradient_lambda_residual(x1, x2, y1, y2, false);
    } else {
        double largest = 0.;
        for (const auto& x : x1) largest = std::max(largest, x->absmax());
        for (const auto& x : x2) largest = std::max(largest, x->absmax());
        if (largest > 0.) {
            // R(t+s*x) is at most quartic. Its linear coefficient follows
            // exactly from the two odd parts at s=1 and s=2 (see equation record).
            const double scale = 0.125 / largest;
            auto r1 = ccsd_detail::zero_blocks(T_ia_), r2 = ccsd_detail::zero_blocks(T_iajb_);
            auto rn = ccsd_detail::zero_blocks(T_iajb_);
            std::vector<std::vector<SharedMatrix>> buffers(Process::environment.get_n_threads());
            for (auto& buffer : buffers) buffer = ccsd_detail::zero_blocks(T_ia_);
            const auto t1 = T_ia_, t2 = T_iajb_, u = Tt_iajb_;
            auto restore = [&]() {
                T_ia_ = t1;
                T_iajb_ = t2;
                Tt_iajb_ = u;
                prepare_ccsd_residual();
            };
            try {
                for (int node : {-2,-1,1,2}) {
                    for (size_t i = 0; i < t1.size(); ++i) {
                        T_ia_[i] = t1[i]->clone();
                        T_ia_[i]->axpy(node * scale, x1[i]);
                    }
                    for (size_t p = 0; p < t2.size(); ++p) {
                        T_iajb_[p] = t2[p]->clone();
                        T_iajb_[p]->axpy(node * scale, x2[p]);
                        Tt_iajb_[p] = T_iajb_[p]->clone();
                        Tt_iajb_[p]->scale(2.);
                        Tt_iajb_[p]->subtract(T_iajb_[p]->transpose());
                    }
                    prepare_ccsd_residual();
                    compute_R_ia(r1, buffers);
                    compute_R_iajb(r2, rn);
                    const double weight = (std::abs(node) == 1 ? 8.*node : -0.5*node) / (12.*scale);
                    for (size_t i = 0; i < t1.size(); ++i) y1[i]->axpy(weight, r1[i]);
                    for (size_t p = 0; p < t2.size(); ++p) y2[p]->axpy(weight, r2[p]);
                }
            } catch (...) {
                restore();
                throw;
            }
            restore();
        }
    }
    PairBlocks result;
    for (size_t p = 0; p < y2.size(); ++p) {
        const auto ij = ij_to_i_j_[p];
        if (ij.first <= ij.second) result[ij] = y2[p];
    }
    return {y1, result};
}

void DLPNOCCSD::reverse_R1(CCAdjoints& g) {
    // Jiang Eqs. 87--90. The bare F_ai seed is lambda1 itself.
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        const auto [i, k] = ij_to_i_j_[p];
        const int rev = ij_to_ji_[p], ii = i_j_to_ij_[i][i], n = n_pno_[p];
        auto y = linalg::doublet(S_PNO(p, ii), g.z1[i]);
        auto yrev = linalg::doublet(S_PNO(rev, ii), g.z1[i]);
        // Contract the (cd,a) view of K directly without copying its rank-three storage.
        auto a = std::make_shared<Matrix>(n, n);
        if (n)
            C_DGEMV('N', n * n, n, 1., K_ivvv_[rev]->get_pointer(), n, yrev->get_pointer(), 1, 0., a->get_pointer(), 1);
        g.u[rev]->add(a);
        g.u[p]->add(linalg::doublet(y, Fkc_[rev], false, true));
        g.fkc[rev]->add(linalg::doublet(Tt_iajb_[p], y, true, false));
        for (int l : lmopair_to_lmos_[p]) {
            const int kl = i_j_to_ij_[k][l], ll = i_j_to_ij_[l][l];
            auto S = S_PNO(kl, rev);
            auto Kproj = gradient_lambda_matrix(1, p, l, [&]() {
                return linalg::triplet(S, K_iajb_[kl], S, true, false, false);
            });
            auto dt = linalg::doublet(S_PNO(ii, ll), g.z1[i], true, false);
            g.u[rev]->axpy(-dt->vector_dot(T_ia_[l]), Kproj);
            g.t1[l]->axpy(-Kproj->vector_dot(Tt_iajb_[rev]), dt);
        }

        // B has one row per target occupied orbital in this pair's domain.
        auto rows = std::make_shared<Matrix>(lmopair_to_lmos_[p].size(), n);
        for (int m = 0; m < rows->nrow(); ++m) {
            const int o = lmopair_to_lmos_[p][m];
            add_row(rows, m, linalg::doublet(S_PNO(p, i_j_to_ij_[o][o]), g.z1[o]), -1.);
        }
        auto M = gradient_lambda_matrix(2, p, 0, [&]() {
            auto x = K_mibj_[p]->clone();
            x->add(linalg::doublet(T_n_ij_[p], K_iajb_[p]));
            return x;
        });
        g.u[p]->add(linalg::doublet(rows, M, true, false));
        g.tn[p]->add(linalg::triplet(rows, Tt_iajb_[p], K_iajb_[p], false, false, true));
    }
}

void DLPNOCCSD::reverse_R2(CCAdjoints& g) {
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        const int rev = ij_to_ji_[p];
        const auto& domain = lmopair_to_lmos_[p];
        auto y = g.z2[p]->clone();
        y->add(g.z2[rev]->transpose());
        y->scale(0.5);  // seed of Rn_p after Rn_p + Rn_ji^T
        if (i <= j) {
            // K, A, B, E are formed once and copied into ji, except for ii.
            auto z = y->clone();
            if (i == j) {
                z = g.z2[p]->clone();
                z->scale(0.5);
            }
            g.qa[p]->add(linalg::doublet(i_Qa_t1_[rev], z, false, true));
            g.qa[rev]->add(linalg::doublet(i_Qa_t1_[p], z));
            auto qma = QIA_PNO(p);
            std::vector<SharedMatrix> qab;
            if (!gradient_lambda_fixed_ || gradient_lambda_fixed_->q[p].empty()) qab = QAB_PNO(p);
            for (int q = 0; q < qma.size(); ++q) {
                SharedMatrix Q;
                if (gradient_lambda_fixed_ && !gradient_lambda_fixed_->q[p].empty()) {
                    Q = gradient_lambda_fixed_->q[p][q];
                } else {
                    Q = qab[q]->clone();
                    Q->subtract(linalg::doublet(T_n_ij_[p], qma[q], true, false));
                }
                g.t2[p]->add(linalg::triplet(Q, z, Q, true, false, false));
                auto dq = linalg::triplet(z, Q, T_iajb_[p], false, false, true);
                dq->add(linalg::triplet(z, Q, T_iajb_[p], true, false, false));
                g.tn[p]->subtract(linalg::doublet(qma[q], dq, false, true));
            }

            const bool cached_f = gradient_lambda_fixed_ && gradient_lambda_fixed_->f[p];
            auto F = cached_f ? gradient_lambda_fixed_->f[p] : Fab_[p]->clone();
            auto dF = linalg::doublet(z, T_iajb_[p], true, false);
            dF->add(linalg::doublet(z, T_iajb_[p], false, true));
            g.fab[p]->add(dF);
            for (int k = 0; k < domain.size(); ++k)
                for (int l = 0; l < domain.size(); ++l) {
                    int kl = i_j_to_ij_[domain[k]][domain[l]];
                    if (kl < 0 || !n_pno_[kl]) continue;
                    auto S = S_PNO(kl, p);
                    auto zp = linalg::triplet(S, z, S, false, false, true);
                    g.t2[kl]->axpy((*beta_[p])(k, l), zp);
                    (*g.beta[p])(k, l) += zp->vector_dot(T_iajb_[kl]);
                    auto dp = linalg::triplet(S, dF, S, false, false, true);
                    g.u[kl]->subtract(linalg::doublet(dp, K_iajb_[kl]));
                    if (!cached_f) {
                        auto uk = linalg::doublet(Tt_iajb_[kl], K_iajb_[kl], false, true);
                        F->subtract(linalg::triplet(S, uk, S, true, false, false));
                    }
                }
            g.t2[p]->add(linalg::doublet(z, F));
            g.t2[p]->add(linalg::doublet(F, z, true, false));
        }

        // C and D have different source and target pair spaces on each leg.
        auto yc = y->transpose();
        yc->axpy(0.5, y);
        for (int m = 0; m < domain.size(); ++m) {
            const int k = domain[m], ki = i_j_to_ij_[k][i], kj = i_j_to_ij_[k][j];
            const int ik = ij_to_ji_[ki], jk = ij_to_ji_[kj];
            auto reverse_mixed = [&](int a, int b, const SharedMatrix& intermediate, const SharedMatrix& integral, int family,
                                     const SharedMatrix& amp, const SharedMatrix& seed, double factor,
                                     const SharedMatrix& d_amp, const SharedMatrix& d_intermediate) {
                auto s1 = S_PNO(p, a), s2 = S_PNO(a, b), s3 = S_PNO(b, p);
                auto bare = gradient_lambda_matrix(family, p, k, [&]() {
                    auto x = integral->clone();
                    if (family == 4) {
                        x->scale(2.);
                        x->subtract(J_ikac_non_proj_[p][m]);
                    }
                    x->add(linalg::triplet(s1, intermediate, s2));
                    return x;
                });
                auto dz = linalg::triplet(seed, s3, amp, false, true, false);
                d_intermediate->axpy(factor, linalg::triplet(s1, dz, s2, true, false, true));
                d_amp->axpy(factor, linalg::triplet(s3, seed, bare, false, true, false));
            };
            reverse_mixed(ki, kj, gamma_[ki], J_ikac_non_proj_[p][m], 3, T_iajb_[kj], yc, -1., g.t2[kj],
                          g.gamma[ki]);
            reverse_mixed(ik, jk, delta_[ik], K_iakc_non_proj_[p][m], 4, Tt_iajb_[jk], y, 0.5, g.u[jk], g.delta[ik]);
        }

        for (int k : lmopair_to_lmos_[i_j_to_ij_[i][i]]) {
            const int ik = i_j_to_ij_[i][k], kj = i_j_to_ij_[k][j];
            auto S = S_PNO(p, ik);
            auto zp = linalg::triplet(S, y, S, true, false, false);
            g.t2[ik]->axpy(-(*F_im_double_tilde_)(k, j), zp);
            if (kj >= 0) g.fkj[kj] -= zp->vector_dot(T_iajb_[ik]);
        }
    }
}

void DLPNOCCSD::reverse_pair_intermediates(CCAdjoints& g) {
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        auto [a, b] = ij_to_i_j_[p];
        const int rev = ij_to_ji_[p], n = n_pno_[p];
        auto qma = QIA_PNO(p);
        // beta and F''_ij, Jiang Eqs. 82 and 86.
        g.qk[p]->add(linalg::doublet(i_Qk_t1_[rev], g.beta[p], false, true));
        g.qk[rev]->add(linalg::doublet(i_Qk_t1_[p], g.beta[p]));
        for (const auto& q : qma) g.t2[p]->add(linalg::triplet(q, g.beta[p], q, true, false, false));
        for (int l : lmopair_to_lmos_[p]) {
            int la = i_j_to_ij_[l][a], lb = i_j_to_ij_[l][b];
            auto S = S_PNO(la, lb);
            auto projected = gradient_lambda_matrix(5, p, l, [&]() {
                return linalg::triplet(S, K_iajb_[la], S, true, false, false);
            });
            g.u[lb]->axpy(g.fkj[p], projected);
        }

        // gamma_ki and delta_ik (Eqs. 83--84). The external single is on
        // the second occupied label of gamma and on the first label of delta.
        for (bool gamma : {true, false}) {
            int i = gamma ? b : a, k = gamma ? a : b;
            int ii = i_j_to_ij_[i][i];
            const auto& z = gamma ? g.gamma[p] : g.delta[p];
            auto M = J_ijmb_[p]->clone();
            if (!gamma) {
                M->scale(-1.);
                M->axpy(2., K_mibj_[p]);
            }
            g.tn[p]->subtract(linalg::doublet(M, z, false, true));
            auto S = S_PNO(p, ii);
            auto flat = z->clone();
            flat->reshape(n * n, 1);
            const auto& K = K_ivvv_[gamma ? p : rev];
            auto di = linalg::doublet(K, flat);
            if (!gamma) {
                // 2 (kc|ab) t_i^b - (kb|ac) t_i^b
                di->scale(-1.);
                auto zt = z->transpose();
                zt->reshape(n * n, 1);
                if (n) C_DGEMV('T', n * n, n, 2., K->get_pointer(), n, zt->get_pointer(), 1, 1., di->get_pointer(), 1);
            }
            g.t1[i]->add(linalg::doublet(S, di, true, false));
            for (int l : lmopair_to_lmos_[p]) {
                const int ll = i_j_to_ij_[l][l];
                const int other = gamma ? i_j_to_ij_[k][l] : i_j_to_ij_[l][k];
                const int amp = gamma ? i_j_to_ij_[l][i] : i_j_to_ij_[i][l];
                const auto& integral = gamma ? K_iajb_[other] : L_iajb_[other];
                auto s1 = S_PNO(p, ll), s2 = S_PNO(other, ii), so = S_PNO(p, other);
                auto tl = gradient_lambda_matrix(gamma ? 6 : 7, p, l, [&]() { return linalg::doublet(s1, T_ia_[l]); });
                // - t_l^p (S_p,other K^T S_other,ii t_i)^T
                auto map = gradient_lambda_matrix(gamma ? 8 : 9, p, l, [&]() {
                    return linalg::triplet(so, integral, s2, false, true, false);
                });
                auto v = gradient_lambda_matrix(gamma ? 10 : 11, p, l, [&]() { return linalg::doublet(map, T_ia_[i]); });
                g.t1[l]->subtract(linalg::triplet(s1, z, v, true, false, false));
                g.t1[i]->subtract(linalg::triplet(map, z, tl, true, true, false));
                // +/- 1/2 S_p,amp (t or u)_amp S_amp,other K S_other,p
                auto right = gradient_lambda_matrix(gamma ? 12 : 13, p, l, [&]() {
                    return linalg::triplet(S_PNO(amp, other), integral, S_PNO(other, p));
                });
                auto result = linalg::triplet(S_PNO(p, amp), z, right, true, false, true);
                if (gamma)
                    g.t2[amp]->axpy(-0.5, result);
                else
                    g.u[amp]->axpy(0.5, result);
            }
        }
    }
}

void DLPNOCCSD::reverse_t1_transform(CCAdjoints& g) {
    // Fai_i = Fai_bar_i + Fab_ii t_i - sum_k t_k^ii Fij_bar(k,i).
    for (int i = 0; i < T_ia_.size(); ++i) {
        const int ii = i_j_to_ij_[i][i];
        const auto& y = g.z1[i];
        g.t1[i]->add(linalg::doublet(Fab_[ii], y, true, false));
        g.fab[ii]->add(linalg::doublet(y, T_ia_[i], false, true));
        const auto& domain = lmopair_to_lmos_[ii];
        for (int m = 0; m < domain.size(); ++m) {
            int k = domain[m], ki = i_j_to_ij_[k][i];
            double fij = (*Fkj_)(k, i) - row_as_column(Fkc_bar_[ii], m)->vector_dot(T_ia_[i]);
            add_row(g.tn[ii], m, y, -fij);
            g.fij_bar[ki] -= row_as_column(T_n_ij_[ii], m)->vector_dot(y);
        }
    }
    // Fkj and Fkc (one row per ordered pair) precede Fab/Fkc_bar in the reverse pass.
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        const int jj = i_j_to_ij_[j][j], m = lmopair_to_lmos_dense_[jj][i];
        g.fij_bar[p] += g.fkj[p];
        add_row(g.fkc_bar[jj], m, T_ia_[j], g.fkj[p]);
        g.t1[j]->axpy(g.fkj[p], row_as_column(Fkc_bar_[jj], m));
        for (int k : lmopair_to_lmos_[p]) {
            int ik = i_j_to_ij_[i][k], kk = i_j_to_ij_[k][k];
            auto v = linalg::triplet(L_iajb_[ik], S_PNO(p, ik), g.fkc[p], true, true, false);
            g.t1[k]->add(linalg::doublet(S_PNO(ik, kk), v, true, false));
        }
    }

    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        const int rev = ij_to_ji_[p], m = lmopair_to_lmos_dense_[p][i];
        auto M = J_ijmb_[p]->clone();
        M->scale(2.);
        M->subtract(K_mibj_[rev]);
        g.tn[p]->axpy(g.fij_bar[p], M);
        auto qma = QIA_PNO(p), qab = QAB_PNO(p);
        if (i <= j) {
            // Fab = Fab_bar - Tn^T Fkc_bar. Each partially dressed Fock
            // contribution has the form 2 <Qma,Tn> Q - Q Tn^T Qma.
            g.tn[p]->subtract(linalg::doublet(Fkc_bar_[p], g.fab[p], false, true));
            g.fkc_bar[p]->subtract(linalg::doublet(T_n_ij_[p], g.fab[p]));
            for (int q = 0; q < qma.size(); ++q) {
                double dgamma = g.fab[p]->vector_dot(qab[q]) + g.fkc_bar[p]->vector_dot(qma[q]);
                g.tn[p]->axpy(2. * dgamma, qma[q]);
                g.tn[p]->subtract(linalg::triplet(qma[q], g.fab[p], qab[q], false, true, false));
                g.tn[p]->subtract(linalg::triplet(qma[q], g.fkc_bar[p], qma[q], false, true, false));
                if (i == j) {
                    auto qa = row_as_column(i_Qa_ij_[p], q);
                    auto qk = row_as_column(i_Qk_ij_[p], q);
                    const auto& y = g.z1[i];
                    g.tn[p]->axpy(2. * y->vector_dot(qa), qma[q]);
                    g.tn[p]->subtract(linalg::triplet(qk, y, qab[q], false, true, false));
                }
            }
        }

        // i_Qa_t1 = i_Qa - i_Qk_t1 Tn + Qab t_i^p.
        // i_Qk_t1(q,k) = i_Qk(q,k) + Qma(q,k,:) . t_i^p.
        g.qk[p]->subtract(linalg::doublet(g.qa[p], T_n_ij_[p], false, true));
        g.tn[p]->subtract(linalg::doublet(i_Qk_t1_[p], g.qa[p], true, false));
        for (int q = 0; q < qma.size(); ++q) {
            add_row(g.tn[p], m, linalg::doublet(qab[q], row_as_column(g.qa[p], q), true, false));
            add_row(g.tn[p], m, linalg::doublet(qma[q], row_as_column(g.qk[p], q), true, false));
        }
        for (int k = 0; k < lmopair_to_lmos_[p].size(); ++k) {
            int o = lmopair_to_lmos_[p][k];
            g.t1[o]->add(linalg::doublet(S_PNO(p, i_j_to_ij_[o][o]), row_as_column(g.tn[p], k), true, false));
        }
    }
}

SharedMatrix DLPNOCCSD::gradient_lambda_matrix(int family, int pair, int member,
                                               const std::function<SharedMatrix()>& build) {
    auto* fixed = gradient_lambda_fixed_.get();
    if (!fixed) return build();
    const auto key = std::make_tuple(family, pair, member);
    auto found = fixed->matrices.find(key);
    if (found != fixed->matrices.end()) {
        ++fixed->hits;
        return found->second;
    }
    ++fixed->builds;
    auto value = build();
    const size_t bytes = sizeof(double) * size_t(value->nrow()) * value->ncol();
    if (bytes <= fixed->limit - fixed->bytes) {
        fixed->matrices.emplace(key, value);
        fixed->bytes += bytes;
    }
    return value;
}

void DLPNOCCSD::build_gradient_lambda_fixed() {
    timer_on("DLPNO Lambda fixed intermediates");
    gradient_lambda_fixed_ = std::make_unique<GradientLambdaFixed>();
    auto& fixed = gradient_lambda_fixed_;
    fixed->q.resize(T_iajb_.size());
    fixed->f.resize(T_iajb_.size());
    const size_t peak = ccsd_response_words_ * sizeof(double);
    const size_t headroom = 0.9 * memory_ > peak ? size_t(0.9 * memory_) - peak : 0;
    const size_t budget = std::min({std::max(size_t(64) * 1024 * 1024,
                                            ccsd_baseline_memory_doubles_ * sizeof(double)),
                                    size_t(memory_ / 32), headroom});
    fixed->limit = budget;
    auto& used = fixed->bytes;
    size_t cached_q = 0, cached_f = 0;
    for (size_t p = 0; p < T_iajb_.size(); ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        if (i > j) continue;
        const size_t n = n_pno_[p], words = n * n;
        if (words * sizeof(double) <= budget - used) {
            used += words * sizeof(double); // Reserve before overlap fallback admits panels.
            auto F = Fab_[p]->clone();
            const auto& domain = lmopair_to_lmos_[p];
            for (int k : domain) for (int l : domain) {
                const int kl = i_j_to_ij_[k][l];
                if (kl < 0 || !n_pno_[kl]) continue;
                auto S = S_PNO(kl, p);
                auto uk = linalg::doublet(Tt_iajb_[kl], K_iajb_[kl], false, true);
                F->subtract(linalg::triplet(S, uk, S, true, false, false));
            }
            fixed->f[p] = F;
            ++cached_f;
        }
        const size_t qwords = lmopair_to_ribfs_[p].size() * words;
        if (qwords * sizeof(double) <= budget - used) {
            auto qma = QIA_PNO(p), qab = QAB_PNO(p);
            for (size_t q = 0; q < qab.size(); ++q) {
                auto Q = qab[q]->clone();
                Q->subtract(linalg::doublet(T_n_ij_[p], qma[q], true, false));
                fixed->q[p].push_back(Q);
            }
            used += qwords * sizeof(double);
            ++cached_q;
        }
    }
    outfile->Printf("    Lambda fixed intermediates: %zu Q pairs, %zu F pairs, %.1f MiB\n",
                    cached_q, cached_f, used / 1048576.);
    timer_off("DLPNO Lambda fixed intermediates");
}

void DLPNOCCSD::prepare_gradient_lambda_overlaps() {
    auto& fixed = *gradient_lambda_fixed_;
    if (fixed.overlap_requests.size() == fixed.overlaps.size()) return;
    std::map<int, std::set<int>> domains;
    std::map<int, size_t> full_bytes, resident_bytes;
    for (auto [p, q] : fixed.overlap_requests) {
        domains[p].insert(lmopair_to_paos_[q].begin(), lmopair_to_paos_[q].end());
        const size_t bytes = sizeof(double) * size_t(n_pno_[p]) * n_pno_[q];
        full_bytes[p] += bytes;
        if (fixed.overlaps.count({p,q})) resident_bytes[p] += bytes;
    }
    // Same one-sided extended-PAO factorization as LOW_MEMORY_OVERLAP.
    // Choose storage once after the source traversal reveals the actual graph.
    // A later iteration forms only the final small right transformation.
    for (const auto& [p, labels] : domains) {
        if (resident_bytes[p] == full_bytes[p]) continue;
        const size_t bytes = sizeof(double) * size_t(n_pno_[p]) * labels.size();
        if (bytes >= full_bytes[p] || bytes > fixed.limit - fixed.bytes + resident_bytes[p]) continue;
        std::vector<int> ids(labels.begin(), labels.end());
        for (auto it = fixed.overlaps.begin(); it != fixed.overlaps.end();) {
            if (it->first.first == p) it = fixed.overlaps.erase(it);
            else ++it;
        }
        fixed.bytes -= resident_bytes[p];
        auto metric = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[p], ids);
        GradientLambdaFixed::OverlapPanel panel;
        panel.left = linalg::doublet(X_pno_[p], metric, true, false);
        for (size_t k = 0; k < ids.size(); ++k) panel.columns.emplace(ids[k], k);
        fixed.overlap_panels.emplace(p, std::move(panel));
        fixed.bytes += bytes;
    }
}

void DLPNOCCSD::solve_gradient_lambda() {
    reset_lambda_state();
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    auto elapsed = [](auto since) { return std::chrono::duration<double>(Clock::now() - since).count(); };
    estimate_gradient_lambda_memory();
    prepare_ccsd_residual();
    beta_ = compute_beta();
    gamma_ = compute_gamma();
    delta_ = compute_delta();
    F_im_double_tilde_ = compute_Fkj_double_tilde();
    lambda_ia_ = zero_blocks(T_ia_);
    lambda_iajb_ = zero_blocks(T_iajb_);
    std::vector<SharedMatrix> l1(T_ia_.size()), l2(T_iajb_.size());
    // Clear on both normal exit and exceptions, before any debug amplitude
    // perturbation. Cached matrices are read-only throughout this scope.
    struct FixedGuard {
        std::unique_ptr<GradientLambdaFixed>& fixed;
        ~FixedGuard() { fixed.reset(); }
    } fixed_guard{gradient_lambda_fixed_};
    build_gradient_lambda_fixed();
    const double setup_seconds = elapsed(start);
    start = Clock::now();
    std::vector<SharedMatrix> source1(T_ia_.size()), source2(T_iajb_.size());
    gradient_lambda_residual(lambda_ia_, lambda_iajb_, source1, source2, true);
    prepare_gradient_lambda_overlaps();
    auto transpose_workspace = std::make_unique<CCAdjoints>(*this, lambda_ia_, lambda_iajb_);
    const double source_seconds = elapsed(start);
    start = Clock::now();

    const double rtol = options_.get_double("R_CONVERGENCE"), etol = options_.get_double("E_CONVERGENCE");
    const double step = 1. - options_.get_double("DLPNO_LAMBDA_DAMPING");
    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), "DLPNO gradient Lambda DIIS",
                     DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);
    outfile->Printf("\n  ==> Ordinary (T)-sourced RHF Lambda <==\n\n");
    outfile->Printf("      Iter    pseudoenergy      delta E      max L1 RMS   max L2 RMS\n");
    double last_energy = 0.0;
    for (int iter = 0;; ++iter) {
        gradient_lambda_residual(lambda_ia_, lambda_iajb_, l1, l2, false, transpose_workspace.get());
        for (size_t i = 0; i < l1.size(); ++i) l1[i]->add(source1[i]);
        for (size_t p = 0; p < l2.size(); ++p) l2[p]->add(source2[p]);
        if (debug_ && iter == 1) {
            // Compare the persistent implementation with the original complete
            // reverse at a nonzero Lambda vector, including the fixed source.
            auto saved = std::move(gradient_lambda_fixed_);
            std::vector<SharedMatrix> direct1(l1.size()), direct2(l2.size());
            try { gradient_lambda_residual(lambda_ia_, lambda_iajb_, direct1, direct2, true); }
            catch (...) { gradient_lambda_fixed_ = std::move(saved); throw; }
            gradient_lambda_fixed_ = std::move(saved);
            double error = 0.;
            for (size_t i = 0; i < l1.size(); ++i) {
                direct1[i]->subtract(l1[i]);
                error = std::max(error, direct1[i]->absmax());
            }
            for (size_t p = 0; p < l2.size(); ++p) {
                direct2[p]->subtract(l2[p]);
                error = std::max(error, direct2[p]->absmax());
            }
            set_scalar_variable("DLPNO LAMBDA PERSISTENT REPLAY ERROR", error);
            if (!std::isfinite(error) || error > 1.e-11)
                throw PSIEXCEPTION("Persistent Lambda reverse disagrees with direct evaluation.");
        }
        auto max_rms = [](const auto& blocks) {
            double r = 0.;
            for (const auto& b : blocks) {
                const double value = b->rms();
                if (!std::isfinite(value)) throw PSIEXCEPTION("Nonfinite residual in RHF gradient Lambda equations.");
                r = std::max(r, value);
            }
            return r;
        };
        const double r1 = max_rms(l1), r2 = max_rms(l2);
        double energy = 0.;
        for (int p = 0; p < l2.size(); ++p) energy += 0.5 * K_iajb_[p]->vector_dot(lambda_iajb_[p]);
        outfile->Printf("      %3d  %16.10f   %10.2e   %10.2e   %10.2e\n", iter, energy, energy - last_energy, r1, r2);
        if (std::max(r1, r2) < rtol && std::abs(energy - last_energy) < etol) {
            set_scalar_variable("DLPNO CCSD(T) LAMBDA1 RESIDUAL", r1);
            set_scalar_variable("DLPNO CCSD(T) LAMBDA2 RESIDUAL", r2);
            break;
        }
        if (iter >= options_.get_int("DLPNO_MAXITER"))
            throw PSIEXCEPTION("Ordinary (T)-sourced DLPNO Lambda equations did not converge.");
        last_energy = energy;
        for (int i = 0; i < l1.size(); ++i) {
            const auto& eps = *e_pno_[i_j_to_ij_[i][i]];
            for (int a = 0; a < l1[i]->nrow(); ++a)
                (*lambda_ia_[i])(a, 0) -= step * (*l1[i])(a, 0) / (eps(a) - (*F_lmo_)(i, i));
        }
        for (int p = 0; p < l2.size(); ++p) {
            const auto [i, j] = ij_to_i_j_[p];
            const auto& eps = *e_pno_[p];
            const double occ = (*F_lmo_)(i, i) + (*F_lmo_)(j, j);
            for (int a = 0; a < l2[p]->nrow(); ++a)
                for (int b = 0; b < l2[p]->ncol(); ++b)
                    (*lambda_iajb_[p])(a, b) -= step * (*l2[p])(a, b) / (eps(a) + eps(b) - occ);
        }
        std::vector<SharedMatrix> amps = lambda_ia_, errs = l1;
        amps.insert(amps.end(), lambda_iajb_.begin(), lambda_iajb_.end());
        errs.insert(errs.end(), l2.begin(), l2.end());
        auto x = flatten_mats(amps), r = flatten_mats(errs);
        if (!iter) {
            diis.set_error_vector_size(r);
            diis.set_vector_size(x);
        }
        diis.add_entry(r.get(), x.get());
        diis.extrapolate(x.get());
        copy_flat_mats(x, amps);
    }

    const double iteration_seconds = elapsed(start);
    lambda_solved_ = true;
    lambda_iajb_bar_.resize(l2.size());
    // Basis-independent contractions used for the canonical-limit comparison.
    double l1t1 = 0., l2t2 = 0., eta1 = 0., eta2 = 0.;
    for (int i = 0; i < l1.size(); ++i) {
        l1t1 += lambda_ia_[i]->vector_dot(T_ia_[i]);
        eta1 += triples_source1_[i]->vector_dot(T_ia_[i]);
    }
    for (int p = 0; p < l2.size(); ++p) {
        lambda_iajb_bar_[p] = lambda_iajb_[p]->transpose();
        lambda_iajb_bar_[p]->axpy(0.5, lambda_iajb_[p]);
        l2t2 += 0.5 * lambda_iajb_[p]->vector_dot(T_iajb_[p]);
        eta2 += 0.5 * triples_source2_[p]->vector_dot(T_iajb_[p]);
    }
    set_scalar_variable("DLPNO CCSD(T) LAMBDA1 T1 CONTRACTION", l1t1);
    set_scalar_variable("DLPNO CCSD(T) LAMBDA2 T2 CONTRACTION", l2t2);
    set_scalar_variable("DLPNO (T) T1 SCALE DERIVATIVE", eta1);
    set_scalar_variable("DLPNO (T) T2 SCALE DERIVATIVE", eta2);
    const auto& fixed = *gradient_lambda_fixed_;
    outfile->Printf("    Lambda persistent work: %.1f/%.1f MiB; fixed matrix builds/reuses %zu/%zu; "
                    "fallback overlap builds/reuses %zu/%zu; semidirect panels/uses %zu/%zu\n", fixed.bytes / 1048576., fixed.limit / 1048576.,
                    fixed.builds, fixed.hits, fixed.overlap_builds, fixed.overlap_hits,
                    fixed.overlap_panels.size(), fixed.overlap_panel_uses);
    gradient_lambda_fixed_.reset();
    transpose_workspace.reset();
    source1.clear();
    source2.clear();
    start = Clock::now();
    form_ccsd_pair_response();
    backtransform_ccsd_one_body();
    outfile->Printf("    Lambda phase times: setup %.3f s; fixed source %.3f s; iterations %.3f s; "
                    "pair density %.3f s\n", setup_seconds, source_seconds, iteration_seconds, elapsed(start));
    if (debug_) {
        check_ccsd_adjoint();
        check_ccsd_pair_response();
    }
}

double DLPNOCCSD::ccsd_lagrangian(const std::vector<SharedMatrix>& z1, const std::vector<SharedMatrix>& z2,
                                  std::vector<SharedMatrix>* right1, std::vector<SharedMatrix>* right2) {
    std::vector<SharedMatrix> r1, r2, rn;
    for (const auto& t : T_ia_) r1.push_back(t->clone());
    for (const auto& t : T_iajb_) {
        r2.push_back(t->clone());
        rn.push_back(t->clone());
    }
    const int nth = Process::environment.get_n_threads();
    std::vector<std::vector<SharedMatrix>> buffers(nth);
    for (auto& buffer : buffers)
        for (const auto& t : T_ia_) buffer.push_back(t->clone());
    prepare_ccsd_residual();
    compute_R_ia(r1, buffers);
    compute_R_iajb(r2, rn);
    double value = singles_fock_energy();
    for (int i = 0; i < r1.size(); ++i) value += z1[i]->vector_dot(r1[i]);
    for (int pair = 0; pair < r2.size(); ++pair) {
        const auto [i, j] = ij_to_i_j_[pair];
        auto ti = linalg::doublet(S_PNO(pair, i_j_to_ij_[i][i]), T_ia_[i]);
        auto tj = linalg::doublet(S_PNO(pair, i_j_to_ij_[j][j]), T_ia_[j]);
        auto tau = linalg::doublet(ti, tj, false, true);
        tau->add(T_iajb_[pair]);
        value += tau->vector_dot(L_iajb_[pair]) + 0.5 * z2[pair]->vector_dot(r2[pair]);
    }
    if (right1) *right1 = std::move(r1);
    if (right2) *right2 = std::move(r2);
    return value;
}

// The probe respects t_{ji}^{ba}=t_{ij}^{ab}, including diagonal-pair symmetry.
std::vector<SharedMatrix> DLPNOCCSD::amplitude_probe(bool singles, int seed) const {
    const auto& amplitudes = singles ? T_ia_ : T_iajb_;
    std::vector<SharedMatrix> probe(amplitudes.size());
    double norm2 = 0.0;
    for (size_t m = 0; m < probe.size(); ++m) {
        auto x = amplitudes[m]->clone();
        auto [i, j] = singles ? std::make_pair(static_cast<int>(m), static_cast<int>(m)) : ij_to_i_j_[m];
        for (int a = 0; a < x->nrow(); ++a)
            for (int b = 0; b < x->ncol(); ++b) {
                const int p = i <= j ? a : b, q = i <= j ? b : a;
                auto component = [&](int u, int v) {
                    return std::sin(0.43 * (1 + std::min(i, j)) + 0.57 * (1 + std::max(i, j)) +
                                    seed * (1.3 * (1 + u) + 0.79 * (1 + v)));
                };
                (*x)(a, b) = !singles && i == j ? 0.5 * (component(p, q) + component(q, p)) : component(p, q);
            }
        norm2 += x->vector_dot(x);
        probe[m] = std::move(x);
    }
    if (norm2 > 0)
        for (auto& x : probe) x->scale(1. / std::sqrt(norm2));
    return probe;
}

double DLPNOCCSD::probe_amplitudes(bool singles, const std::vector<SharedMatrix>& direction, double h,
                                   const std::function<double()>& evaluate) {
    auto& amplitudes = singles ? T_ia_ : T_iajb_;
    const auto original = amplitudes, original_u = Tt_iajb_;
    auto sample = [&](double step) {
        for (int m = 0; m < amplitudes.size(); ++m) {
            amplitudes[m] = original[m]->clone();
            amplitudes[m]->axpy(step, direction[m]);
            if (!singles) {
                Tt_iajb_[m] = amplitudes[m]->clone();
                Tt_iajb_[m]->scale(2.);
                Tt_iajb_[m]->subtract(amplitudes[m]->transpose());
            }
        }
        return evaluate();
    };
    double plus, minus;
    try {
        plus = sample(h);
        minus = sample(-h);
    } catch (...) {
        amplitudes = original;
        Tt_iajb_ = original_u;
        throw;
    }
    amplitudes = original;
    Tt_iajb_ = original_u;
    return (plus - minus) / (2 * h);
}

void DLPNOCCSD::check_ccsd_adjoint() {
    // Use independent test multipliers: cancellation at a stationary point
    // must not conceal an incorrect transpose. Finite differences vary only
    // amplitudes at a single geometry. Leave the solved multipliers intact.
    const auto z1 = amplitude_probe(true, 3), z2 = amplitude_probe(false, 4);
    std::vector<SharedMatrix> l1(z1.size()), l2(z2.size());
    gradient_lambda_residual(z1, z2, l1, l2);
    auto lagrangian = [&]() { return ccsd_lagrangian(z1, z2); };
    for (bool singles : {true, false}) {
        const auto probe = amplitude_probe(singles);
        const auto& l = singles ? l1 : l2;
        const auto& source = singles ? triples_source1_ : triples_source2_;
        double analytic = 0.0;
        for (int m = 0; m < l.size(); ++m) {
            analytic += probe[m]->vector_dot(l[m]);
            if (!source.empty()) analytic -= probe[m]->vector_dot(source[m]);
        }
        if (!singles) analytic *= 0.5;
        const double numeric = probe_amplitudes(singles, probe, 1.e-5, lagrangian);
        outfile->Printf("    CCSD L%d direction: analytic % .12e, numerical % .12e, error %.3e\n", singles ? 1 : 2,
                        analytic, numeric, std::abs(analytic - numeric));
        set_scalar_variable(singles ? "DLPNO CCSD L1 DIRECTION ERROR" : "DLPNO CCSD L2 DIRECTION ERROR",
                            std::abs(analytic - numeric));
    }
    prepare_ccsd_residual();
}

}  // namespace dlpno
}  // namespace psi

// END Exact residual transpose response
