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
#include <array>
#include <cstring>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {
namespace dlpno {

#ifdef USING_Einsums
using einsums::All;
using einsums::Indices;
using einsums::Tensor;
using einsums::TensorView;
using einsums::tensor_algebra::einsum;
using einsums::tensor_algebra::permute;
namespace index = einsums::index;
#endif

// The right-hand W/V/T construction follows Jiang et al., J. Chem. Phys. 161,
// 082502 (2024), DOI: 10.1063/5.0219963. The left moment and asymmetric energy
// contraction follow Toth, Jiang, and Schaefer, J. Chem. Theory Comput. 22,
// 7667--7681 (2026), DOI: 10.1021/acs.jctc.6c00758. Triples tensors are stored
// as matrices with row a and flattened column (b * ntno + c).

DLPNOCCSD_T::DLPNOCCSD_T(SharedWavefunction ref_wfn, Options &options) : DLPNOCCSD(ref_wfn, options) {}
DLPNOCCSD_T::~DLPNOCCSD_T() {}

void DLPNOCCSD_T::print_header(DLPNOCCSDPhase phase) {
    bool t0_only = options_.get_bool("T0_APPROXIMATION");
    std::string triples_algorithm = (t0_only) ? "SEMICANONICAL (T0)" : "ITERATIVE (T)";
    const bool bccd_result = brueckner_orbs_ && phase == DLPNOCCSDPhase::FinalBrueckner;
    std::string method = bccd_result ? "DLPNO-BCCD" : "DLPNO-CCSD";
    method += t0_only ? "(T0)" : (lambda_requested_ ? "(T)_L" : "(T)");
    double t_cut_tno = options_.get_double("T_CUT_TNO");
    double t_cut_tno_strong_scale = options_.get_double("T_CUT_TNO_STRONG_SCALE");
    double t_cut_tno_weak_scale = options_.get_double("T_CUT_TNO_WEAK_SCALE");

    outfile->Printf("   --------------------------------------------\n");
    outfile->Printf("                 %-27s\n", method.c_str());
    outfile->Printf("                    by Andy Jiang              \n");
    outfile->Printf("               DOI: 10.1063/5.0219963          \n");
    if (lambda_requested_) {
        outfile->Printf("          DOI: 10.1021/acs.jctc.6c00758        \n");
    }
    outfile->Printf("   --------------------------------------------\n\n");
    outfile->Printf("  DLPNO convergence set to %s.\n\n", options_.get_str("PNO_CONVERGENCE").c_str());
    outfile->Printf("  Detailed DLPNO thresholds and cutoffs:\n");
    outfile->Printf("    ALGORITHM    = %6s   \n", triples_algorithm.c_str());
    outfile->Printf("    T_CUT_TNO (T0)             = %6.3e \n", t_cut_tno);
    outfile->Printf("    T_CUT_DO_TRIPLES (T0)      = %6.3e \n", options_.get_double("T_CUT_DO_TRIPLES"));
    outfile->Printf("    T_CUT_MKN_TRIPLES (T0)     = %6.3e \n", options_.get_double("T_CUT_MKN_TRIPLES"));
    outfile->Printf("    T_CUT_TRIPLES_WEAK (T0)    = %6.3e \n", options_.get_double("T_CUT_TRIPLES_WEAK"));
    outfile->Printf("    T_CUT_TNO_PRE (T0)         = %6.3e \n", options_.get_double("T_CUT_TNO_PRE"));
    outfile->Printf("    T_CUT_DO_TRIPLES_PRE (T0)  = %6.3e \n", options_.get_double("T_CUT_DO_TRIPLES_PRE"));
    outfile->Printf("    T_CUT_MKN_TRIPLES_PRE (T0) = %6.3e \n", options_.get_double("T_CUT_MKN_TRIPLES_PRE"));
    if (!t0_only) {
        outfile->Printf("    T_CUT_TNO_STRONG (T)       = %6.3e \n", t_cut_tno * t_cut_tno_strong_scale);
        outfile->Printf("    T_CUT_TNO_WEAK (T)         = %6.3e \n", t_cut_tno * t_cut_tno_weak_scale);
        outfile->Printf("    F_CUT_T (T)                = %6.3e \n", options_.get_double("F_CUT_T"));
        outfile->Printf("    T_CUT_ITER (T)             = %6.3e \n", options_.get_double("T_CUT_ITER"));
    }
    outfile->Printf("    MIN_TNOS                   = %6d   \n", options_.get_int("MIN_TNOS"));
    outfile->Printf("    TRIPLES_MAX_WEAK_PAIRS     = %6d   \n", options_.get_int("TRIPLES_MAX_WEAK_PAIRS"));
    outfile->Printf("\n\n");
}

SharedMatrix DLPNOCCSD_T::matmul_3d(SharedMatrix A, SharedMatrix X, int dim_old, int dim_new) {
    /*
    Performs the operation A'[i,j,k] = A[I,J,K] * X[i,I] * X[j,J] * X[k,K] for cube 3d tensors
    */

    SharedMatrix A_new = linalg::doublet(X, A, false, false);
    A_new->reshape(dim_new * dim_old, dim_old);
    A_new = linalg::doublet(A_new, X, false, true);

    SharedMatrix A_T = std::make_shared<Matrix>(dim_new * dim_new, dim_old);
    for (int ind = 0; ind < dim_new * dim_new * dim_old; ++ind) {
        int a = ind / (dim_new * dim_old), b = (ind / dim_old) % dim_new, c = ind % dim_old;
        (*A_T)(a *dim_new + b, c) = (*A_new)(a * dim_old + c, b);
    }
    A_T = linalg::doublet(A_T, X, false, true);

    A_new = std::make_shared<Matrix>(dim_new, dim_new * dim_new);

    for (int ind = 0; ind < dim_new * dim_new * dim_new; ++ind) {
        int a = ind / (dim_new * dim_new), b = (ind / dim_new) % dim_new, c = ind % dim_new;
        (*A_new)(a, b *dim_new + c) = (*A_T)(a * dim_new + c, b);
    }

    return A_new;
}

void DLPNOCCSD_T::triples_sparsity(bool prescreening) {
    /* 
    In the prescreening step, this generates the initial list of triplets from
    strong and weak pairs ijk from strong and weak pairs ij, jk, and ik 
    (weak pairs set by TRIPLES_MAX_WEAK_PAIRS).

    In the second prescreening step, screened triplets are determined and removed
    from the rest of the computation. Ordinary and asymmetric screened energies are
    accumulated separately.
    */

    timer_on("Triples Sparsity");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();
    int MAX_WEAK_PAIRS = options_.get_int("TRIPLES_MAX_WEAK_PAIRS");

    if (prescreening) {
        int ijk = 0;
        for (int ij = 0; ij < n_lmo_pairs; ij++) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            if (i > j) continue;
            for (int k : lmopair_to_lmos_[ij]) {
                if (i > k || j > k) continue;
                if (i == j && j == k) continue;
                int ij_weak = i_j_to_ij_weak_[i][j], ik_weak = i_j_to_ij_weak_[i][k], kj_weak = i_j_to_ij_weak_[k][j];

                int weak_pair_count = 0;
                if (ij_weak != -1) weak_pair_count += 1;
                if (ik_weak != -1) weak_pair_count += 1;
                if (kj_weak != -1) weak_pair_count += 1;

                if (weak_pair_count > MAX_WEAK_PAIRS) continue;

                ijk_to_i_j_k_.push_back(std::make_tuple(i, j, k));
                i_j_k_to_ijk_[i * naocc * naocc + j * naocc + k] = ijk;
                i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j] = ijk;
                i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k] = ijk;
                i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i] = ijk;
                i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j] = ijk;
                i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i] = ijk;
                ++ijk;
            }
        }
    } else {
        std::unordered_map<int, int> i_j_k_to_ijk_new;
        std::vector<std::tuple<int, int, int>> ijk_to_i_j_k_new;

        double t_cut_triples_weak = options_.get_double("T_CUT_TRIPLES_WEAK");
        de_lccsd_t_screened_ = 0.0;
        de_lccsd_t_l_screened_ = 0.0;

        int ijk_new = 0;
        for (int ijk = 0; ijk < ijk_to_i_j_k_.size(); ++ijk) {
            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

            const bool right_significant = std::fabs(e_ijk_right_[ijk]) >= t_cut_triples_weak;
            const bool target_significant = std::fabs(e_ijk_[ijk]) >= t_cut_triples_weak;
            // For an asymmetric calculation retain the union of triplets
            // significant to ordinary (T) and (T)_L. This lets one run publish
            // both corrections without biasing either screened contribution.
            if (right_significant || target_significant) {
                ijk_to_i_j_k_new.push_back(std::make_tuple(i, j, k));
                i_j_k_to_ijk_new[i * naocc * naocc + j * naocc + k] = ijk_new;
                i_j_k_to_ijk_new[i * naocc * naocc + k * naocc + j] = ijk_new;
                i_j_k_to_ijk_new[j * naocc * naocc + i * naocc + k] = ijk_new;
                i_j_k_to_ijk_new[j * naocc * naocc + k * naocc + i] = ijk_new;
                i_j_k_to_ijk_new[k * naocc * naocc + i * naocc + j] = ijk_new;
                i_j_k_to_ijk_new[k * naocc * naocc + j * naocc + i] = ijk_new;
                ++ijk_new;
            } else {
                de_lccsd_t_screened_ += e_ijk_right_[ijk];
                if (lambda_requested_) de_lccsd_t_l_screened_ += e_ijk_[ijk];
            }
        }
        i_j_k_to_ijk_ = i_j_k_to_ijk_new;
        ijk_to_i_j_k_ = ijk_to_i_j_k_new;
    }

    int n_lmo_triplets = ijk_to_i_j_k_.size();
    int natom = molecule_->natom();
    int nbf = basisset_->nbf();

    tno_scale_.clear();
    tno_scale_.resize(n_lmo_triplets, 1.0);

    // => Local density fitting domains <= //

    SparseMap lmo_to_ribfs(naocc);
    SparseMap lmo_to_riatoms(naocc);

    double t_cut_mkn_triples = (prescreening) ? options_.get_double("T_CUT_MKN_TRIPLES_PRE") : options_.get_double("T_CUT_MKN_TRIPLES");

    for (size_t i = 0; i < naocc; ++i) {
        // atomic mulliken populations for this orbital
        std::vector<double> mkn_pop(natom, 0.0);

        auto P_i = reference_wavefunction_->S()->clone();

        for (size_t u = 0; u < nbf; u++) {
            P_i->scale_row(0, u, C_lmo_->get(u, i));
            P_i->scale_column(0, u, C_lmo_->get(u, i));
        }

        for (size_t u = 0; u < nbf; u++) {
            int centerU = basisset_->function_to_center(u);
            double p_uu = P_i->get(u, u);

            for (size_t v = 0; v < nbf; v++) {
                int centerV = basisset_->function_to_center(v);
                double p_vv = P_i->get(v, v);

                // off-diag pops (p_uv) split between u and v prop to diag pops
                double p_uv = P_i->get(u, v);
                mkn_pop[centerU] += p_uv * ((p_uu) / (p_uu + p_vv));
                mkn_pop[centerV] += p_uv * ((p_vv) / (p_uu + p_vv));
            }
        }

        // if non-zero mulliken pop on atom, include atom in the LMO's fitting domain
        for (size_t a = 0; a < natom; a++) {
            if (fabs(mkn_pop[a]) > t_cut_mkn_triples) {
                lmo_to_riatoms[i].push_back(a);

                // each atom's aux orbitals are all-or-nothing for each LMO
                for (int u : atom_to_ribf_[a]) {
                    lmo_to_ribfs[i].push_back(u);
                }
            }
        }
    }

    // => PAO domains <= //

    SparseMap lmo_to_paos(naocc);

    double t_cut_do_triples = (prescreening) ? options_.get_double("T_CUT_DO_TRIPLES_PRE") : options_.get_double("T_CUT_DO_TRIPLES");

    for (size_t i = 0; i < naocc; ++i) {
        // PAO domains determined by differential overlap integral
        std::vector<int> lmo_to_paos_temp;
        for (size_t u = 0; u < nbf; ++u) {
            if (fabs(DOI_iu_->get(i, u)) > t_cut_do_triples) {
                lmo_to_paos_temp.push_back(u);
            }
        }

        // if any PAO on an atom is in the list, we take all of the PAOs on that atom
        lmo_to_paos[i] = contract_lists(lmo_to_paos_temp, atom_to_bf_);
    }

    if (!prescreening) {
        lmotriplet_to_ribfs_.clear();
        lmotriplet_to_lmos_.clear();
        lmotriplet_to_paos_.clear();
    }

    lmotriplet_to_ribfs_.resize(n_lmo_triplets);
    lmotriplet_to_lmos_.resize(n_lmo_triplets);
    lmotriplet_to_paos_.resize(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        lmotriplet_to_ribfs_[ijk] = merge_lists(lmo_to_ribfs[i], merge_lists(lmo_to_ribfs[j], lmo_to_ribfs[k]));
        for (int l = 0; l < naocc; ++l) {
            int il = i_j_to_ij_[i][l], jl = i_j_to_ij_[j][l], kl = i_j_to_ij_[k][l];
            if (il != -1 && jl != -1 && kl != -1) lmotriplet_to_lmos_[ijk].push_back(l);
        }
        lmotriplet_to_paos_[ijk] = merge_lists(lmo_to_paos[i], merge_lists(lmo_to_paos[j], lmo_to_paos[k]));
    }


    timer_off("Triples Sparsity");
}

void DLPNOCCSD_T::sort_triplets(double e_total) {
    /* Sorting triplets by energy values to determine strong and weak
       triplets for the iterative (T) portion of the computation */

    timer_on("Sort Triplets");

    outfile->Printf("  ==> Sorting Triplets <== \n\n");

    int n_lmo_triplets = ijk_to_i_j_k_.size();
    if (n_lmo_triplets == 0) {
        outfile->Printf("    No surviving LMO triplets.\n\n");
        timer_off("Sort Triplets");
        return;
    }

    std::vector<std::pair<int, double>> ijk_e_pairs(n_lmo_triplets);

#pragma omp parallel for
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        ijk_e_pairs[ijk] = std::make_pair(ijk, e_ijk_[ijk]);
    }

    std::sort(ijk_e_pairs.begin(), ijk_e_pairs.end(), [&](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        return (std::fabs(a.second) > std::fabs(b.second));
    });

    double e_curr = 0.0;
    double strong_scale = options_.get_double("T_CUT_TNO_STRONG_SCALE");
    double weak_scale = options_.get_double("T_CUT_TNO_WEAK_SCALE");
    is_strong_triplet_.resize(n_lmo_triplets, false);
    tno_scale_.clear();
    tno_scale_.resize(n_lmo_triplets, weak_scale);

    int strong_count = 0;
    for (int idx = 0; idx < n_lmo_triplets; ++idx) {
        is_strong_triplet_[ijk_e_pairs[idx].first] = true;
        tno_scale_[ijk_e_pairs[idx].first] = strong_scale;
        e_curr += ijk_e_pairs[idx].second;
        ++strong_count;
        if (std::fabs(e_total) > 0.0 && std::fabs(e_curr / e_total) > 0.9) break;
    }

    outfile->Printf("    Number of Strong Triplets: %6d, Total Triplets: %6d, Ratio: %.4f\n\n", strong_count, n_lmo_triplets, 
                            (double) strong_count / n_lmo_triplets);

    timer_off("Sort Triplets");
}

void DLPNOCCSD_T::tno_transform(double t_cut_tno) {
    // Computes TNOs from converged LCCSD densities of triplets ij, jk, and ik
    
    timer_on("TNO transform");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();
    const int MIN_TNOS = options_.get_int("MIN_TNOS");

    X_tno_.clear();
    e_tno_.clear();
    n_tno_.clear();

    X_tno_.resize(n_lmo_triplets);
    e_tno_.resize(n_lmo_triplets);
    n_tno_.resize(n_lmo_triplets);

    if (n_lmo_triplets == 0) {
        outfile->Printf("    Number of (Unique) Local MO triplets: 0\n\n");
        timer_off("TNO transform");
        return;
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        // number of PAOs in the triplet domain (before removing linear dependencies)
        int npao_ijk = lmotriplet_to_paos_[ijk].size();

        // number of auxiliary basis in the domain
        int naux_ijk = lmotriplet_to_ribfs_[ijk].size();

        //                                          //
        // ==> Canonicalize PAOs of triplet ijk <== //
        //                                          //

        auto S_pao_ijk = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijk]);
        auto F_pao_ijk = submatrix_rows_and_cols(*F_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijk]);

        SharedMatrix X_pao_ijk;
        SharedVector e_pao_ijk;
        std::tie(X_pao_ijk, e_pao_ijk) = orthocanonicalizer(S_pao_ijk, F_pao_ijk);

        F_pao_ijk = linalg::triplet(X_pao_ijk, F_pao_ijk, X_pao_ijk, true, false, false);

        // number of PAOs in the domain after removing linear dependencies
        int npao_can_ijk = X_pao_ijk->ncol();

        // S_ijk partially transformed overlap matrix
        std::vector<int> triples_ext_domain = merge_lists(lmo_to_paos_[i], merge_lists(lmo_to_paos_[j], lmo_to_paos_[k]));
        auto S_ijk = submatrix_rows_and_cols(*S_pao_, triples_ext_domain, lmotriplet_to_paos_[ijk]);
        S_ijk = linalg::doublet(S_ijk, X_pao_ijk, false, false);
        

        //                                           //
        // ==> Canonical PAOs  to Canonical TNOs <== //
        //                                           //

        size_t nvir_ijk = F_pao_ijk->nrow();

        // Construct pair densities from amplitudes
        auto D_ij = linalg::doublet(Tt_iajb_[ij], T_iajb_[ij], false, true);
        D_ij->add(linalg::doublet(Tt_iajb_[ij], T_iajb_[ij], true, false));
        if (i == j) D_ij->scale(0.5);

        auto D_jk = linalg::doublet(Tt_iajb_[jk], T_iajb_[jk], false, true);
        D_jk->add(linalg::doublet(Tt_iajb_[jk], T_iajb_[jk], true, false));
        if (j == k) D_jk->scale(0.5);

        auto D_ik = linalg::doublet(Tt_iajb_[ik], T_iajb_[ik], false, true);
        D_ik->add(linalg::doublet(Tt_iajb_[ik], T_iajb_[ik], true, false));
        if (i == k) D_ik->scale(0.5);

        // Project pair densities into combined PAO space of ijk
        std::vector<int> ij_index = index_list(triples_ext_domain, lmopair_to_paos_[ij]);
        auto S_ij = linalg::doublet(X_pno_[ij], submatrix_rows(*S_ijk, ij_index), true, false);
        D_ij = linalg::triplet(S_ij, D_ij, S_ij, true, false, false);

        std::vector<int> jk_index = index_list(triples_ext_domain, lmopair_to_paos_[jk]);
        auto S_jk = linalg::doublet(X_pno_[jk], submatrix_rows(*S_ijk, jk_index), true, false);
        D_jk = linalg::triplet(S_jk, D_jk, S_jk, true, false, false);

        std::vector<int> ik_index = index_list(triples_ext_domain, lmopair_to_paos_[ik]);
        auto S_ik = linalg::doublet(X_pno_[ik], submatrix_rows(*S_ijk, ik_index), true, false);
        D_ik = linalg::triplet(S_ik, D_ik, S_ik, true, false, false);

        // Construct triplet density from pair densities
        auto D_ijk = D_ij->clone();
        D_ijk->add(D_jk);
        D_ijk->add(D_ik);
        D_ijk->scale(1.0 / 3.0);

        // Diagonalization of triplet density gives TNOs (in basis of LMO's virtual domain)
        // as well as TNO occ numbers
        auto X_tno_ijk = std::make_shared<Matrix>("eigenvectors", nvir_ijk, nvir_ijk);
        Vector tno_occ("eigenvalues", nvir_ijk);
        D_ijk->diagonalize(*X_tno_ijk, tno_occ, descending);

        double tno_scale = tno_scale_[ijk];

        int nvir_ijk_final = 0;
        for (size_t a = 0; a < nvir_ijk; ++a) {
            if (fabs(tno_occ.get(a)) >= tno_scale * t_cut_tno || a < MIN_TNOS) {
                nvir_ijk_final++;
            }
        }

        nvir_ijk_final = std::max(1, nvir_ijk_final);

        Dimension zero(1);
        Dimension dim_final(1);
        dim_final.fill(nvir_ijk_final);

        // This transformation gives orbitals that are orthonormal but not canonical
        X_tno_ijk = X_tno_ijk->get_block({zero, X_tno_ijk->rowspi()}, {zero, dim_final});
        tno_occ = tno_occ.get_block({zero, dim_final});

        SharedMatrix tno_canon;
        SharedVector e_tno_ijk;
        std::tie(tno_canon, e_tno_ijk) = canonicalizer(X_tno_ijk, F_pao_ijk);

        X_tno_ijk = linalg::doublet(X_tno_ijk, tno_canon, false, false);
        X_tno_ijk = linalg::doublet(X_pao_ijk, X_tno_ijk, false, false);

        X_tno_[ijk] = X_tno_ijk;
        e_tno_[ijk] = e_tno_ijk;
        n_tno_[ijk] = X_tno_ijk->ncol();
    }

    int tno_count_total = 0, tno_count_min = C_pao_->ncol(), tno_count_max = 0;
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        tno_count_total += n_tno_[ijk];
        tno_count_min = std::min(tno_count_min, n_tno_[ijk]);
        tno_count_max = std::max(tno_count_max, n_tno_[ijk]);
    }

    // (n + 3 choose 3) minus triples from the same orbital (iii)
    size_t n_total_possible = (naocc + 2) * (naocc + 1) * (naocc) / 6 - naocc;

    outfile->Printf("  \n");
    outfile->Printf("    Number of (Unique) Local MO triplets: %d\n", n_lmo_triplets);
    outfile->Printf("    Max Number of Possible (Unique) LMO Triplets: %zu (Ratio: %.4f)\n", n_total_possible,
                    (double)n_lmo_triplets / n_total_possible);
    outfile->Printf("    Natural Orbitals per Local MO triplet:\n");
    outfile->Printf("      Avg: %3d NOs \n", tno_count_total / n_lmo_triplets);
    outfile->Printf("      Min: %3d NOs \n", tno_count_min);
    outfile->Printf("      Max: %3d NOs \n", tno_count_max);
    outfile->Printf("  \n");

    timer_off("TNO transform");
}

void DLPNOCCSD_T::estimate_triples_memory() {
    outfile->Printf("\n  ==> DLPNO-(T) Memory Requirements <== \n\n");

    int n_lmo_triplets = ijk_to_i_j_k_.size();

    size_t tno_total_memory = 0;
#pragma omp parallel for reduction(+ : tno_total_memory)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        tno_total_memory += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijk];
    }

    // Depending on which intermediates are written to disk, the RAM usage varies
    double W3_memory = tno_total_memory;
    double V3_memory = tno_total_memory;
    double L3_memory = lambda_requested_ ? tno_total_memory : 0.0;
    double T3_memory = tno_total_memory;

    // Write W and V intermediates to disk (this is more efficient than writing amplitudes)
    write_intermediates_ = options_.get_bool("WRITE_TRIPLES_INTERMEDIATES");
    if (write_intermediates_) W3_memory = 0.0, V3_memory = 0.0, L3_memory = 0.0;

    // Write T3 amplitudes to disk (this should only be turned on as a last resort)
    write_amplitudes_ = options_.get_bool("WRITE_TRIPLES_AMPLITUDES");
    if (write_amplitudes_) T3_memory = 0.0;

    size_t total_memory = qij_memory_ + qia_memory_ + qab_memory_ + W3_memory + V3_memory + L3_memory + T3_memory;

    // 1 GB = 1000^3 = 10^9 Bytes
    const double DOUBLES_TO_GB = pow(10.0, -9) * sizeof(double);
    const double WORDS_TO_GB = pow(10.0, -9);

    outfile->Printf("    (q | i j) integrals    : %.3f [GB]\n", qij_memory_ * DOUBLES_TO_GB);
    outfile->Printf("    (q | i a) integrals    : %.3f [GB]\n", qia_memory_ * DOUBLES_TO_GB);
    outfile->Printf("    (q | a b) integrals    : %.3f [GB]\n", qab_memory_ * DOUBLES_TO_GB);
    outfile->Printf("    W_{ijk}^{abc}          : %.3f [GB]\n", W3_memory * DOUBLES_TO_GB);
    outfile->Printf("    V_{ijk}^{abc}          : %.3f [GB]\n", V3_memory * DOUBLES_TO_GB);
    outfile->Printf("    L_{ijk}^{abc}          : %.3f [GB]\n", L3_memory * DOUBLES_TO_GB);
    outfile->Printf("    T_{ijk}^{abc}          : %.3f [GB]\n", T3_memory * DOUBLES_TO_GB);
    outfile->Printf("    Total Memory Required  : %.3f [GB]\n", total_memory * DOUBLES_TO_GB);
    outfile->Printf("    Total Memory Given     : %.3f [GB]\n\n", memory_ * WORDS_TO_GB);
    

    // Memory checks!!!
    bool memory_changed = false;

    if (toggle_memory_ && !write_intermediates_ && total_memory * sizeof(double) > 0.9 * memory_) {
        outfile->Printf("  Total Required Memory is more than 90%% of Available Memory!\n");
        outfile->Printf("    Attempting to switch to disk IO for W and V intermediates...\n");

        total_memory -= (W3_memory + V3_memory + L3_memory);
        W3_memory = 0.0, V3_memory = 0.0, L3_memory = 0.0;
        write_intermediates_ = true;
        memory_changed = true;
        outfile->Printf("    Required Memory Reduced to %.3f [GB]\n\n", total_memory * DOUBLES_TO_GB);
    }

    if (toggle_memory_ && !write_amplitudes_ && total_memory * sizeof(double) > 0.9 * memory_) {
        outfile->Printf("  Total Required Memory is (still) more than 90%% of Available Memory!\n");
        outfile->Printf("    Attempting to switch to disk IO for T3 amplitudes...\n");

        total_memory -= T3_memory;
        T3_memory = 0.0;
        write_amplitudes_ = true;
        memory_changed = true;
        outfile->Printf("    Required Memory Reduced to %.3f [GB]\n\n", total_memory * DOUBLES_TO_GB);
    }

    // This will likely never be executed, barring a pathological case 
    // (as the memory here is less than what is needed for CCSD)
    if (toggle_memory_ && total_memory * sizeof(double) > 0.9 * memory_) {
        outfile->Printf("  Total Required Memory is (still) more than 90%% of Available Memory!\n");
        throw PSIEXCEPTION("   Too little memory given for DLPNO-(T) Algorithm!");
    }

    if (memory_changed) {
        outfile->Printf("\n  ==> (Updated) DLPNO-(T) Memory Requirements <== \n\n");
        outfile->Printf("    (q | i j) integrals    : %.3f [GB]\n", qij_memory_ * DOUBLES_TO_GB);
        outfile->Printf("    (q | i a) integrals    : %.3f [GB]\n", qia_memory_ * DOUBLES_TO_GB);
        outfile->Printf("    (q | a b) integrals    : %.3f [GB]\n", qab_memory_ * DOUBLES_TO_GB);
        outfile->Printf("    W_{ijk}^{abc}          : %.3f [GB]\n", W3_memory * DOUBLES_TO_GB);
        outfile->Printf("    V_{ijk}^{abc}          : %.3f [GB]\n", V3_memory * DOUBLES_TO_GB);
        outfile->Printf("    L_{ijk}^{abc}          : %.3f [GB]\n", L3_memory * DOUBLES_TO_GB);
        outfile->Printf("    T_{ijk}^{abc}          : %.3f [GB]\n", T3_memory * DOUBLES_TO_GB);
        outfile->Printf("    Total Memory Required  : %.3f [GB]\n", total_memory * DOUBLES_TO_GB);
        outfile->Printf("    Total Memory Given     : %.3f [GB]\n\n", memory_ * WORDS_TO_GB);
        
    }

    if (write_intermediates_) {
        if (lambda_requested_) {
            outfile->Printf("    Writing W_{ijk}^{abc}, V_{ijk}^{abc}, and L_{ijk}^{abc} to disk...\n\n");
        } else {
            outfile->Printf("    Writing W_{ijk}^{abc} and V_{ijk}^{abc} to disk...\n\n");
        }
    }

    if (write_amplitudes_) {
        outfile->Printf("    Writing T_{ijk}^{abc} to disk...\n\n");
    }
    
    if (!write_intermediates_ && !write_amplitudes_) {
        outfile->Printf("    Storing all X_{ijk}^{abc} quantities in RAM...\n\n");
    }
}

std::pair<double, double> DLPNOCCSD_T::compute_lccsd_t0(bool save_memory) {
    // Form the semicanonical right triples numerator W, the ordinary energy
    // moment V, and, when requested, the left triples moment L. The Toth et al.
    // asymmetric correction contracts L with the same right-hand T3 amplitudes
    // used by the ordinary Jiang et al. correction.
    timer_on("LCCSD(T0)");

    int n_lmo_triplets = ijk_to_i_j_k_.size();

    double E_T0 = 0.0, E_T_L0 = 0;

    if (save_memory) {
        W_iajbkc_.resize(n_lmo_triplets);
        V_iajbkc_.resize(n_lmo_triplets);
        T_iajbkc_.resize(n_lmo_triplets);
        if (lambda_requested_) L_iajbkc_.resize(n_lmo_triplets);
    }

    e_ijk_.clear();
    e_ijk_.resize(n_lmo_triplets, 0.0);
    e_ijk_right_.clear();
    e_ijk_right_.resize(n_lmo_triplets, 0.0);

    std::time_t time_start = std::time(nullptr);
    std::time_t time_lap = std::time(nullptr);

    // Sort Triplets by the approximate number of operations (for maximal parallel efficiency)
    std::vector<std::pair<int, size_t>> ijk_cost_tuple(n_lmo_triplets);
    
#pragma omp parallel for
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        const int npao_ijk = lmotriplet_to_paos_[ijk].size();
        const int naux_ijk = lmotriplet_to_ribfs_[ijk].size();

        // Cost of transforming q_vv from PAO to TNO basis
        size_t cost = naux_ijk * n_tno_[ijk] * npao_ijk * npao_ijk;
        cost += naux_ijk * n_tno_[ijk] * n_tno_[ijk] * npao_ijk;

        ijk_cost_tuple[ijk] = std::make_pair(ijk, cost);
    }
    
    std::sort(ijk_cost_tuple.begin(), ijk_cost_tuple.end(), [&](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b) {
        return (a.second > b.second);
    });

    std::vector<int> ijk_sorted_by_cost(n_lmo_triplets);
    
#pragma omp parallel for
    for (int ijk_idx = 0; ijk_idx < n_lmo_triplets; ++ijk_idx) {
        ijk_sorted_by_cost[ijk_idx] = ijk_cost_tuple[ijk_idx].first;
    }

#pragma omp parallel for schedule(dynamic) reduction(+ : E_T0) reduction(+ : E_T_L0)
    for (int ijk_idx = 0; ijk_idx < n_lmo_triplets; ++ijk_idx) {
        // Triplets assigned to threads dynamically, sorted in descending order of cost
        // This maximizes parallel efficiency
        int ijk = ijk_sorted_by_cost[ijk_idx];

        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        int ntno_ijk = n_tno_[ijk];

        if (ntno_ijk == 0) continue;

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        if (thread == 0) timer_on("LCCSD(T0): Setup Integrals");

        // => Step 1: Compute all necessary integrals

        // number of LMOs in the triplet domain
        const int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();
        // number of PAOs in the triplet domain (before removing linear dependencies)
        const int npao_ijk = lmotriplet_to_paos_[ijk].size();
        // number of auxiliary functions in the triplet domain
        const int naux_ijk = lmotriplet_to_ribfs_[ijk].size();

        /// => Build (i a_ijk | b_ijk d_jk) and (k c_ijk | j l) integrals <= ///

        auto q_iv = std::make_shared<Matrix>(naux_ijk, npao_ijk); // (Q_{ijk} | i u_{ijk})
        auto q_jv = std::make_shared<Matrix>(naux_ijk, npao_ijk); // (Q_{ijk} | j u_{ijk})
        auto q_kv = std::make_shared<Matrix>(naux_ijk, npao_ijk); // (Q_{ijk} | k u_{ijk})

        auto q_io = std::make_shared<Matrix>(naux_ijk, nlmo_ijk); // (Q_{ijk} | m_{ijk} i)
        auto q_jo = std::make_shared<Matrix>(naux_ijk, nlmo_ijk); // (Q_{ijk} | m_{ijk} j)
        auto q_ko = std::make_shared<Matrix>(naux_ijk, nlmo_ijk); // (Q_{ijk} | m_{ijk} k)

        auto q_vv = std::make_shared<Matrix>(naux_ijk, ntno_ijk * ntno_ijk); // (Q_{ijk} | a_{ijk} b_{ijk})

        for (int q_ijk = 0; q_ijk < naux_ijk; q_ijk++) {
            const int q = lmotriplet_to_ribfs_[ijk][q_ijk];
            const int centerq = ribasis_->function_to_center(q);

            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                (*q_io)(q_ijk, l_ijk) = (*qij_[q])(riatom_to_lmos_ext_dense_[centerq][i], riatom_to_lmos_ext_dense_[centerq][l]);
                (*q_jo)(q_ijk, l_ijk) = (*qij_[q])(riatom_to_lmos_ext_dense_[centerq][j], riatom_to_lmos_ext_dense_[centerq][l]);
                (*q_ko)(q_ijk, l_ijk) = (*qij_[q])(riatom_to_lmos_ext_dense_[centerq][k], riatom_to_lmos_ext_dense_[centerq][l]);
            }

            for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                int u = lmotriplet_to_paos_[ijk][u_ijk];
                (*q_iv)(q_ijk, u_ijk) = (*qia_[q])(riatom_to_lmos_ext_dense_[centerq][i], riatom_to_paos_ext_dense_[centerq][u]);
                (*q_jv)(q_ijk, u_ijk) = (*qia_[q])(riatom_to_lmos_ext_dense_[centerq][j], riatom_to_paos_ext_dense_[centerq][u]);
                (*q_kv)(q_ijk, u_ijk) = (*qia_[q])(riatom_to_lmos_ext_dense_[centerq][k], riatom_to_paos_ext_dense_[centerq][u]);
            }

            auto q_vv_tmp = std::make_shared<Matrix>(npao_ijk, npao_ijk);
            q_vv_tmp->zero();

            for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                int u = lmotriplet_to_paos_[ijk][u_ijk];
                for (int v_ijk = 0; v_ijk < npao_ijk; ++v_ijk) {
                    int v = lmotriplet_to_paos_[ijk][v_ijk];
                    int uv_idx = riatom_to_pao_pairs_dense_[centerq][u][v];
                    if (uv_idx == -1) continue;
                    (*q_vv_tmp)(u_ijk, v_ijk) = (*qab_[q])(uv_idx, 0);
                } // end v_ijk
            } // end u_ijk
            
            // naux_{ijk} * npao_{ijk}^{2} * ntno_{ijk} (this is the most expensive operation in this loop)
            q_vv_tmp = linalg::triplet(X_tno_[ijk], q_vv_tmp, X_tno_[ijk], true, false, false);
            ::memcpy(&(*q_vv)(q_ijk, 0), &(*q_vv_tmp)(0, 0), ntno_ijk * ntno_ijk * sizeof(double));

            // naux_ijk * npao_ijk^2 * ntno_{ijk}
        } // end q_ijk

        q_iv = linalg::doublet(q_iv, X_tno_[ijk]); // (Q_{ijk} | i u_{ijk}) -> (Q_{ijk} | i a_{ijk})
        q_jv = linalg::doublet(q_jv, X_tno_[ijk]); // (Q_{ijk} | j u_{ijk}) -> (Q_{ijk} | j a_{ijk})
        q_kv = linalg::doublet(q_kv, X_tno_[ijk]); // (Q_{ijk} | k u_{ijk}) -> (Q_{ijk} | k a_{ijk})
        
        auto q_iv_clone = q_iv->clone();
        auto q_jv_clone = q_jv->clone();
        auto q_kv_clone = q_kv->clone();

        auto A_solve = submatrix_rows_and_cols(*full_metric_, lmotriplet_to_ribfs_[ijk], lmotriplet_to_ribfs_[ijk]);

        /* These are cloned and inverted by the full coulomb metric (not to the half power)
            to make formation of (i a | b c)-type integrals more efficient later */

        C_DGESV_wrapper(A_solve->clone(), q_iv_clone);
        C_DGESV_wrapper(A_solve->clone(), q_jv_clone);
        C_DGESV_wrapper(A_solve->clone(), q_kv_clone);
        
        A_solve->power(0.5, 1.0e-14);

        C_DGESV_wrapper(A_solve->clone(), q_iv);
        C_DGESV_wrapper(A_solve->clone(), q_jv);
        C_DGESV_wrapper(A_solve->clone(), q_kv);
        C_DGESV_wrapper(A_solve->clone(), q_io);
        C_DGESV_wrapper(A_solve->clone(), q_jo);
        C_DGESV_wrapper(A_solve->clone(), q_ko);

        if (thread == 0) timer_off("LCCSD(T0): Setup Integrals");

        if (thread == 0) timer_on("LCCSD(T0): Contract Integrals");

        // W integrals
        auto K_ivvv = linalg::doublet(q_iv_clone, q_vv, true, false); // (i a_{ijk} | b_{ijk} d_{ijk})
        auto K_jvvv = linalg::doublet(q_jv_clone, q_vv, true, false); // (j b_{ijk} | c_{ijk} d_{ijk})
        auto K_kvvv = linalg::doublet(q_kv_clone, q_vv, true, false); // (k c_{ijk} | a_{ijk} d_{ijk})

        auto K_iojv = linalg::doublet(q_io, q_jv, true, false); // (i l_{ijk} | j b_{ijk})
        auto K_joiv = linalg::doublet(q_jo, q_iv, true, false); // (j l_{ijk} | i a_{ijk})
        auto K_kojv = linalg::doublet(q_ko, q_jv, true, false); // (k l_{ijk} | j b_{ijk})
        auto K_jokv = linalg::doublet(q_jo, q_kv, true, false); // (j l_{ijk} | k c_{ijk})
        auto K_iokv = linalg::doublet(q_io, q_kv, true, false); // (i l_{ijk} | k c_{ijk})
        auto K_koiv = linalg::doublet(q_ko, q_iv, true, false); // (k l_{ijk} | i a_{ijk})

        // V integrals
        auto K_jk = linalg::doublet(q_jv, q_kv, true, false); // (j b_{ijk} | k c_{ijk})
        auto K_ik = linalg::doublet(q_iv, q_kv, true, false); // (i a_{ijk} | k c_{ijk})
        auto K_ij = linalg::doublet(q_iv, q_jv, true, false); // (i a_{ijk} | j b_{ijk})

        // S integrals (semi-direct algorithm)
        std::vector<int> triples_ext_domain = merge_lists(lmo_to_paos_[i], merge_lists(lmo_to_paos_[j], lmo_to_paos_[k]));
        for (int l_ijk = 0; l_ijk < lmotriplet_to_lmos_[ijk].size(); ++l_ijk) {
            int l = lmotriplet_to_lmos_[ijk][l_ijk];
            triples_ext_domain = merge_lists(triples_ext_domain, lmo_to_paos_[l]);
        }
        auto S_ijk = submatrix_rows_and_cols(*S_pao_, triples_ext_domain, lmotriplet_to_paos_[ijk]);
        S_ijk = linalg::doublet(S_ijk, X_tno_[ijk], false, false);

        // => Step 1: Compute W_ijk (Jiang Eq. 109) <= //
        // W_{ijk}^{abc} = P_{ijk}^{abc}[(ia|bd)t_{kj}^{cd} - t_{il}^{ab}(jl|kc)]
        // P_{ijk}^{abc} is explicitly applied through the perms

        std::stringstream w_name;
        w_name << "W " << (ijk);
        auto W_ijk = std::make_shared<Matrix>(w_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
        W_ijk->zero();

        SharedMatrix L_ijk;
        if (lambda_requested_) {
            // L_ijk mirrors the W/V construction with converged Lambda1/2 in
            // place of the corresponding right-hand CCSD amplitudes.
            std::stringstream l_name;
            l_name << "L " << (ijk);
            L_ijk = std::make_shared<Matrix>(l_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
            L_ijk->zero();
        }

        std::vector<std::tuple<int, int, int>> perms = {std::make_tuple(i, j, k), std::make_tuple(i, k, j),
                                                        std::make_tuple(j, i, k), std::make_tuple(j, k, i),
                                                        std::make_tuple(k, i, j), std::make_tuple(k, j, i)};
        std::vector<SharedMatrix> Wperms(perms.size());
        std::vector<SharedMatrix> Lperms;
        if (lambda_requested_) Lperms.resize(perms.size());

        std::vector<SharedMatrix> K_ovvv_list = {K_ivvv, K_ivvv, K_jvvv, K_jvvv, K_kvvv, K_kvvv};
        std::vector<SharedMatrix> K_ooov_list = {K_jokv, K_kojv, K_iokv, K_koiv, K_iojv, K_joiv};

        if (thread == 0) timer_off("LCCSD(T0): Contract Integrals");

        if (thread == 0) timer_on("LCCSD(T0): Form W");

        for (int idx = 0; idx < perms.size(); ++idx) {
            int i, j, k;
            std::tie(i, j, k) = perms[idx];

            int jk = i_j_to_ij_[j][k];
            int kj = ij_to_ji_[jk];

            Wperms[idx] = std::make_shared<Matrix>(ntno_ijk, ntno_ijk * ntno_ijk);
            Wperms[idx]->zero();

            if (lambda_requested_) {
                Lperms[idx] = std::make_shared<Matrix>(ntno_ijk, ntno_ijk * ntno_ijk);
                Lperms[idx]->zero();
            }
            
            // Compute overlap between TNOs of triplet ijk and PNOs of pair kj
            std::vector<int> kj_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[kj]);
            auto S_kj_ijk = linalg::doublet(X_pno_[kj], submatrix_rows(*S_ijk, kj_idx_list), true, false);
            // (c_{kj}, d_{kj}) -> (c_{ijk}, d_{ijk})
            auto T_kj = linalg::triplet(S_kj_ijk, T_iajb_[kj], S_kj_ijk, true, false, false); 

            SharedMatrix lambda_kj;
            if (lambda_requested_) {
                // Project Lambda2 from its pair PNO space into the common TNO
                // space before forming this permutation of the left moment.
                lambda_kj = linalg::triplet(S_kj_ijk, lambda_iajb_[kj], S_kj_ijk, true, false, false);
            }

            auto K_ovvv = K_ovvv_list[idx]->clone(); // (i a | b d) stored as: (a, b * d)

            // Jiang Eq. 109a
            // W_{ijk}^{abc} += (ia|bd)t_{kj}^{cd}
            K_ovvv->reshape(ntno_ijk * ntno_ijk, ntno_ijk);  // (a, b * d) -> (a * b, d)
            K_ovvv = linalg::doublet(K_ovvv, T_kj, false, true); // (a * b, d) (c, d) -> (a * b, c)
            K_ovvv->reshape(ntno_ijk, ntno_ijk * ntno_ijk); // (a * b, c) -> (a, b * c)
            Wperms[idx]->add(K_ovvv);

            if (lambda_requested_) {
                // Left analogue of Jiang Eq. 109a: (ia|bd) lambda_kj^{cd}.
                auto K_ovvv_lambda = K_ovvv_list[idx]->clone(); // (i a | b d) stored as: (a, b * d)
                K_ovvv_lambda->reshape(ntno_ijk * ntno_ijk, ntno_ijk);  // (a, b * d) -> (a * b, d)
                K_ovvv_lambda = linalg::doublet(K_ovvv_lambda, lambda_kj, false,
                                                true); // (a * b, d) (c, d) -> (a * b, c)
                K_ovvv_lambda->reshape(ntno_ijk, ntno_ijk * ntno_ijk); // (a * b, c) -> (a, b * c)
                Lperms[idx]->add(K_ovvv_lambda);
            }

            for (int l_ijk = 0; l_ijk < lmotriplet_to_lmos_[ijk].size(); ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int il = i_j_to_ij_[i][l];

                // Compute overlap between TNOs of triplet ijk and PNOs of pair il
                std::vector<int> il_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[il]);
                auto S_il_ijk = linalg::doublet(X_pno_[il], submatrix_rows(*S_ijk, il_idx_list), true, false);
                // (a_{il}, b_{il}) -> (a_{ijk}, b_{ijk})
                auto T_il = linalg::triplet(S_il_ijk, T_iajb_[il], S_il_ijk, true, false, false);
                
                SharedMatrix lambda_il;
                if (lambda_requested_) {
                    // Left analogue of Jiang Eq. 109b uses the projected
                    // Lambda2 pair in place of T2.
                    lambda_il = linalg::triplet(S_il_ijk, lambda_iajb_[il], S_il_ijk, true, false, false);
                }


                // Jiang Eq. 109b
                // W_{ijk}^{abc} -= t_{il}^{ab}(jl|kc)
                for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
                    for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                        for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                            (*Wperms[idx])(a_ijk, b_ijk * ntno_ijk + c_ijk) -=
                                (*T_il)(a_ijk, b_ijk) * (*K_ooov_list[idx])(l_ijk, c_ijk); // (a, b) * (c) -> (a, b * c)
                            if (lambda_requested_) {
                                (*Lperms[idx])(a_ijk, b_ijk * ntno_ijk + c_ijk) -=
                                    (*lambda_il)(a_ijk, b_ijk) *
                                    (*K_ooov_list[idx])(l_ijk, c_ijk); // (a, b) * (c) -> (a, b * c)
                            }
                        }
                    }
                }  // end a_ijk
            }      // end l_ijk
        }

        // Apply P_{ijk}^{abc} to both the right numerator W and, when present,
        // the left moment L.
        // Reminder: P_{ijk}^{abc}X_{ijk}^{abc} =>
        // X_{ijk}^{abc} + X_{ikj}^{acb} + X_{jik}^{bac} + X_{jki}^{bca} + X_{kij}^{cab} + X_{kji}^{cba}
        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*W_ijk)(a_ijk, b_ijk *ntno_ijk + c_ijk) =
                        (*Wperms[0])(a_ijk, b_ijk * ntno_ijk + c_ijk) + (*Wperms[1])(a_ijk, c_ijk * ntno_ijk + b_ijk) +
                        (*Wperms[2])(b_ijk, a_ijk * ntno_ijk + c_ijk) + (*Wperms[3])(b_ijk, c_ijk * ntno_ijk + a_ijk) +
                        (*Wperms[4])(c_ijk, a_ijk * ntno_ijk + b_ijk) + (*Wperms[5])(c_ijk, b_ijk * ntno_ijk + a_ijk);

                    if (lambda_requested_) {
                        (*L_ijk)(a_ijk, b_ijk *ntno_ijk + c_ijk) =
                            (*Lperms[0])(a_ijk, b_ijk * ntno_ijk + c_ijk) + (*Lperms[1])(a_ijk, c_ijk * ntno_ijk + b_ijk) +
                            (*Lperms[2])(b_ijk, a_ijk * ntno_ijk + c_ijk) + (*Lperms[3])(b_ijk, c_ijk * ntno_ijk + a_ijk) +
                            (*Lperms[4])(c_ijk, a_ijk * ntno_ijk + b_ijk) + (*Lperms[5])(c_ijk, b_ijk * ntno_ijk + a_ijk);
                    }
                }
            }
        }

        if (thread == 0) timer_off("LCCSD(T0): Form W");

        if (thread == 0) timer_on("LCCSD(T0): Form V");

        // => Step 2: Compute V_ijk (Jiang Eq. 110) <= //
        // V_{ijk}^{abc} = W_{ijk}^{abc} + T_{i}^{a}(jb|kc) + T_{j}^{b}(ia|kc) + T_{k}^{c}(ia|jb)

        auto V_ijk = W_ijk->clone();
        std::stringstream v_name;
        v_name << "V " << (ijk);
        V_ijk->set_name(v_name.str());

        // Compute overlap between TNOs of triplet ijk and PNOs of pair ii, jj, and kk
        int ii = i_j_to_ij_[i][i];
        std::vector<int> ii_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[ii]);
        auto S_ii_ijk = linalg::doublet(X_pno_[ii], submatrix_rows(*S_ijk, ii_idx_list), true, false);

        int jj = i_j_to_ij_[j][j];
        std::vector<int> jj_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[jj]);
        auto S_jj_ijk = linalg::doublet(X_pno_[jj], submatrix_rows(*S_ijk, jj_idx_list), true, false);

        int kk = i_j_to_ij_[k][k];
        std::vector<int> kk_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[kk]);
        auto S_kk_ijk = linalg::doublet(X_pno_[kk], submatrix_rows(*S_ijk, kk_idx_list), true, false);

        // Transform singles amplitude to TNO space
        auto T_i = linalg::doublet(S_ii_ijk, T_ia_[i], true, false); // (i, a_{ii}) -> (i, a_{ijk})
        auto T_j = linalg::doublet(S_jj_ijk, T_ia_[j], true, false); // (j, b_{ii}) -> (j, b_{ijk})
        auto T_k = linalg::doublet(S_kk_ijk, T_ia_[k], true, false); // (k, c_{ii}) -> (k, c_{ijk})

        SharedMatrix lambda_i, lambda_j, lambda_k;
        if (lambda_requested_) {
            // Lambda1 projections supply the left counterparts of the T1
            // terms in the V moment.
            lambda_i = linalg::doublet(S_ii_ijk, lambda_ia_[i], true, false); // (i, a_{ii}) -> (i, a_{ijk})
            lambda_j = linalg::doublet(S_jj_ijk, lambda_ia_[j], true, false); // (j, b_{ii}) -> (j, b_{ijk})
            lambda_k = linalg::doublet(S_kk_ijk, lambda_ia_[k], true, false); // (k, c_{ii}) -> (k, c_{ijk})
        }

        // Compute Fia couplings (in case non-HF or Brueckner orbitals are used)
        auto Fia = submatrix_rows_and_cols(*F_lmo_pao_, std::vector<int>(1, i), lmotriplet_to_paos_[ijk]);
        Fia = linalg::doublet(Fia, X_tno_[ijk])->transpose();
        auto Fjb = submatrix_rows_and_cols(*F_lmo_pao_, std::vector<int>(1, j), lmotriplet_to_paos_[ijk]);
        Fjb = linalg::doublet(Fjb, X_tno_[ijk])->transpose();
        auto Fkc = submatrix_rows_and_cols(*F_lmo_pao_, std::vector<int>(1, k), lmotriplet_to_paos_[ijk]);
        Fkc = linalg::doublet(Fkc, X_tno_[ijk])->transpose();

        // TODO: Compute projected doubles amplitudes
        std::vector<int> ij_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[ij]);
        auto S_ij_ijk = linalg::doublet(X_pno_[ij], submatrix_rows(*S_ijk, ij_idx_list), true, false);
        auto T_ij = linalg::triplet(S_ij_ijk, T_iajb_[ij], S_ij_ijk, true, false, false);

        std::vector<int> jk_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[jk]);
        auto S_jk_ijk = linalg::doublet(X_pno_[jk], submatrix_rows(*S_ijk, jk_idx_list), true, false);
        auto T_jk = linalg::triplet(S_jk_ijk, T_iajb_[jk], S_jk_ijk, true, false, false);

        std::vector<int> ik_idx_list = index_list(triples_ext_domain, lmopair_to_paos_[ik]);
        auto S_ik_ijk = linalg::doublet(X_pno_[ik], submatrix_rows(*S_ijk, ik_idx_list), true, false);
        auto T_ik = linalg::triplet(S_ik_ijk, T_iajb_[ik], S_ik_ijk, true, false, false);

        SharedMatrix lambda_ij, lambda_jk, lambda_ik;
        if (lambda_requested_) {
            // The occupied-virtual Fock couplings multiply Lambda2 in the left
            // moment, just as they multiply T2 in the ordinary V moment.
            lambda_ij = linalg::triplet(S_ij_ijk, lambda_iajb_[ij], S_ij_ijk, true, false, false);
            lambda_jk = linalg::triplet(S_jk_ijk, lambda_iajb_[jk], S_jk_ijk, true, false, false);
            lambda_ik = linalg::triplet(S_ik_ijk, lambda_iajb_[ik], S_ik_ijk, true, false, false);
        }

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    // Now including the Fock terms
                    (*V_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) += (*T_i)(a_ijk, 0) * (*K_jk)(b_ijk, c_ijk) +
                        (*T_j)(b_ijk, 0) * (*K_ik)(a_ijk, c_ijk) + (*T_k)(c_ijk, 0) * (*K_ij)(a_ijk, b_ijk) +
                        (*Fia)(a_ijk, 0) * (*T_jk)(b_ijk, c_ijk) + (*Fjb)(b_ijk, 0) * (*T_ik)(a_ijk, c_ijk) + (*Fkc)(c_ijk, 0) * (*T_ij)(a_ijk, b_ijk);

                    if (lambda_requested_) {
                        // Complete the asymmetric left triples moment with its
                        // Lambda1-integral and F_ov-Lambda2 contributions.
                        (*L_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) += (*lambda_i)(a_ijk, 0) * (*K_jk)(b_ijk, c_ijk) +
                            (*lambda_j)(b_ijk, 0) * (*K_ik)(a_ijk, c_ijk) + (*lambda_k)(c_ijk, 0) * (*K_ij)(a_ijk, b_ijk) +
                            (*Fia)(a_ijk, 0) * (*lambda_jk)(b_ijk, c_ijk) + (*Fjb)(b_ijk, 0) * (*lambda_ik)(a_ijk, c_ijk)
                            + (*Fkc)(c_ijk, 0) * (*lambda_ij)(a_ijk, b_ijk);
                    }
                } // end c_ijk
            } // end b_ijk
        } // end a_ijk

        if (thread == 0) timer_off("LCCSD(T0): Form V");

        // Step 3: Compute T0 energy through amplitudes (Jiang Eq. 53)

        // T_{ijk}^{abc} = W_{ijk}^{abc} (\eps_{ijk}^{abc})^{-1} 
        // (initial semicanonical T3 amplitudes)
        auto T_ijk = W_ijk->clone();
        std::stringstream t_name;
        t_name << "T " << (ijk);
        T_ijk->set_name(t_name.str());

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*T_ijk)(a_ijk, b_ijk *ntno_ijk + c_ijk) =
                        -(*T_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) /
                        (e_tno_[ijk]->get(a_ijk) + e_tno_[ijk]->get(b_ijk) + e_tno_[ijk]->get(c_ijk) - (*F_lmo_)(i, i) -
                         (*F_lmo_)(j, j) - (*F_lmo_)(k, k));
                }
            }
        }

        /* E_{(T)} = prefactor * T_{ijk}^{abc} * (8 V_{ijk}^{abc} - 4 V_{ijk}^{bac} - 4 V_{ijk}^{acb}
                        - 4 V_{ijk}^{cab} + 2 V_{ijk}^{bca} + 2 V_{ijk}^{cab}) (Jiang Eq. 53) */

        double prefactor = 1.0;
        if (i == j && j == k) {
            prefactor /= 6.0;
        } else if (i == j || j == k || i == k) {
            prefactor /= 2.0;
        }

        double e_t0_ijk = 8.0 * prefactor * V_ijk->vector_dot(T_ijk);
        e_t0_ijk -= 4.0 * prefactor * triples_permuter(V_ijk, k, j, i)->vector_dot(T_ijk);
        e_t0_ijk -= 4.0 * prefactor * triples_permuter(V_ijk, i, k, j)->vector_dot(T_ijk);
        e_t0_ijk -= 4.0 * prefactor * triples_permuter(V_ijk, j, i, k)->vector_dot(T_ijk);
        e_t0_ijk += 2.0 * prefactor * triples_permuter(V_ijk, j, k, i)->vector_dot(T_ijk);
        e_t0_ijk += 2.0 * prefactor * triples_permuter(V_ijk, k, i, j)->vector_dot(T_ijk);
        E_T0 += e_t0_ijk;
        e_ijk_right_[ijk] = e_t0_ijk;

        if (lambda_requested_) {
            // Asymmetric triples contraction of Toth et al.:
            // E_(T)_L(ijk) = p_ijk <T_ijk, 8 L_ijk - 4 L_kji - 4 L_ikj
            //                                  - 4 L_jik + 2 L_jki + 2 L_kij>.
            e_ijk_[ijk] += 8.0 * prefactor * L_ijk->vector_dot(T_ijk);
            e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(L_ijk, k, j, i)->vector_dot(T_ijk);
            e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(L_ijk, i, k, j)->vector_dot(T_ijk);
            e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(L_ijk, j, i, k)->vector_dot(T_ijk);
            e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(L_ijk, j, k, i)->vector_dot(T_ijk);
            e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(L_ijk, k, i, j)->vector_dot(T_ijk);
            E_T_L0 += e_ijk_[ijk];
        } else {
            e_ijk_[ijk] = e_t0_ijk;
        }

        // Step 4: Save Matrices (if doing full (T))

        if (save_memory && !write_intermediates_) {
            W_iajbkc_[ijk] = W_ijk;
            V_iajbkc_[ijk] = V_ijk;
            if (lambda_requested_) L_iajbkc_[ijk] = L_ijk;
        } else if (save_memory && write_intermediates_) {
#pragma omp critical
            W_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
#pragma omp critical
            V_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
            if (lambda_requested_) {
#pragma omp critical
                L_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
            }
        }

        if (save_memory && !write_amplitudes_) {
            T_iajbkc_[ijk] = T_ijk;
        } else if (save_memory && write_amplitudes_) {
#pragma omp critical
            T_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
        }

        if (thread == 0) {
            std::time_t time_curr = std::time(nullptr);
            int time_elapsed = (int) time_curr - (int) time_lap;
            if (time_elapsed > 60) {
                outfile->Printf("  Time Elapsed from last checkpoint %4d (s), Progress %2d %%, Amplitudes for (%6d / %6d) Triplets Computed\n", time_elapsed, 
                                    (100 * ijk_idx) / n_lmo_triplets, ijk_idx, n_lmo_triplets);
                time_lap = std::time(nullptr);
            }
        }
    }

    timer_off("LCCSD(T0)");

    std::time_t time_stop = std::time(nullptr);
    int time_elapsed = (int) time_stop - (int) time_start;
    outfile->Printf(
        "    (Relevant) Semicanonical LCCSD(T0) Computation Complete!!! Time Elapsed: %4d seconds\n\n",
        time_elapsed);

    return std::make_pair(E_T0, E_T_L0);
}

double DLPNOCCSD_T::compute_t_iteration_energy() {
    /* Jiang Eq. 53 */
    /* E_{(T)} = prefactor * T_{ijk}^{abc} *(8 V_{ijk}^{abc} - 4 V_{ijk}^{bac} - 4 V_{ijk}^{acb}
                    - 4 V_{ijk}^{cab} + 2 V_{ijk}^{bca} + 2 V_{ijk}^{cab}) */

    timer_on("Compute (T) Energy");

    int n_lmo_triplets = ijk_to_i_j_k_.size();

    double E_T = 0.0;

#pragma omp parallel for schedule(dynamic) reduction(+ : E_T)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;

        double prefactor = 1.0;
        if (i == j && j == k) {
            prefactor /= 6.0;
        } else if (i == j || j == k || i == k) {
            prefactor /= 2.0;
        }

        SharedMatrix V_ijk;
        SharedMatrix T_ijk;

        // Grab V3 and T3 as needed
        if (write_intermediates_) {
            std::stringstream v_name;
            v_name << "V " << (ijk);
            V_ijk = std::make_shared<Matrix>(v_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
            V_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
        } else {
            V_ijk = V_iajbkc_[ijk];
        }

        if (write_amplitudes_) {
            std::stringstream t_name;
            t_name << "T " << (ijk);
            T_ijk = std::make_shared<Matrix>(t_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
            T_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
        } else {
            T_ijk = T_iajbkc_[ijk];
        }

        double e_triplet = 8.0 * prefactor * V_ijk->vector_dot(T_ijk);
        e_triplet -= 4.0 * prefactor * triples_permuter(V_ijk, k, j, i)->vector_dot(T_ijk);
        e_triplet -= 4.0 * prefactor * triples_permuter(V_ijk, i, k, j)->vector_dot(T_ijk);
        e_triplet -= 4.0 * prefactor * triples_permuter(V_ijk, j, i, k)->vector_dot(T_ijk);
        e_triplet += 2.0 * prefactor * triples_permuter(V_ijk, j, k, i)->vector_dot(T_ijk);
        e_triplet += 2.0 * prefactor * triples_permuter(V_ijk, k, i, j)->vector_dot(T_ijk);
        e_ijk_[ijk] = e_triplet;
        E_T += e_triplet;
    }

    timer_off("Compute (T) Energy");

    return E_T;
}

double DLPNOCCSD_T::compute_t_l_iteration_energy() {
    // Iterative asymmetric-triples energy of Toth et al. The spin-adapted
    // permutation functional is identical to the ordinary Jiang Eq. 53
    // contraction, but the right energy moment V is replaced by the left
    // Lambda-dependent moment L:
    // E_(T)_L = sum_ijk p_ijk <T_ijk, 8 L_ijk - 4 L_kji - 4 L_ikj
    //                                      - 4 L_jik + 2 L_jki + 2 L_kij>.

    timer_on("Compute (T)_L Energy");

    int n_lmo_triplets = ijk_to_i_j_k_.size();

    double E_T_L = 0.0;

#pragma omp parallel for schedule(dynamic) reduction(+ : E_T_L)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;

        double prefactor = 1.0;
        if (i == j && j == k) {
            prefactor /= 6.0;
        } else if (i == j || j == k || i == k) {
            prefactor /= 2.0;
        }

        SharedMatrix L_ijk;
        SharedMatrix T_ijk;

        // Grab L3 and T3 as needed
        if (write_intermediates_) {
            std::stringstream l_name;
            l_name << "L " << (ijk);
            L_ijk = std::make_shared<Matrix>(l_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
            L_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
        } else {
            L_ijk = L_iajbkc_[ijk];
        }

        if (write_amplitudes_) {
            std::stringstream t_name;
            t_name << "T " << (ijk);
            T_ijk = std::make_shared<Matrix>(t_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
            T_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
        } else {
            T_ijk = T_iajbkc_[ijk];
        }

        e_ijk_[ijk] = 8.0 * prefactor * L_ijk->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(L_ijk, k, j, i)->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(L_ijk, i, k, j)->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(L_ijk, j, i, k)->vector_dot(T_ijk);
        e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(L_ijk, j, k, i)->vector_dot(T_ijk);
        e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(L_ijk, k, i, j)->vector_dot(T_ijk);

        E_T_L += e_ijk_[ijk];
    }

    timer_off("Compute (T)_L Energy");

    return E_T_L;
}

SharedMatrix DLPNOCCSD_T::triples_permuter(const SharedMatrix &X, int i, int j, int k, bool reverse) {
    /* A helper function that permutes the indices of a triples amplitude based on ordering i, j, k.
        This is helpful for getting the true permutation of the virtual indices for triples that are
        not in i <= j <= k ordering */

    SharedMatrix Xperm = X->clone();
    int ntno_ijk = X->nrow();

    int perm_idx;
    if (i <= j && j <= k && i <= k) {
        perm_idx = 0;
    } else if (i <= k && k <= j && i <= j) {
        perm_idx = 1;
    } else if (j <= i && i <= k && j <= k) {
        perm_idx = 2;
    } else if (j <= k && k <= i && j <= i) {
        perm_idx = 3;
    } else if (k <= i && i <= j && k <= j) {
        perm_idx = 4;
    } else {
        perm_idx = 5;
    }

    for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
        for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
            for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                if (perm_idx == 0)
                    (*Xperm)(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*X)(a_ijk, b_ijk * ntno_ijk + c_ijk);
                else if (perm_idx == 1)
                    (*Xperm)(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*X)(a_ijk, c_ijk * ntno_ijk + b_ijk);
                else if (perm_idx == 2)
                    (*Xperm)(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*X)(b_ijk, a_ijk * ntno_ijk + c_ijk);
                else if ((perm_idx == 3 && !reverse) || (perm_idx == 4 && reverse))
                    (*Xperm)(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*X)(b_ijk, c_ijk * ntno_ijk + a_ijk);
                else if ((perm_idx == 4 && !reverse) || (perm_idx == 3 && reverse))
                    (*Xperm)(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*X)(c_ijk, a_ijk * ntno_ijk + b_ijk);
                else
                    (*Xperm)(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*X)(c_ijk, b_ijk * ntno_ijk + a_ijk);
            }
        }
    }

    return Xperm;
}

std::pair<double, double> DLPNOCCSD_T::lccsd_t_iterations() {
    timer_on("LCCSD(T) Iterations");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    if (n_lmo_triplets == 0) {
        timer_off("LCCSD(T) Iterations");
        return std::make_pair(0.0, 0.0);
    }

    outfile->Printf("\n  ==> Local CCSD(T) <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    if (lambda_requested_) {
        outfile->Printf("                         Corr. Energy    Lambda   Delta E     Max R     Time (s)\n");
    } else {
        outfile->Printf("                         Corr. Energy   Delta E     Max R     Time (s)\n");
    }

    int iteration = 1, max_iteration = options_.get_int("DLPNO_MAXITER");
    double e_curr = 0.0, e_t = 0.0, e_t_lambda = 0.0, e_prev = 0.0;
    bool e_converged = false, r_converged = false;

    double F_CUT = options_.get_double("F_CUT_T");
    double T_CUT_ITER = options_.get_double("T_CUT_ITER");

    std::vector<double> e_ijk_old(n_lmo_triplets, 0.0);

    // Sort Triplets by the approximate number of operations (for maximal parallel efficiency)
    std::vector<std::pair<int, size_t>> ijk_cost_tuple(n_lmo_triplets);
    
#pragma omp parallel for
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        size_t cost = 0;

        // Compute the cost of the TNO projections per triple
        for (int l = 0; l < naocc; ++l) {
            int ijl_dense = i * naocc * naocc + j * naocc + l;
            int ilk_dense = i * naocc * naocc + l * naocc + k;
            int ljk_dense = l * naocc * naocc + j * naocc + k;
            
            if (l != k && i_j_k_to_ijk_.count(ijl_dense) && std::fabs((*F_lmo_)(l, k)) >= F_CUT) {
                int ijl = i_j_k_to_ijk_[ijl_dense];
                cost += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijl];
                cost += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijl] * n_tno_[ijl];
                cost += n_tno_[ijk] * n_tno_[ijl] * n_tno_[ijl] * n_tno_[ijl];
            }
            
            if (l != j && i_j_k_to_ijk_.count(ilk_dense) && std::fabs((*F_lmo_)(l, j)) >= F_CUT) {
                int ilk = i_j_k_to_ijk_[ilk_dense];
                cost += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijk] * n_tno_[ilk];
                cost += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ilk] * n_tno_[ilk];
                cost += n_tno_[ijk] * n_tno_[ilk] * n_tno_[ilk] * n_tno_[ilk];
            }
            
            if (l != i && i_j_k_to_ijk_.count(ljk_dense) && std::fabs((*F_lmo_)(l, i)) >= F_CUT) {
                int ljk = i_j_k_to_ijk_[ljk_dense];
                cost += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijk] * n_tno_[ljk];
                cost += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ljk] * n_tno_[ljk];
                cost += n_tno_[ijk] * n_tno_[ljk] * n_tno_[ljk] * n_tno_[ljk];
            }
        }

        ijk_cost_tuple[ijk] = std::make_pair(ijk, cost);
    }
    
    std::sort(ijk_cost_tuple.begin(), ijk_cost_tuple.end(), [&](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b) {
        return (a.second > b.second);
    });

    std::vector<int> ijk_sorted_by_cost(n_lmo_triplets);
    
#pragma omp parallel for
    for (int ijk_idx = 0; ijk_idx < n_lmo_triplets; ++ijk_idx) {
        ijk_sorted_by_cost[ijk_idx] = ijk_cost_tuple[ijk_idx].first;
    }

    while (!(e_converged && r_converged)) {
        // RMS of residual per LMO triplet, for assessing convergence
        std::vector<double> R_iajbkc_rms(n_lmo_triplets, 0.0);

        std::time_t time_start = std::time(nullptr);

#pragma omp parallel for schedule(dynamic, 1)
        for (int ijk_idx = 0; ijk_idx < n_lmo_triplets; ++ijk_idx) {
            // Triplets assigned to threads dynamically, sorted in descending order of cost
            // This maximizes parallel efficiency
            int ijk = ijk_sorted_by_cost[ijk_idx];

            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

            int ntno_ijk = n_tno_[ijk];

            if (std::fabs(e_ijk_[ijk] - e_ijk_old[ijk]) < std::fabs(e_ijk_old[ijk] * T_CUT_ITER)) continue;

            // S integrals
            std::vector<int> triplet_ext_domain;
            for (int l = 0; l < naocc; ++l) {
                int ijl_dense = i * naocc * naocc + j * naocc + l;
                int ilk_dense = i * naocc * naocc + l * naocc + k;
                int ljk_dense = l * naocc * naocc + j * naocc + k;
                
                if (l != k && i_j_k_to_ijk_.count(ijl_dense) && std::fabs((*F_lmo_)(l, k)) >= F_CUT) {
                    int ijl = i_j_k_to_ijk_[ijl_dense];
                    triplet_ext_domain = merge_lists(triplet_ext_domain, lmotriplet_to_paos_[ijl]);
                }
                
                if (l != j && i_j_k_to_ijk_.count(ilk_dense) && std::fabs((*F_lmo_)(l, j)) >= F_CUT) {
                    int ilk = i_j_k_to_ijk_[ilk_dense];
                    triplet_ext_domain = merge_lists(triplet_ext_domain, lmotriplet_to_paos_[ilk]);
                }
                
                if (l != i && i_j_k_to_ijk_.count(ljk_dense) && std::fabs((*F_lmo_)(l, i)) >= F_CUT) {
                    int ljk = i_j_k_to_ijk_[ljk_dense];
                    triplet_ext_domain = merge_lists(triplet_ext_domain, lmotriplet_to_paos_[ljk]);
                    
                }
            }

            // Overlap integrals are formed using semi-direct algorithm
            auto S_ijk = submatrix_rows_and_cols(*S_pao_, triplet_ext_domain, lmotriplet_to_paos_[ijk]);
            S_ijk = linalg::doublet(S_ijk, X_tno_[ijk], false, false);

            auto R_ijk = std::make_shared<Matrix>("R_ijk", ntno_ijk, ntno_ijk * ntno_ijk);
            
            SharedMatrix W_ijk;
            SharedMatrix T_ijk;

            // Grab W3 and T3 as needed
            if (write_intermediates_) {
                std::stringstream w_name;
                w_name << "W " << (ijk);
                W_ijk = std::make_shared<Matrix>(w_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
                W_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
            } else {
                W_ijk = W_iajbkc_[ijk];
            }

            if (write_amplitudes_) {
                std::stringstream t_name;
                t_name << "T " << (ijk);
                T_ijk = std::make_shared<Matrix>(t_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
                T_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
            } else {
                T_ijk = T_iajbkc_[ijk];
            }

            // => Jiang Eq. 111 <= //
            // R_{ijk}^{abc} = W_{ijk}^{abc} + (typo in original work, plus, not minus)
            // T_{ijk}^{abc} (e_{a} + e_{b} + e_{c} - f_{ii} - f_{jj} - f_{kk})
            // - f_{il} t_{ljk}^{abc} - f_{jl} t_{ilk}^{abc} - f_{kl} t_{ijl}^{abc}
            R_ijk->copy(W_ijk);

            for (int a_ijk = 0; a_ijk < ntno_ijk; ++a_ijk) {
                for (int b_ijk = 0; b_ijk < ntno_ijk; ++b_ijk) {
                    for (int c_ijk = 0; c_ijk < ntno_ijk; ++c_ijk) {
                        (*R_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) += (*T_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) *
                            ((*e_tno_[ijk])(a_ijk) + (*e_tno_[ijk])(b_ijk) + (*e_tno_[ijk])(c_ijk) 
                                - (*F_lmo_)(i, i) - (*F_lmo_)(j, j) - (*F_lmo_)(k, k));
                    }
                }
            }

            // This inner loop performs the operation
            // R_{ijk}^{abc} += - f_{il} t_{ljk}^{abc} - f_{jl} t_{ilk}^{abc} - f_{kl} t_{ijl}^{abc}
            for (int l = 0; l < naocc; l++) {
                int ijl_dense = i * naocc * naocc + j * naocc + l;
                if (l != k && i_j_k_to_ijk_.count(ijl_dense) && std::fabs((*F_lmo_)(l, k)) >= F_CUT) {
                    int ijl = i_j_k_to_ijk_[ijl_dense];

                    // S(a_{ijk}, a_{ijl})
                    std::vector<int> ijl_idx_list = index_list(triplet_ext_domain, lmotriplet_to_paos_[ijl]);
                    auto S_ijk_ijl = linalg::doublet(submatrix_rows(*S_ijk, ijl_idx_list), X_tno_[ijl], true, false);

                    SharedMatrix T_ijl;
                    if (write_amplitudes_) {
                        std::stringstream t_name;
                        t_name << "T " << (ijl);
                        T_ijl = std::make_shared<Matrix>(t_name.str(), n_tno_[ijl], n_tno_[ijl] * n_tno_[ijl]);
#pragma omp critical
                        T_ijl->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                    } else {
                        T_ijl = T_iajbkc_[ijl];
                    }

                    // (a_{ijl}, b_{ijl}, c_{ijl}) -> (a_{ijk}, b_{ijk}, c_{ijk})
                    auto T_temp1 =
                        matmul_3d(triples_permuter(T_ijl, i, j, l), S_ijk_ijl, n_tno_[ijl], n_tno_[ijk]);
                    C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, k), &(*T_temp1)(0, 0), 1,
                            &(*R_ijk)(0, 0), 1);
                }

                int ilk_dense = i * naocc * naocc + l * naocc + k;
                if (l != j && i_j_k_to_ijk_.count(ilk_dense) && std::fabs((*F_lmo_)(l, j)) >= F_CUT) {
                    int ilk = i_j_k_to_ijk_[ilk_dense];

                    // S(a_{ijk}, a_{ilk})
                    std::vector<int> ilk_idx_list = index_list(triplet_ext_domain, lmotriplet_to_paos_[ilk]);
                    auto S_ijk_ilk = linalg::doublet(submatrix_rows(*S_ijk, ilk_idx_list), X_tno_[ilk], true, false);

                    SharedMatrix T_ilk;
                    if (write_amplitudes_) {
                        std::stringstream t_name;
                        t_name << "T " << (ilk);
                        T_ilk = std::make_shared<Matrix>(t_name.str(), n_tno_[ilk], n_tno_[ilk] * n_tno_[ilk]);
#pragma omp critical
                        T_ilk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                    } else {
                        T_ilk = T_iajbkc_[ilk];
                    }

                    // (a_{ilk}, b_{ilk}, c_{ilk}) -> (a_{ijk}, b_{ijk}, c_{ijk})
                    auto T_temp1 =
                        matmul_3d(triples_permuter(T_ilk, i, l, k), S_ijk_ilk, n_tno_[ilk], n_tno_[ijk]);
                    C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, j), &(*T_temp1)(0, 0), 1,
                            &(*R_ijk)(0, 0), 1);
                }

                int ljk_dense = l * naocc * naocc + j * naocc + k;
                if (l != i && i_j_k_to_ijk_.count(ljk_dense) && std::fabs((*F_lmo_)(l, i)) >= F_CUT) {
                    int ljk = i_j_k_to_ijk_[ljk_dense];

                    // S(a_{ijk}, a_{ljk})
                    std::vector<int> ljk_idx_list = index_list(triplet_ext_domain, lmotriplet_to_paos_[ljk]);
                    auto S_ijk_ljk = linalg::doublet(submatrix_rows(*S_ijk, ljk_idx_list), X_tno_[ljk], true, false);

                    SharedMatrix T_ljk;
                    if (write_amplitudes_) {
                        std::stringstream t_name;
                        t_name << "T " << (ljk);
                        T_ljk = std::make_shared<Matrix>(t_name.str(), n_tno_[ljk], n_tno_[ljk] * n_tno_[ljk]);
#pragma omp critical
                        T_ljk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                    } else {
                        T_ljk = T_iajbkc_[ljk];
                    }

                    // (a_{ljk}, b_{ljk}, c_{ljk}) -> (a_{ijk}, b_{ijk}, c_{ijk})
                    auto T_temp1 =
                        matmul_3d(triples_permuter(T_ljk, l, j, k), S_ijk_ljk, n_tno_[ljk], n_tno_[ijk]);
                    C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, i), &(*T_temp1)(0, 0), 1,
                            &(*R_ijk)(0, 0), 1);
                }
            }

            // => Update T3 Amplitudes (Jiang Eq. 112) <= //
            for (int a_ijk = 0; a_ijk < ntno_ijk; ++a_ijk) {
                for (int b_ijk = 0; b_ijk < ntno_ijk; ++b_ijk) {
                    for (int c_ijk = 0; c_ijk < ntno_ijk; ++c_ijk) {
                        (*T_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) -= (*R_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) /
                            ((*e_tno_[ijk])(a_ijk) + (*e_tno_[ijk])(b_ijk) + (*e_tno_[ijk])(c_ijk) 
                                - (*F_lmo_)(i, i) - (*F_lmo_)(j, j) - (*F_lmo_)(k, k));
                    }
                }
            }

            if (write_amplitudes_) {
#pragma omp critical
                T_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
            }
            
            R_iajbkc_rms[ijk] = R_ijk->rms();
        }

        // evaluate convergence
        e_prev = e_curr;
        e_ijk_old = e_ijk_;
        // Compute LCCSD(T) energy
        e_t = compute_t_iteration_energy();
        if (lambda_requested_) e_t_lambda = compute_t_l_iteration_energy();
        e_curr = lambda_requested_ ? e_t_lambda : e_t;

        double r_curr = *max_element(R_iajbkc_rms.begin(), R_iajbkc_rms.end());

        r_converged = fabs(r_curr) < options_.get_double("R_CONVERGENCE");
        e_converged = fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE");

        std::time_t time_stop = std::time(nullptr);

        if (lambda_requested_) {
            outfile->Printf("  @LCCSD(T) iter %3d: %16.12f %16.12f %10.3e %10.3e %8d\n", iteration, e_t,
                            e_t_lambda, e_curr - e_prev, r_curr, (int)time_stop - (int)time_start);
        } else {
            outfile->Printf("  @LCCSD(T) iter %3d: %16.12f %10.3e %10.3e %8d\n", iteration, e_t,
                            e_curr - e_prev, r_curr, (int)time_stop - (int)time_start);
        }

        iteration++;

        if (iteration > max_iteration + 1) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }
    }

    timer_off("LCCSD(T) Iterations");

    return std::make_pair(e_t, e_t_lambda);
}

void DLPNOCCSD_T::compute_triples_correction(DLPNOCCSDPhase phase) {
    timer_on("DLPNO-CCSD(T)");

    if (lambda_requested_ && !lambda_solved_) {
        throw PSIEXCEPTION("DLPNO-CCSD(T)_L requires converged Lambda amplitudes.");
    }

    const bool bccd_result = brueckner_orbs_ && phase == DLPNOCCSDPhase::FinalBrueckner;
    const std::string reference_method = bccd_result ? "BCCD" : "CCSD";
    const std::string right_method = reference_method + "(T)";
    const std::string left_method = "A-" + reference_method + "(T)";

    print_header(phase);

    const int naocc = nalpha_ - nfrzc();
    const int n_lmo_pairs = ij_to_i_j_.size();

    // Convert the solver's spin-adapted Lambda convention to the amplitudes
    // used in the Toth et al. asymmetric triples moment:
    //     Lambda1 <- Lambda1 / 2
    //     Lambda2_ij <- Lambda2_ij / 3 + (Lambda2_ij)^T / 6.
    // No Lambda objects exist for ordinary (T), so this work and its memory
    // cost are completely avoided there.
    if (lambda_requested_) {
#pragma omp parallel for
        for (int i = 0; i < naocc; ++i) {
            lambda_ia_[i]->scale(0.5);
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto lambda_buffer_a = lambda_iajb_[ij]->clone();
            lambda_buffer_a->scale(1.0 / 3.0);
            auto lambda_buffer_b = lambda_iajb_[ij]->transpose();
            lambda_buffer_b->scale(1.0 / 6.0);

            lambda_iajb_[ij] = lambda_buffer_a;
            lambda_iajb_[ij]->add(lambda_buffer_b);
        }
    }

    // Clear integral storage that is no longer required after the CCSD and,
    // when requested, Lambda iterations.
    K_mibj_.clear();
    J_ijmb_.clear();
    L_mibj_.clear();
    L_iajb_.clear();
    J_ikac_non_proj_.clear();
    K_iakc_non_proj_.clear();
    K_ivvv_.clear();
    Qma_ij_.clear();
    Qab_ij_.clear();
    i_Qk_ij_.clear();
    i_Qa_ij_.clear();
    i_Qk_t1_.clear();
    i_Qa_t1_.clear();
    S_pno_ij_kj_.clear();
    S_pno_ij_nn_.clear();
    S_pno_ij_mn_.clear();
    Fkc_.clear();
    Fai_.clear();
    Fab_.clear();
    T_n_ij_.clear();

    // A Brueckner calculation evaluates triples twice, so discard every
    // triplet-dependent object before constructing the next orbital phase.
    lmotriplet_to_ribfs_.clear();
    lmotriplet_to_lmos_.clear();
    lmotriplet_to_paos_.clear();
    i_j_k_to_ijk_.clear();
    ijk_to_i_j_k_.clear();
    W_iajbkc_.clear();
    V_iajbkc_.clear();
    L_iajbkc_.clear();
    T_iajbkc_.clear();
    X_tno_.clear();
    e_tno_.clear();
    n_tno_.clear();
    e_ijk_.clear();
    e_ijk_right_.clear();
    tno_scale_.clear();
    is_strong_triplet_.clear();
    de_lccsd_t_screened_ = 0.0;
    de_lccsd_t_l_screened_ = 0.0;
    e_lccsd_t_ = 0.0;
    E_T_ = 0.0;
    e_lccsd_t_l_ = 0.0;
    E_T_L_ = 0.0;

    psio_->open(PSIF_DLPNO_TRIPLES, PSIO_OPEN_NEW);

    const double t_cut_tno_pre = options_.get_double("T_CUT_TNO_PRE");
    const double t_cut_tno = options_.get_double("T_CUT_TNO");

    // Step 1: Perform triplet prescreening.
    outfile->Printf("\n   Starting Triplet Prescreening...\n");
    outfile->Printf("     T_CUT_TNO set to %6.3e \n", t_cut_tno_pre);
    outfile->Printf("     T_CUT_DO  set to %6.3e \n", options_.get_double("T_CUT_DO_TRIPLES_PRE"));
    outfile->Printf("     T_CUT_MKN set to %6.3e \n\n", options_.get_double("T_CUT_MKN_TRIPLES_PRE"));

    triples_sparsity(true);
    tno_transform(t_cut_tno_pre);
    compute_lccsd_t0();

    // Step 2: Compute DLPNO-CCSD(T0) with the surviving triplets.
    outfile->Printf("\n   Continuing computation with surviving triplets...\n");
    outfile->Printf("     Eliminated all triples with energy less than %6.3e Eh... \n\n",
                    options_.get_double("T_CUT_TRIPLES_WEAK"));
    triples_sparsity(false);
    outfile->Printf("    * Ordinary Energy From Screened Triplets:   %.12f \n", de_lccsd_t_screened_);
    if (lambda_requested_) {
        outfile->Printf("    * Asymmetric Energy From Screened Triplets: %.12f \n", de_lccsd_t_l_screened_);
    }
    outfile->Printf("\n");

    outfile->Printf("     T_CUT_TNO (re)set to %6.3e \n", t_cut_tno);
    outfile->Printf("     T_CUT_DO  (re)set to %6.3e \n", options_.get_double("T_CUT_DO_TRIPLES"));
    outfile->Printf("     T_CUT_MKN (re)set to %6.3e \n\n", options_.get_double("T_CUT_MKN_TRIPLES"));

    tno_transform(t_cut_tno);
    auto [E_T0, E_T_L0] = compute_lccsd_t0();
    e_lccsd_t_ = e_lccsd_ + E_T0 + de_lccsd_t_screened_;
    if (lambda_requested_) e_lccsd_t_l_ = e_lccsd_ + E_T_L0 + de_lccsd_t_l_screened_;

    outfile->Printf("    DLPNO-%s(T0) Correlation Energy: %16.12f \n", reference_method.c_str(), e_lccsd_t_);
    outfile->Printf("    * DLPNO-%s Contribution:         %16.12f \n", reference_method.c_str(), e_lccsd_);
    outfile->Printf("    * DLPNO-(T0) Contribution:       %16.12f \n", E_T0);
    outfile->Printf("    * Screened Triplets Contribution:%16.12f \n\n", de_lccsd_t_screened_);

    if (lambda_requested_) {
        outfile->Printf("    DLPNO-%s(T0)_L Correlation Energy: %16.12f \n", reference_method.c_str(), e_lccsd_t_l_);
        outfile->Printf("    * DLPNO-%s Contribution:           %16.12f \n", reference_method.c_str(), e_lccsd_);
        outfile->Printf("    * DLPNO-(T0)_L Contribution:       %16.12f \n", E_T_L0);
        outfile->Printf("    * Screened Triplets Contribution:  %16.12f \n\n", de_lccsd_t_l_screened_);
    }

    // Step 3: Compute the full iterative correction unless (T0) was requested.
    if (!options_.get_bool("T0_APPROXIMATION")) {
        outfile->Printf("\n\n  ==> Computing Full Iterative (T) <==\n\n");

        sort_triplets(lambda_requested_ ? E_T_L0 : E_T0);

        const double t_cut_tno_strong_scale = options_.get_double("T_CUT_TNO_STRONG_SCALE");
        const double t_cut_tno_weak_scale = options_.get_double("T_CUT_TNO_WEAK_SCALE");
        outfile->Printf("     T_CUT_TNO (re)set to %6.3e for strong triples \n", t_cut_tno * t_cut_tno_strong_scale);
        outfile->Printf("     T_CUT_TNO (re)set to %6.3e for weak triples   \n\n", t_cut_tno * t_cut_tno_weak_scale);

        tno_transform(t_cut_tno);
        estimate_triples_memory();

        auto [E_T0_crude, E_T_L0_crude] = compute_lccsd_t0(true);
        auto [E_T, E_T_L] = lccsd_t_iterations();
        E_T_ = E_T;
        E_T_L_ = E_T_L;

        const double dE_T = E_T_ - E_T0_crude;
        const double dE_T_L = E_T_L_ - E_T_L0_crude;

        outfile->Printf("\n");
        outfile->Printf("    DLPNO-%s(T0) energy at looser tolerance: %16.12f\n", reference_method.c_str(), E_T0_crude);
        outfile->Printf("    DLPNO-%s(T)  energy at looser tolerance: %16.12f\n", reference_method.c_str(), E_T_);
        outfile->Printf("    * Net Iterative (T) contribution:        %16.12f\n\n", dE_T);
        e_lccsd_t_ += dE_T;

        if (lambda_requested_) {
            outfile->Printf("\n");
            outfile->Printf("    DLPNO-%s(T0)_L energy at looser tolerance: %16.12f\n",
                            reference_method.c_str(), E_T_L0_crude);
            outfile->Printf("    DLPNO-%s(T)_L  energy at looser tolerance: %16.12f\n", reference_method.c_str(),
                            E_T_L_);
            outfile->Printf("    * Net Iterative (T)_L contribution:       %16.12f\n\n", dE_T_L);
            e_lccsd_t_l_ += dE_T_L;
        }
    }

    const double e_scf = scalar_variable("SCF TOTAL ENERGY");
    const double right_corr = e_lccsd_t_ + de_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;
    const double right_total = e_scf + right_corr;
    const double right_correction = e_lccsd_t_ - e_lccsd_;

    set_scalar_variable(right_method + " CORRELATION ENERGY", right_corr);
    set_scalar_variable(right_method + " TOTAL ENERGY", right_total);
    set_scalar_variable("DLPNO-" + right_method + " CORRELATION ENERGY", right_corr);
    set_scalar_variable("DLPNO-" + right_method + " TOTAL ENERGY", right_total);
    set_scalar_variable("(T) CORRECTION ENERGY", right_correction);
    if (bccd_result) set_scalar_variable("B(T) CORRECTION ENERGY", right_correction);

    set_scalar_variable("DLPNO SEMICANONICAL (T0) ENERGY", E_T0 + de_lccsd_t_screened_);
    set_scalar_variable("DLPNO SCREENED TRIPLETS ENERGY", de_lccsd_t_screened_);
    if (lambda_requested_) {
        set_scalar_variable("DLPNO ASYMMETRIC SCREENED TRIPLETS ENERGY", de_lccsd_t_l_screened_);
    }

    if (phase == DLPNOCCSDPhase::InitialBrueckner) {
        set_scalar_variable("INITIAL DLPNO-CCSD(T) CORRELATION ENERGY", right_corr);
        set_scalar_variable("INITIAL DLPNO-CCSD(T) TOTAL ENERGY", right_total);
        set_scalar_variable("INITIAL DLPNO-(T) CORRECTION ENERGY", right_correction);
    }

    if (lambda_requested_) {
        const double left_corr = e_lccsd_t_l_ + de_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;
        const double left_total = e_scf + left_corr;
        const double left_correction = e_lccsd_t_l_ - e_lccsd_;
        const std::string dlpno_left_method = "DLPNO-" + reference_method + "(T)_L";
        const std::string at_method = reference_method + "(AT)";

        set_scalar_variable(left_method + " CORRELATION ENERGY", left_corr);
        set_scalar_variable(left_method + " TOTAL ENERGY", left_total);
        set_scalar_variable(at_method + " CORRELATION ENERGY", left_corr);
        set_scalar_variable(at_method + " TOTAL ENERGY", left_total);
        set_scalar_variable(dlpno_left_method + " CORRELATION ENERGY", left_corr);
        set_scalar_variable(dlpno_left_method + " TOTAL ENERGY", left_total);
        set_scalar_variable("DLPNO-" + at_method + " CORRELATION ENERGY", left_corr);
        set_scalar_variable("DLPNO-" + at_method + " TOTAL ENERGY", left_total);
        set_scalar_variable("A-(T) CORRECTION ENERGY", left_correction);

        if (phase == DLPNOCCSDPhase::InitialBrueckner) {
            set_scalar_variable("INITIAL DLPNO-CCSD(T)_L CORRELATION ENERGY", left_corr);
            set_scalar_variable("INITIAL DLPNO-CCSD(T)_L TOTAL ENERGY", left_total);
            set_scalar_variable("INITIAL DLPNO-(T)_L CORRECTION ENERGY", left_correction);
        }

        set_scalar_variable("CURRENT CORRELATION ENERGY", left_corr);
        set_scalar_variable("CURRENT ENERGY", left_total);
    } else {
        set_scalar_variable("CURRENT CORRELATION ENERGY", right_corr);
        set_scalar_variable("CURRENT ENERGY", right_total);
    }

    print_results(phase);
    psio_->close(PSIF_DLPNO_TRIPLES, 0);
    timer_off("DLPNO-CCSD(T)");
}

void DLPNOCCSD_T::post_ccsd_correction(DLPNOCCSDPhase phase) {
    if (lambda_requested_ && phase == DLPNOCCSDPhase::InitialBrueckner) {
        // The first Brueckner macroiteration is intentionally the ordinary
        // reference-orbital CCSD(T)_L result, including Lambda singles. At the
        // final Brueckner point, the base hook omits Lambda1 by stationarity.
        solve_lambda(true);
    } else {
        DLPNOCCSD::post_ccsd_correction(phase);
    }

    compute_triples_correction(phase);
}

double DLPNOCCSD_T::compute_energy() {
    DLPNOCCSD::compute_energy();
    return scalar_variable("CURRENT ENERGY");
}

void DLPNOCCSD_T::print_results(DLPNOCCSDPhase phase) {
    const bool bccd_result = brueckner_orbs_ && phase == DLPNOCCSDPhase::FinalBrueckner;
    const std::string reference_method = bccd_result ? "BCCD" : "CCSD";
    const std::string triples_suffix = options_.get_bool("T0_APPROXIMATION") ? "(T0)" : "(T)";
    const std::string right_method = "DLPNO-" + reference_method + triples_suffix;
    const double reference_corr = e_lccsd_ + de_weak_ + de_lmp2_eliminated_ + de_pno_total_ + de_dipole_;
    const double right_corr = e_lccsd_t_ + de_weak_ + de_lmp2_eliminated_ + de_pno_total_ + de_dipole_;
    const double right_total = scalar_variable("SCF TOTAL ENERGY") + right_corr;

    outfile->Printf("  \n");
    outfile->Printf("  Total %s Correlation Energy: %16.12f \n", right_method.c_str(), right_corr);
    outfile->Printf("    DLPNO-%s Contribution:       %16.12f \n", reference_method.c_str(), reference_corr);
    outfile->Printf("    DLPNO-%s Contribution:          %16.12f \n", triples_suffix.c_str(),
                    e_lccsd_t_ - e_lccsd_ - de_lccsd_t_screened_);
    outfile->Printf("    Screened Triplets Contribution: %16.12f \n", de_lccsd_t_screened_);
    outfile->Printf("\n\n  @Total %s Energy: %16.12f \n", right_method.c_str(), right_total);
    outfile->Printf("    *** Andy Jiang... FOR THREEEEEEEEEEE!!!\n\n");

    if (lambda_requested_) {
        const std::string left_method = "DLPNO-" + reference_method + "(T)_L";
        const double left_corr = e_lccsd_t_l_ + de_weak_ + de_lmp2_eliminated_ + de_pno_total_ + de_dipole_;
        const double left_total = scalar_variable("SCF TOTAL ENERGY") + left_corr;

        outfile->Printf("  \n");
        outfile->Printf("  Total %s Correlation Energy: %16.12f \n", left_method.c_str(), left_corr);
        outfile->Printf("    DLPNO-%s Contribution:         %16.12f \n", reference_method.c_str(), reference_corr);
        outfile->Printf("    DLPNO-(T)_L Contribution:      %16.12f \n",
                        e_lccsd_t_l_ - e_lccsd_ - de_lccsd_t_l_screened_);
        outfile->Printf("    Screened Triplets Contribution:%16.12f \n", de_lccsd_t_l_screened_);
        outfile->Printf("\n\n  @Total %s Energy: %16.12f \n", left_method.c_str(), left_total);
        outfile->Printf("    *** Oh yeah... it's all coming together -Zizi\n\n");
    }
}

#ifdef USING_Einsums

DLPNOCCSD_cT::DLPNOCCSD_cT(SharedWavefunction ref_wfn, Options& options) : DLPNOCCSD_T(ref_wfn, options) {}
DLPNOCCSD_cT::~DLPNOCCSD_cT() = default;

void DLPNOCCSD_cT::project_triplet_singles() {
    const int n_lmo_triplets = static_cast<int>(ijk_to_i_j_k_.size());
    T_n_ijk_.resize(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        const int nlmo_ijk = static_cast<int>(lmotriplet_to_lmos_[ijk].size());
        const int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;
        T_n_ijk_[ijk] = Tensor<double, 2>("T_n_ijk", nlmo_ijk, ntno_ijk);

        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            const int l = lmotriplet_to_lmos_[ijk][l_ijk];
            const int ll = i_j_to_ij_[l][l];
            auto S_ijk_ll =
                submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ll]);
            S_ijk_ll = linalg::triplet(X_tno_[ijk], S_ijk_ll, X_pno_[ll], true, false, false);
            auto T_l = linalg::doublet(S_ijk_ll, T_ia_[l]);
            std::memcpy(&T_n_ijk_[ijk](l_ijk, 0), T_l->get_pointer(), ntno_ijk * sizeof(double));
        }
    }
}

void DLPNOCCSD_cT::compute_ct_integrals() {
    const int n_lmo_triplets = static_cast<int>(ijk_to_i_j_k_.size());
    q_io_.resize(n_lmo_triplets);
    q_jo_.resize(n_lmo_triplets);
    q_ko_.resize(n_lmo_triplets);
    q_iv_.resize(n_lmo_triplets);
    q_jv_.resize(n_lmo_triplets);
    q_kv_.resize(n_lmo_triplets);
    q_ov_.resize(n_lmo_triplets);
    q_vv_.resize(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        const int naux_ijk = static_cast<int>(lmotriplet_to_ribfs_[ijk].size());
        const int nlmo_ijk = static_cast<int>(lmotriplet_to_lmos_[ijk].size());
        const int npao_ijk = static_cast<int>(lmotriplet_to_paos_[ijk].size());
        const int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;

        auto q_io = std::make_shared<Matrix>("(Q_ijk | m i)", naux_ijk, nlmo_ijk);
        auto q_jo = std::make_shared<Matrix>("(Q_ijk | m j)", naux_ijk, nlmo_ijk);
        auto q_ko = std::make_shared<Matrix>("(Q_ijk | m k)", naux_ijk, nlmo_ijk);
        auto q_iv = std::make_shared<Matrix>("(Q_ijk | i a)", naux_ijk, npao_ijk);
        auto q_jv = std::make_shared<Matrix>("(Q_ijk | j a)", naux_ijk, npao_ijk);
        auto q_kv = std::make_shared<Matrix>("(Q_ijk | k a)", naux_ijk, npao_ijk);
        auto q_ov = std::make_shared<Matrix>("(Q_ijk | m a)", naux_ijk, nlmo_ijk * ntno_ijk);
        auto q_vv = std::make_shared<Matrix>("(Q_ijk | a b)", naux_ijk, ntno_ijk * ntno_ijk);

        for (int q_ijk = 0; q_ijk < naux_ijk; ++q_ijk) {
            const int q = lmotriplet_to_ribfs_[ijk][q_ijk];
            const int center_q = ribasis_->function_to_center(q);

            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                const int l = lmotriplet_to_lmos_[ijk][l_ijk];
                (*q_io)(q_ijk, l_ijk) =
                    (*qij_[q])(riatom_to_lmos_ext_dense_[center_q][i], riatom_to_lmos_ext_dense_[center_q][l]);
                (*q_jo)(q_ijk, l_ijk) =
                    (*qij_[q])(riatom_to_lmos_ext_dense_[center_q][j], riatom_to_lmos_ext_dense_[center_q][l]);
                (*q_ko)(q_ijk, l_ijk) =
                    (*qij_[q])(riatom_to_lmos_ext_dense_[center_q][k], riatom_to_lmos_ext_dense_[center_q][l]);
            }

            for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                const int u = lmotriplet_to_paos_[ijk][u_ijk];
                (*q_iv)(q_ijk, u_ijk) =
                    (*qia_[q])(riatom_to_lmos_ext_dense_[center_q][i], riatom_to_paos_ext_dense_[center_q][u]);
                (*q_jv)(q_ijk, u_ijk) =
                    (*qia_[q])(riatom_to_lmos_ext_dense_[center_q][j], riatom_to_paos_ext_dense_[center_q][u]);
                (*q_kv)(q_ijk, u_ijk) =
                    (*qia_[q])(riatom_to_lmos_ext_dense_[center_q][k], riatom_to_paos_ext_dense_[center_q][u]);
            }

            auto q_ov_pao = std::make_shared<Matrix>(nlmo_ijk, npao_ijk);
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                const int l = lmotriplet_to_lmos_[ijk][l_ijk];
                for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                    const int u = lmotriplet_to_paos_[ijk][u_ijk];
                    (*q_ov_pao)(l_ijk, u_ijk) =
                        (*qia_[q])(riatom_to_lmos_ext_dense_[center_q][l], riatom_to_paos_ext_dense_[center_q][u]);
                }
            }
            q_ov_pao = linalg::doublet(q_ov_pao, X_tno_[ijk]);
            std::memcpy(&(*q_ov)(q_ijk, 0), q_ov_pao->get_pointer(), nlmo_ijk * ntno_ijk * sizeof(double));

            auto q_vv_pao = std::make_shared<Matrix>(npao_ijk, npao_ijk);
            for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                const int u = lmotriplet_to_paos_[ijk][u_ijk];
                for (int v_ijk = 0; v_ijk < npao_ijk; ++v_ijk) {
                    const int v = lmotriplet_to_paos_[ijk][v_ijk];
                    const int uv = riatom_to_pao_pairs_dense_[center_q][u][v];
                    if (uv != -1) (*q_vv_pao)(u_ijk, v_ijk) = (*qab_[q])(uv, 0);
                }
            }
            q_vv_pao = linalg::triplet(X_tno_[ijk], q_vv_pao, X_tno_[ijk], true, false, false);
            std::memcpy(&(*q_vv)(q_ijk, 0), q_vv_pao->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
        }

        q_iv = linalg::doublet(q_iv, X_tno_[ijk]);
        q_jv = linalg::doublet(q_jv, X_tno_[ijk]);
        q_kv = linalg::doublet(q_kv, X_tno_[ijk]);

        auto metric = submatrix_rows_and_cols(*full_metric_, lmotriplet_to_ribfs_[ijk], lmotriplet_to_ribfs_[ijk]);
        metric->power(-0.5, 1.0e-14);
        q_io = linalg::doublet(metric, q_io);
        q_jo = linalg::doublet(metric, q_jo);
        q_ko = linalg::doublet(metric, q_ko);
        q_iv = linalg::doublet(metric, q_iv);
        q_jv = linalg::doublet(metric, q_jv);
        q_kv = linalg::doublet(metric, q_kv);
        q_ov = linalg::doublet(metric, q_ov);
        q_vv = linalg::doublet(metric, q_vv);

        q_io_[ijk] = Tensor<double, 2>("q_io", naux_ijk, nlmo_ijk);
        q_jo_[ijk] = Tensor<double, 2>("q_jo", naux_ijk, nlmo_ijk);
        q_ko_[ijk] = Tensor<double, 2>("q_ko", naux_ijk, nlmo_ijk);
        q_iv_[ijk] = Tensor<double, 2>("q_iv", naux_ijk, ntno_ijk);
        q_jv_[ijk] = Tensor<double, 2>("q_jv", naux_ijk, ntno_ijk);
        q_kv_[ijk] = Tensor<double, 2>("q_kv", naux_ijk, ntno_ijk);
        q_ov_[ijk] = Tensor<double, 3>("q_ov", naux_ijk, nlmo_ijk, ntno_ijk);
        q_vv_[ijk] = Tensor<double, 3>("q_vv", naux_ijk, ntno_ijk, ntno_ijk);
        std::memcpy(q_io_[ijk].data(), q_io->get_pointer(), naux_ijk * nlmo_ijk * sizeof(double));
        std::memcpy(q_jo_[ijk].data(), q_jo->get_pointer(), naux_ijk * nlmo_ijk * sizeof(double));
        std::memcpy(q_ko_[ijk].data(), q_ko->get_pointer(), naux_ijk * nlmo_ijk * sizeof(double));
        std::memcpy(q_iv_[ijk].data(), q_iv->get_pointer(), naux_ijk * ntno_ijk * sizeof(double));
        std::memcpy(q_jv_[ijk].data(), q_jv->get_pointer(), naux_ijk * ntno_ijk * sizeof(double));
        std::memcpy(q_kv_[ijk].data(), q_kv->get_pointer(), naux_ijk * ntno_ijk * sizeof(double));
        std::memcpy(q_ov_[ijk].data(), q_ov->get_pointer(), naux_ijk * nlmo_ijk * ntno_ijk * sizeof(double));
        std::memcpy(q_vv_[ijk].data(), q_vv->get_pointer(), naux_ijk * ntno_ijk * ntno_ijk * sizeof(double));
    }
}

SharedMatrix DLPNOCCSD_cT::build_ct_moment(int ijk) {
    int i, j, k;
    std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
    const int ij = i_j_to_ij_[i][j];
    const int jk = i_j_to_ij_[j][k];
    const int ik = i_j_to_ij_[i][k];
    const int ntno_ijk = n_tno_[ijk];
    const int naux_ijk = static_cast<int>(lmotriplet_to_ribfs_[ijk].size());
    const int nlmo_ijk = static_cast<int>(lmotriplet_to_lmos_[ijk].size());

    auto M_ijk = std::make_shared<Matrix>("M(cT)", ntno_ijk, ntno_ijk * ntno_ijk);

    // This is the T3-independent part of
    // DLPNOCCSDT::compute_R_iajbkc() on higher_order_dlpno_112.  At T3 = 0,
    // only the long-permutation W source survives; every short-permutation V
    // term and the cross-triplet additions to rho are linear in T3 and vanish.
    // Consequently, the source moment needs pair domains but no neighboring
    // triplet domains.
    std::vector<int> ext_domain =
        merge_lists(merge_lists(lmopair_to_paos_[ij], lmopair_to_paos_[jk]), lmopair_to_paos_[ik]);
    for (int m : lmotriplet_to_lmos_[ijk]) {
        for (int p : std::array<int, 3>{i, j, k}) {
            const int mp = i_j_to_ij_[m][p];
            if (mp != -1) ext_domain = merge_lists(ext_domain, lmopair_to_paos_[mp]);
        }
        for (int l : lmotriplet_to_lmos_[ijk]) {
            const int ml = i_j_to_ij_[m][l];
            if (ml != -1) ext_domain = merge_lists(ext_domain, lmopair_to_paos_[ml]);
        }
    }

    auto S_ijk = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], ext_domain);
    S_ijk = linalg::doublet(X_tno_[ijk], S_ijk, true, false);

    const std::array<int, 3> occ = {i, j, k};
    const std::array<Tensor<double, 2>, 3> q_io = {q_io_[ijk], q_jo_[ijk], q_ko_[ijk]};
    const std::array<Tensor<double, 2>, 3> q_iv = {q_iv_[ijk], q_jv_[ijk], q_kv_[ijk]};
    std::array<Tensor<double, 1>, 3> T_i;
    std::array<Tensor<double, 2>, 3> q_io_t1;
    std::array<Tensor<double, 2>, 3> q_iv_t1;

    // T1-dressed density-fitting factors (Jiang et al., JCTC 21, 2386 (2025),
    // Eqs. 20--26 and Supporting Information Algorithm S1).
    for (int idx = 0; idx < 3; ++idx) {
        const auto pos = std::find(lmotriplet_to_lmos_[ijk].begin(), lmotriplet_to_lmos_[ijk].end(), occ[idx]);
        const int i_ijk = static_cast<int>(pos - lmotriplet_to_lmos_[ijk].begin());
        T_i[idx] = Tensor<double, 1>("T_i", ntno_ijk);
        std::memcpy(T_i[idx].data(), &T_n_ijk_[ijk](i_ijk, 0), ntno_ijk * sizeof(double));

        q_iv_t1[idx] = q_iv[idx];
        einsum(1.0, Indices{index::Q, index::a}, &q_iv_t1[idx], -1.0,
               Indices{index::Q, index::l}, q_io[idx], Indices{index::l, index::a}, T_n_ijk_[ijk]);
        einsum(1.0, Indices{index::Q, index::a}, &q_iv_t1[idx], 1.0,
               Indices{index::Q, index::a, index::b}, q_vv_[ijk], Indices{index::b}, T_i[idx]);
        Tensor<double, 2> tmp("q_iv_t1_tmp", naux_ijk, nlmo_ijk);
        einsum(0.0, Indices{index::Q, index::l}, &tmp, 1.0,
               Indices{index::Q, index::l, index::b}, q_ov_[ijk], Indices{index::b}, T_i[idx]);
        einsum(1.0, Indices{index::Q, index::a}, &q_iv_t1[idx], -1.0,
               Indices{index::Q, index::l}, tmp, Indices{index::l, index::a}, T_n_ijk_[ijk]);

        q_io_t1[idx] = q_io[idx];
        einsum(1.0, Indices{index::Q, index::l}, &q_io_t1[idx], 1.0,
               Indices{index::Q, index::l, index::a}, q_ov_[ijk], Indices{index::a}, T_i[idx]);
    }

    Tensor<double, 3> q_vv_t1 = q_vv_[ijk];
    Tensor<double, 3> q_vo("q_vo", naux_ijk, ntno_ijk, nlmo_ijk);
    permute(Indices{index::Q, index::a, index::l}, &q_vo,
            Indices{index::Q, index::l, index::a}, q_ov_[ijk]);
    einsum(1.0, Indices{index::Q, index::a, index::b}, &q_vv_t1, -1.0,
           Indices{index::Q, index::a, index::l}, q_vo,
           Indices{index::l, index::b}, T_n_ijk_[ijk]);

    Tensor<double, 1> gamma_Q("gamma_Q", naux_ijk);
    einsum(0.0, Indices{index::Q}, &gamma_Q, 1.0,
           Indices{index::Q, index::m, index::e}, q_ov_[ijk],
           Indices{index::m, index::e}, T_n_ijk_[ijk]);

    // F_ld is the occupied--virtual dressed Fock block. The canonical Full-T
    // implementation starts this tensor at zero because bare F_ld vanishes.
    // That assumption is invalid after a Brueckner rotation, so seed it from
    // the actual LMO--PAO Fock block before adding the CCSD dressing terms.
    // This supplies the otherwise-missing F_ov * T2 part of the complete R3
    // source used by DLPNO-BCCD(cT).
    auto F_ld_tno = submatrix_rows_and_cols(*F_lmo_pao_, lmotriplet_to_lmos_[ijk], lmotriplet_to_paos_[ijk]);
    F_ld_tno = linalg::doublet(F_ld_tno, X_tno_[ijk]);
    Tensor<double, 2> F_ld("F_ld", nlmo_ijk, ntno_ijk);
    std::memcpy(F_ld.data(), F_ld_tno->get_pointer(), nlmo_ijk * ntno_ijk * sizeof(double));
    einsum(1.0, Indices{index::l, index::d}, &F_ld, 2.0,
           Indices{index::Q, index::l, index::d}, q_ov_[ijk], Indices{index::Q}, gamma_Q);
    Tensor<double, 3> F_ld_K("F_ld_K", naux_ijk, nlmo_ijk, nlmo_ijk);
    einsum(0.0, Indices{index::Q, index::l, index::m}, &F_ld_K, 1.0,
           Indices{index::Q, index::l, index::e}, q_ov_[ijk],
           Indices{index::m, index::e}, T_n_ijk_[ijk]);
    Tensor<double, 3> F_ld_Kt("F_ld_Kt", naux_ijk, nlmo_ijk, nlmo_ijk);
    permute(Indices{index::Q, index::m, index::l}, &F_ld_Kt,
            Indices{index::Q, index::l, index::m}, F_ld_K);
    einsum(1.0, Indices{index::l, index::d}, &F_ld, -1.0,
           Indices{index::Q, index::m, index::l}, F_ld_Kt,
           Indices{index::Q, index::m, index::d}, q_ov_[ijk]);

    Tensor<double, 4> T_lm("T_lm", nlmo_ijk, nlmo_ijk, ntno_ijk, ntno_ijk);
    T_lm.zero();
    for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
        const int l = lmotriplet_to_lmos_[ijk][l_ijk];
        for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
            const int m = lmotriplet_to_lmos_[ijk][m_ijk];
            const int lm = i_j_to_ij_[l][m];
            if (lm == -1) continue;
            auto S_lm = submatrix_cols(*S_ijk, index_list(ext_domain, lmopair_to_paos_[lm]));
            S_lm = linalg::doublet(S_lm, X_pno_[lm]);
            auto T_lm_ijk = linalg::triplet(S_lm, T_iajb_[lm], S_lm, false, false, true);
            std::memcpy(&T_lm(l_ijk, m_ijk, 0, 0), T_lm_ijk->get_pointer(),
                        ntno_ijk * ntno_ijk * sizeof(double));
        }
    }

    Tensor<double, 3> buffer_a("ct_buffer_a", ntno_ijk, ntno_ijk, ntno_ijk);
    Tensor<double, 3> buffer_b("ct_buffer_b", ntno_ijk, ntno_ijk, ntno_ijk);
    std::array<Tensor<double, 3>, 3> rho_dbck;
    for (int idx = 0; idx < 3; ++idx) {
        const int p = occ[idx];
        rho_dbck[idx] = Tensor<double, 3>("rho_dbck", ntno_ijk, ntno_ijk, ntno_ijk);
        einsum(0.0, Indices{index::d, index::b, index::c}, &rho_dbck[idx], 1.0,
               Indices{index::Q, index::d, index::b}, q_vv_t1,
               Indices{index::Q, index::c}, q_iv_t1[idx]);

        Tensor<double, 3> T_lp("T_lp", nlmo_ijk, ntno_ijk, ntno_ijk);
        Tensor<double, 3> U_lp("U_lp", nlmo_ijk, ntno_ijk, ntno_ijk);
        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            const int l = lmotriplet_to_lmos_[ijk][l_ijk];
            const int lp = i_j_to_ij_[l][p];
            auto S_lp = submatrix_cols(*S_ijk, index_list(ext_domain, lmopair_to_paos_[lp]));
            S_lp = linalg::doublet(S_lp, X_pno_[lp]);
            auto T_lp_ijk = linalg::triplet(S_lp, T_iajb_[lp], S_lp, false, false, true);
            auto U_lp_ijk = linalg::triplet(S_lp, Tt_iajb_[lp], S_lp, false, false, true);
            std::memcpy(&T_lp(l_ijk, 0, 0), T_lp_ijk->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
            std::memcpy(&U_lp(l_ijk, 0, 0), U_lp_ijk->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
        }

        einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck[idx], -1.0,
               Indices{index::l, index::d}, F_ld,
               Indices{index::l, index::b, index::c}, T_lp);
        Tensor<double, 3> K_limd("K_limd", nlmo_ijk, nlmo_ijk, ntno_ijk);
        einsum(0.0, Indices{index::l, index::m, index::d}, &K_limd, 1.0,
               Indices{index::Q, index::l}, q_io_t1[idx],
               Indices{index::Q, index::m, index::d}, q_ov_[ijk]);
        Tensor<double, 3> K_ml_d("K_ml_d", nlmo_ijk, nlmo_ijk, ntno_ijk);
        permute(Indices{index::m, index::l, index::d}, &K_ml_d,
                Indices{index::l, index::m, index::d}, K_limd);
        einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck[idx], 1.0,
               Indices{index::m, index::l, index::d}, K_ml_d,
               Indices{index::m, index::l, index::b, index::c}, T_lm);

        Tensor<double, 2> q_c("q_c", naux_ijk, ntno_ijk);
        einsum(0.0, Indices{index::Q, index::c}, &q_c, 1.0,
               Indices{index::Q, index::l, index::e}, q_ov_[ijk],
               Indices{index::l, index::e, index::c}, U_lp);
        einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck[idx], 1.0,
               Indices{index::Q, index::d, index::b}, q_vv_t1,
               Indices{index::Q, index::c}, q_c);

        for (int q = 0; q < naux_ijk; ++q) {
            TensorView<double, 2> q_vv_slice = q_vv_t1(q, All, All);
            TensorView<double, 2> q_ov_slice = q_ov_[ijk](q, All, All);
            einsum(0.0, Indices{index::d, index::e, index::c}, &buffer_a, 1.0,
                   Indices{index::l, index::d}, q_ov_slice,
                   Indices{index::l, index::e, index::c}, T_lp);
            permute(Indices{index::e, index::d, index::c}, &buffer_b,
                    Indices{index::d, index::e, index::c}, buffer_a);
            einsum(0.0, Indices{index::b, index::d, index::c}, &buffer_a, -1.0,
                   Indices{index::e, index::d, index::c}, buffer_b,
                   Indices{index::e, index::b}, q_vv_slice);
            permute(Indices{index::d, index::b, index::c}, &buffer_b,
                    Indices{index::b, index::d, index::c}, buffer_a);
            rho_dbck[idx] += buffer_b;

            einsum(0.0, Indices{index::d, index::b, index::e}, &buffer_a, 1.0,
                   Indices{index::l, index::b, index::e}, T_lp,
                   Indices{index::l, index::d}, q_ov_slice);
            einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck[idx], -1.0,
                   Indices{index::d, index::b, index::e}, buffer_a,
                   Indices{index::e, index::c}, q_vv_slice);
        }
    }

    const std::array<std::tuple<int, int, int>, 6> perms = {
        std::make_tuple(i, j, k), std::make_tuple(i, k, j), std::make_tuple(j, i, k),
        std::make_tuple(j, k, i), std::make_tuple(k, i, j), std::make_tuple(k, j, i)};
    const std::array<std::tuple<int, int, int>, 6> perm_idx = {
        std::make_tuple(0, 1, 2), std::make_tuple(0, 2, 1), std::make_tuple(1, 0, 2),
        std::make_tuple(1, 2, 0), std::make_tuple(2, 0, 1), std::make_tuple(2, 1, 0)};
    std::array<Tensor<double, 2>, 6> rho_ljck;

    for (int idx = 0; idx < 6; ++idx) {
        int pi, pj, pk, pi_idx, pj_idx, pk_idx;
        std::tie(pi, pj, pk) = perms[idx];
        std::tie(pi_idx, pj_idx, pk_idx) = perm_idx[idx];
        const int pjk = i_j_to_ij_[pj][pk];
        rho_ljck[idx] = Tensor<double, 2>("rho_ljck", nlmo_ijk, ntno_ijk);
        einsum(0.0, Indices{index::l, index::c}, &rho_ljck[idx], 1.0,
               Indices{index::Q, index::l}, q_io_t1[pj_idx],
               Indices{index::Q, index::c}, q_iv_t1[pk_idx]);

        Tensor<double, 3> T_jm("T_jm", nlmo_ijk, ntno_ijk, ntno_ijk);
        Tensor<double, 3> T_mk("T_mk", nlmo_ijk, ntno_ijk, ntno_ijk);
        Tensor<double, 3> U_mk("U_mk", nlmo_ijk, ntno_ijk, ntno_ijk);
        for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
            const int m = lmotriplet_to_lmos_[ijk][m_ijk];
            const int jm = i_j_to_ij_[pj][m];
            const int mk = i_j_to_ij_[m][pk];
            auto S_jm = submatrix_cols(*S_ijk, index_list(ext_domain, lmopair_to_paos_[jm]));
            S_jm = linalg::doublet(S_jm, X_pno_[jm]);
            auto T_jm_ijk = linalg::triplet(S_jm, T_iajb_[jm], S_jm, false, false, true);
            std::memcpy(&T_jm(m_ijk, 0, 0), T_jm_ijk->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));

            auto S_mk = submatrix_cols(*S_ijk, index_list(ext_domain, lmopair_to_paos_[mk]));
            S_mk = linalg::doublet(S_mk, X_pno_[mk]);
            auto T_mk_ijk = linalg::triplet(S_mk, T_iajb_[mk], S_mk, false, false, true);
            auto U_mk_ijk = linalg::triplet(S_mk, Tt_iajb_[mk], S_mk, false, false, true);
            std::memcpy(&T_mk(m_ijk, 0, 0), T_mk_ijk->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
            std::memcpy(&U_mk(m_ijk, 0, 0), U_mk_ijk->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
        }

        Tensor<double, 3> K_mdlj("K_mdlj", nlmo_ijk, ntno_ijk, nlmo_ijk);
        einsum(0.0, Indices{index::m, index::d, index::l}, &K_mdlj, 1.0,
               Indices{index::Q, index::m, index::d}, q_ov_[ijk],
               Indices{index::Q, index::l}, q_io_t1[pj_idx]);
        einsum(1.0, Indices{index::l, index::c}, &rho_ljck[idx], 1.0,
               Indices{index::m, index::d, index::l}, K_mdlj,
               Indices{index::m, index::d, index::c}, U_mk);
        Tensor<double, 3> K_lmd("K_lmd", nlmo_ijk, nlmo_ijk, ntno_ijk);
        permute(Indices{index::l, index::m, index::d}, &K_lmd,
                Indices{index::l, index::d, index::m}, K_mdlj);
        einsum(1.0, Indices{index::l, index::c}, &rho_ljck[idx], -1.0,
               Indices{index::l, index::m, index::d}, K_lmd,
               Indices{index::m, index::d, index::c}, T_mk);

        Tensor<double, 3> K_ldmk("K_ldmk", nlmo_ijk, ntno_ijk, nlmo_ijk);
        einsum(0.0, Indices{index::l, index::d, index::m}, &K_ldmk, 1.0,
               Indices{index::Q, index::l, index::d}, q_ov_[ijk],
               Indices{index::Q, index::m}, q_io_t1[pk_idx]);
        Tensor<double, 3> K_lmd2("K_lmd2", nlmo_ijk, nlmo_ijk, ntno_ijk);
        permute(Indices{index::l, index::m, index::d}, &K_lmd2,
                Indices{index::l, index::d, index::m}, K_ldmk);
        einsum(1.0, Indices{index::l, index::c}, &rho_ljck[idx], -1.0,
               Indices{index::l, index::m, index::d}, K_lmd2,
               Indices{index::m, index::d, index::c}, T_jm);

        auto S_jk = submatrix_cols(*S_ijk, index_list(ext_domain, lmopair_to_paos_[pjk]));
        S_jk = linalg::doublet(S_jk, X_pno_[pjk]);
        auto T_jk_psi = linalg::triplet(S_jk, T_iajb_[pjk], S_jk, false, false, true);
        Tensor<double, 2> T_jk("T_jk", ntno_ijk, ntno_ijk);
        std::memcpy(T_jk.data(), T_jk_psi->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
        for (int q = 0; q < naux_ijk; ++q) {
            TensorView<double, 2> q_vv_slice = q_vv_t1(q, All, All);
            TensorView<double, 2> q_ov_slice = q_ov_[ijk](q, All, All);
            Tensor<double, 2> ld("ld", nlmo_ijk, ntno_ijk);
            einsum(0.0, Indices{index::l, index::d}, &ld, 1.0,
                   Indices{index::l, index::e}, q_ov_slice,
                   Indices{index::e, index::d}, T_jk);
            einsum(1.0, Indices{index::l, index::c}, &rho_ljck[idx], 1.0,
                   Indices{index::l, index::d}, ld,
                   Indices{index::d, index::c}, q_vv_slice);
        }
    }

    std::array<Tensor<double, 3>, 6> W;
    for (int idx = 0; idx < 6; ++idx) {
        int pi, pj, pk, pi_idx, pj_idx, pk_idx;
        std::tie(pi, pj, pk) = perms[idx];
        std::tie(pi_idx, pj_idx, pk_idx) = perm_idx[idx];
        const int pij = i_j_to_ij_[pi][pj];
        W[idx] = Tensor<double, 3>("Wperm(cT)", ntno_ijk, ntno_ijk, ntno_ijk);

        auto S_ij = submatrix_cols(*S_ijk, index_list(ext_domain, lmopair_to_paos_[pij]));
        S_ij = linalg::doublet(S_ij, X_pno_[pij]);
        auto T_ij_psi = linalg::triplet(S_ij, T_iajb_[pij], S_ij, false, false, true);
        Tensor<double, 2> T_ij("T_ij", ntno_ijk, ntno_ijk);
        std::memcpy(T_ij.data(), T_ij_psi->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
        einsum(0.0, Indices{index::a, index::b, index::c}, &W[idx], 1.0,
               Indices{index::a, index::d}, T_ij,
               Indices{index::d, index::b, index::c}, rho_dbck[pk_idx]);

        Tensor<double, 3> T_il("T_il", nlmo_ijk, ntno_ijk, ntno_ijk);
        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            const int l = lmotriplet_to_lmos_[ijk][l_ijk];
            const int il = i_j_to_ij_[pi][l];
            auto S_il = submatrix_cols(*S_ijk, index_list(ext_domain, lmopair_to_paos_[il]));
            S_il = linalg::doublet(S_il, X_pno_[il]);
            auto T_il_psi = linalg::triplet(S_il, T_iajb_[il], S_il, false, false, true);
            std::memcpy(&T_il(l_ijk, 0, 0), T_il_psi->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
        }
        einsum(1.0, Indices{index::a, index::b, index::c}, &W[idx], -1.0,
               Indices{index::l, index::a, index::b}, T_il,
               Indices{index::l, index::c}, rho_ljck[idx]);
    }

    for (int a = 0; a < ntno_ijk; ++a) {
        for (int b = 0; b < ntno_ijk; ++b) {
            for (int c = 0; c < ntno_ijk; ++c) {
                (*M_ijk)(a, b * ntno_ijk + c) = W[0](a, b, c) + W[1](a, c, b) + W[2](b, a, c) +
                                                  W[3](b, c, a) + W[4](c, a, b) + W[5](c, b, a);
            }
        }
    }
    return M_ijk;
}

SharedMatrix DLPNOCCSD_cT::load_triples_energy_moment(int ijk) {
    if (!write_intermediates_) return V_iajbkc_[ijk];

    const int ntno_ijk = n_tno_[ijk];
    std::stringstream name;
    name << "V " << ijk;
    auto V_ijk = std::make_shared<Matrix>(name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical(dlpno_ct_psio)
    V_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
    return V_ijk;
}

double DLPNOCCSD_cT::compute_ct_energy() {
    const int n_lmo_triplets = static_cast<int>(ijk_to_i_j_k_.size());
    double E_cT = 0.0;

#pragma omp parallel for schedule(dynamic, 1) reduction(+ : E_cT)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        const int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;

        auto M_ijk = build_ct_moment(ijk);
        auto T_cT = M_ijk->clone();
        T_cT->set_name("T(cT)");
        for (int a = 0; a < ntno_ijk; ++a) {
            for (int b = 0; b < ntno_ijk; ++b) {
                for (int c = 0; c < ntno_ijk; ++c) {
                    const double denominator = (*e_tno_[ijk])(a) + (*e_tno_[ijk])(b) + (*e_tno_[ijk])(c) -
                                               (*F_lmo_)(i, i) - (*F_lmo_)(j, j) - (*F_lmo_)(k, k);
                    (*T_cT)(a, b * ntno_ijk + c) = -(*M_ijk)(a, b * ntno_ijk + c) / denominator;
                }
            }
        }

        auto V_ijk = load_triples_energy_moment(ijk);
        double prefactor = 1.0;
        if (i == j && j == k) {
            prefactor /= 6.0;
        } else if (i == j || j == k || i == k) {
            prefactor /= 2.0;
        }

        double e_ijk = 8.0 * prefactor * V_ijk->vector_dot(T_cT);
        e_ijk -= 4.0 * prefactor * triples_permuter(V_ijk, k, j, i)->vector_dot(T_cT);
        e_ijk -= 4.0 * prefactor * triples_permuter(V_ijk, i, k, j)->vector_dot(T_cT);
        e_ijk -= 4.0 * prefactor * triples_permuter(V_ijk, j, i, k)->vector_dot(T_cT);
        e_ijk += 2.0 * prefactor * triples_permuter(V_ijk, j, k, i)->vector_dot(T_cT);
        e_ijk += 2.0 * prefactor * triples_permuter(V_ijk, k, i, j)->vector_dot(T_cT);
        E_cT += e_ijk;
    }
    return E_cT;
}

void DLPNOCCSD_cT::clear_ct_state() {
    q_io_.clear();
    q_jo_.clear();
    q_ko_.clear();
    q_iv_.clear();
    q_jv_.clear();
    q_kv_.clear();
    q_ov_.clear();
    q_vv_.clear();
    T_n_ijk_.clear();
}

void DLPNOCCSD_cT::compute_ct_correction(DLPNOCCSDPhase phase) {
    timer_on("DLPNO-CCSD(cT)");
    const bool bccd_result = brueckner_orbs_ && phase == DLPNOCCSDPhase::FinalBrueckner;
    const std::string reference = bccd_result ? "BCCD" : "CCSD";
    const std::string method = reference + "(cT)";

    outfile->Printf("\n   --------------------------------------------\n");
    outfile->Printf("                 DLPNO-%-25s\n", method.c_str());
    outfile->Printf("      Complete perturbative triples (cT)      \n");
    outfile->Printf("          DOI: 10.1103/PhysRevLett.131.186401 \n");
    outfile->Printf("   --------------------------------------------\n\n");

    // Release post-CCSD intermediates not used by either the ordinary triples
    // energy moment or the complete CCSDT source moment.
    K_mibj_.clear();
    J_ijmb_.clear();
    L_mibj_.clear();
    L_iajb_.clear();
    J_ikac_non_proj_.clear();
    K_iakc_non_proj_.clear();
    K_ivvv_.clear();
    Qma_ij_.clear();
    Qab_ij_.clear();
    i_Qk_ij_.clear();
    i_Qa_ij_.clear();
    i_Qk_t1_.clear();
    i_Qa_t1_.clear();
    S_pno_ij_kj_.clear();
    S_pno_ij_nn_.clear();
    S_pno_ij_mn_.clear();
    Fkc_.clear();
    Fai_.clear();
    Fab_.clear();
    T_n_ij_.clear();

    // A Brueckner calculation evaluates cT twice. Rebuild every
    // triplet-dependent object in the current orbital frame for each phase.
    lmotriplet_to_ribfs_.clear();
    lmotriplet_to_lmos_.clear();
    lmotriplet_to_paos_.clear();
    i_j_k_to_ijk_.clear();
    ijk_to_i_j_k_.clear();
    W_iajbkc_.clear();
    V_iajbkc_.clear();
    L_iajbkc_.clear();
    T_iajbkc_.clear();
    X_tno_.clear();
    e_tno_.clear();
    n_tno_.clear();
    e_ijk_.clear();
    e_ijk_right_.clear();
    tno_scale_.clear();
    is_strong_triplet_.clear();
    de_lccsd_t_screened_ = 0.0;
    de_lccsd_t_l_screened_ = 0.0;
    e_lccsd_t_ = 0.0;
    E_T_ = 0.0;
    e_lccsd_t_l_ = 0.0;
    E_T_L_ = 0.0;

    psio_->open(PSIF_DLPNO_TRIPLES, PSIO_OPEN_NEW);

    const double t_cut_tno_pre = options_.get_double("T_CUT_TNO_PRE");
    const double t_cut_tno = options_.get_double("T_CUT_TNO");

    // Use the established semicanonical triples estimate to screen local
    // triplets, then evaluate cT only in the retained nominal TNO spaces.
    outfile->Printf("   Starting cT Triplet Prescreening...\n");
    outfile->Printf("     T_CUT_TNO set to %6.3e \n", t_cut_tno_pre);
    outfile->Printf("     T_CUT_DO  set to %6.3e \n", options_.get_double("T_CUT_DO_TRIPLES_PRE"));
    outfile->Printf("     T_CUT_MKN set to %6.3e \n\n", options_.get_double("T_CUT_MKN_TRIPLES_PRE"));
    triples_sparsity(true);
    tno_transform(t_cut_tno_pre);
    compute_lccsd_t0();

    triples_sparsity(false);
    outfile->Printf("    * Energy From Screened Triplets: %.12f \n\n", de_lccsd_t_screened_);
    outfile->Printf("     T_CUT_TNO (re)set to %6.3e \n", t_cut_tno);
    outfile->Printf("     T_CUT_DO  (re)set to %6.3e \n", options_.get_double("T_CUT_DO_TRIPLES"));
    outfile->Printf("     T_CUT_MKN (re)set to %6.3e \n\n", options_.get_double("T_CUT_MKN_TRIPLES"));
    tno_transform(t_cut_tno);
    estimate_triples_memory();
    const double E_T0 = compute_lccsd_t0(true).first;  // supplies the ordinary energy moment V
    set_scalar_variable("DLPNO SEMICANONICAL (T0) ENERGY", E_T0 + de_lccsd_t_screened_);
    set_scalar_variable("DLPNO SCREENED TRIPLETS ENERGY", de_lccsd_t_screened_);

    // W and the semicanonical amplitudes are not needed once V is available.
    W_iajbkc_.clear();
    T_iajbkc_.clear();
    L_iajbkc_.clear();

    project_triplet_singles();
    compute_ct_integrals();
    const double E_cT = compute_ct_energy();
    e_lccsd_ct_ = e_lccsd_ + E_cT + de_lccsd_t_screened_;

    const double correlation =
        e_lccsd_ct_ + de_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;
    const double total = scalar_variable("SCF TOTAL ENERGY") + correlation;
    const double correction = e_lccsd_ct_ - e_lccsd_;

    set_scalar_variable(method + " CORRELATION ENERGY", correlation);
    set_scalar_variable(method + " TOTAL ENERGY", total);
    set_scalar_variable("DLPNO-" + method + " CORRELATION ENERGY", correlation);
    set_scalar_variable("DLPNO-" + method + " TOTAL ENERGY", total);
    set_scalar_variable("(cT) CORRECTION ENERGY", correction);
    if (bccd_result) set_scalar_variable("B(cT) CORRECTION ENERGY", correction);

    if (phase == DLPNOCCSDPhase::InitialBrueckner) {
        set_scalar_variable("INITIAL DLPNO-CCSD(cT) CORRELATION ENERGY", correlation);
        set_scalar_variable("INITIAL DLPNO-CCSD(cT) TOTAL ENERGY", total);
        set_scalar_variable("INITIAL DLPNO-(cT) CORRECTION ENERGY", correction);
    }

    set_scalar_variable("CURRENT CORRELATION ENERGY", correlation);
    set_scalar_variable("CURRENT ENERGY", total);

    outfile->Printf("    DLPNO-%s Correlation Energy:       %16.12f\n", method.c_str(), correlation);
    outfile->Printf("    * DLPNO-%s Contribution:           %16.12f\n", reference.c_str(),
                    e_lccsd_ + de_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_);
    outfile->Printf("    * DLPNO-(cT) Contribution:         %16.12f\n", E_cT);
    outfile->Printf("    * Screened Triplets Contribution:  %16.12f\n", de_lccsd_t_screened_);
    outfile->Printf("\n  @Total DLPNO-%s Energy: %16.12f\n\n", method.c_str(), total);

    clear_ct_state();
    V_iajbkc_.clear();
    psio_->close(PSIF_DLPNO_TRIPLES, 0);
    timer_off("DLPNO-CCSD(cT)");
}

void DLPNOCCSD_cT::post_ccsd_correction(DLPNOCCSDPhase phase) {
    // In a Brueckner job this hook is called once before orbital optimization
    // and once after convergence, so both CCSD(cT) and BCCD(cT) are evaluated
    // and published without running the iterative (T) solver as a side effect.
    DLPNOCCSD::post_ccsd_correction(phase);
    compute_ct_correction(phase);
}

double DLPNOCCSD_cT::compute_energy() {
    DLPNOCCSD_T::compute_energy();
    return scalar_variable("CURRENT ENERGY");
}

#endif

}  // namespace dlpno
}  // namespace psi
