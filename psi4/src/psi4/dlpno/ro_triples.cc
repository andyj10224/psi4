/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with Psi4; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "dlpno.h"
#include "sparse.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ctime>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {
namespace dlpno {

namespace {

constexpr std::array<SpinCase, 2> kSpins = {SpinCase::Alpha, SpinCase::Beta};
constexpr std::array<TripleSpinCase, 4> kTripleSpins = {TripleSpinCase::AAA, TripleSpinCase::AAB, TripleSpinCase::BBA,
                                                        TripleSpinCase::BBB};

const char* triple_spin_label(TripleSpinCase spin) {
    switch (spin) {
        case TripleSpinCase::AAA:
            return "AAA";
        case TripleSpinCase::AAB:
            return "AAB";
        case TripleSpinCase::BBA:
            return "BBA";
        case TripleSpinCase::BBB:
            return "BBB";
    }
    return "UNKNOWN";
}

inline int triple_key(int i, int j, int k, int naocc) { return i * naocc * naocc + j * naocc + k; }

inline double cube_get(const SharedMatrix& X, int n, int a, int b, int c) { return X->get(a, b * n + c); }

inline void cube_set(const SharedMatrix& X, int n, int a, int b, int c, double value) { X->set(a, b * n + c, value); }

inline void cube_add(const SharedMatrix& X, int n, int a, int b, int c, double value) {
    X->add(0, a, b * n + c, value);
}

// K(a,b,d) T(d,c), or K(a,b,d) T(c,d).  The contiguous cube can be
// viewed as (a*b,d) without copying or changing a shared Matrix's shape.
void ro_contract_vvv_t2(const SharedMatrix& K, const SharedMatrix& T, const SharedMatrix& result,
                       int n, bool transpose_t) {
    C_DGEMM('N', transpose_t ? 'T' : 'N', n * n, n, n, 1.0, K->get_pointer(), n,
            T->get_pointer(), n, 0.0, result->get_pointer(), n);
}

// Only the O(n^3) index permutation remains outside BLAS.  Reuse one work
// cube for successive contractions rather than keeping four products live.
void ro_add_permuted_cube(const SharedMatrix& result, const SharedMatrix& source, int n,
                         const std::array<int, 3>& axes, double scale) {
    const auto src = source->get_pointer();
    auto dst = result->get_pointer();
    const size_t nn = static_cast<size_t>(n) * n;
    if (axes == std::array<int, 3>{0, 1, 2}) {
        C_DAXPY(n * nn, scale, src, 1, dst, 1);
        return;
    }
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            for (int c = 0; c < n; ++c) {
                const std::array<int, 3> index = {a, b, c};
                dst[a * nn + b * n + c] +=
                    scale * src[index[axes[0]] * nn + index[axes[1]] * n + index[axes[2]]];
            }
        }
    }
}

// Full spin-dependent virtual Fock action, with no semicanonical or
// spin-averaging approximation.  Fixed-a slices handle the middle index
// without transposing an entire cube or allocating another work tensor.
void ro_add_virtual_fock(const SharedMatrix& residual, const SharedMatrix& T,
                        const std::array<SharedMatrix, 3>& F, double occupied_diagonal, int n) {
    const int nn = n * n;
    auto t = T->get_pointer();
    auto r = residual->get_pointer();
    C_DAXPY(static_cast<size_t>(n) * nn, -occupied_diagonal, t, 1, r, 1);
    C_DGEMM('N', 'N', n, nn, n, 1.0, F[0]->get_pointer(), n, t, nn, 1.0, r, nn);
    C_DGEMM('N', 'T', nn, n, n, 1.0, t, n, F[2]->get_pointer(), n, 1.0, r, n);
    for (int a = 0; a < n; ++a) {
        const size_t offset = static_cast<size_t>(a) * nn;
        C_DGEMM('N', 'N', n, n, n, 1.0, F[1]->get_pointer(), n, t + offset, n, 1.0, r + offset, n);
    }
}

}  // namespace

RO_DLPNOCCSD_T::RO_DLPNOCCSD_T(SharedWavefunction ref_wfn, Options& options) : RO_DLPNOCCSD(ref_wfn, options) {}

RO_DLPNOCCSD_T::~RO_DLPNOCCSD_T() = default;

void RO_DLPNOCCSD_T::ro_print_header() {
    const bool t0_only = options_.get_bool("T0_APPROXIMATION");
    const std::string triples_algorithm = t0_only ? "SEMICANONICAL (T0)" : "ITERATIVE (T)";
    const double t_cut_tno = options_.get_double("T_CUT_TNO");

    outfile->Printf("   --------------------------------------------\n");
    outfile->Printf("                 RO-DLPNO-CCSD(T)              \n");
    outfile->Printf("   --------------------------------------------\n\n");
    outfile->Printf("  Open-shell triples equations: DOI 10.1063/1.5127550\n");
    outfile->Printf("  DLPNO convergence set to %s.\n\n", options_.get_str("PNO_CONVERGENCE").c_str());
    outfile->Printf("  Detailed DLPNO triples thresholds and cutoffs:\n");
    outfile->Printf("    ALGORITHM                    = %6s\n", triples_algorithm.c_str());
    outfile->Printf("    T_CUT_TNO (T0)              = %6.3e\n", t_cut_tno);
    outfile->Printf("    T_CUT_DO_TRIPLES (T0)       = %6.3e\n", options_.get_double("T_CUT_DO_TRIPLES"));
    outfile->Printf("    T_CUT_MKN_TRIPLES (T0)      = %6.3e\n", options_.get_double("T_CUT_MKN_TRIPLES"));
    outfile->Printf("    T_CUT_TRIPLES_WEAK (T0)     = %6.3e\n", options_.get_double("T_CUT_TRIPLES_WEAK"));
    outfile->Printf("    T_CUT_TNO_PRE (T0)          = %6.3e\n", options_.get_double("T_CUT_TNO_PRE"));
    outfile->Printf("    T_CUT_DO_TRIPLES_PRE (T0)   = %6.3e\n", options_.get_double("T_CUT_DO_TRIPLES_PRE"));
    outfile->Printf("    T_CUT_MKN_TRIPLES_PRE (T0)  = %6.3e\n", options_.get_double("T_CUT_MKN_TRIPLES_PRE"));
    if (!t0_only) {
        outfile->Printf("    T_CUT_TNO_STRONG (T)        = %6.3e\n",
                        t_cut_tno * options_.get_double("T_CUT_TNO_STRONG_SCALE"));
        outfile->Printf("    T_CUT_TNO_WEAK (T)          = %6.3e\n",
                        t_cut_tno * options_.get_double("T_CUT_TNO_WEAK_SCALE"));
        outfile->Printf("    F_CUT_T (T)                 = %6.3e\n", options_.get_double("F_CUT_T"));
        outfile->Printf("    T_CUT_ITER (T)              = %6.3e\n", options_.get_double("T_CUT_ITER"));
    }
    outfile->Printf("    MIN_TNOS                    = %6d\n", options_.get_int("MIN_TNOS"));
    outfile->Printf("    TRIPLES_MAX_WEAK_PAIRS      = %6d\n\n", options_.get_int("TRIPLES_MAX_WEAK_PAIRS"));
}

SharedMatrix RO_DLPNOCCSD_T::ro_matmul_3d(SharedMatrix A, SharedMatrix X, int dim_old, int dim_new) {
    // X(new,old); A'[a,b,c] = X(a,I) X(b,J) X(c,K) A[I,J,K].
    auto A_new = linalg::doublet(X, A, false, false);
    A_new->reshape(dim_new * dim_old, dim_old);
    A_new = linalg::doublet(A_new, X, false, true);

    // A_new is now (a,J,c). Contract J in fixed-a slices to produce
    // (a,b,c) directly, avoiding two full-cube permutations and their buffers.
    auto result = std::make_shared<Matrix>(dim_new, dim_new * dim_new);
    for (int a = 0; a < dim_new; ++a) {
        C_DGEMM('N', 'N', dim_new, dim_new, dim_old, 1.0, X->get_pointer(), dim_old,
                A_new->get_pointer() + static_cast<size_t>(a) * dim_old * dim_new, dim_new, 0.0,
                result->get_pointer() + static_cast<size_t>(a) * dim_new * dim_new, dim_new);
    }
    return result;
}

SharedMatrix RO_DLPNOCCSD_T::ro_triples_permuter(const SharedMatrix& X, TripleSpinCase spin, int i, int j, int k) {
    const int n = X->nrow();
    // Callers only read the permuted tensor. An identity permutation may
    // safely share the old Jacobi generation, including disk-loaded blocks.
    if (i <= j && ((spin == TripleSpinCase::AAB || spin == TripleSpinCase::BBA) || j <= k)) return X;
    auto result = std::make_shared<Matrix>(n, n * n);

    if (spin == TripleSpinCase::AAB || spin == TripleSpinCase::BBA) {
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                for (int c = 0; c < n; ++c) cube_set(result, n, a, b, c, cube_get(X, n, b, a, c));
            }
        }
        return result;
    }

    int permutation = 5;
    if (i <= j && j <= k) {
        permutation = 0;
    } else if (i <= k && k <= j) {
        permutation = 1;
    } else if (j <= i && i <= k) {
        permutation = 2;
    } else if (j <= k && k <= i) {
        permutation = 3;
    } else if (k <= i && i <= j) {
        permutation = 4;
    }

    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            for (int c = 0; c < n; ++c) {
                double value = 0.0;
                if (permutation == 0) value = cube_get(X, n, a, b, c);
                if (permutation == 1) value = cube_get(X, n, a, c, b);
                if (permutation == 2) value = cube_get(X, n, b, a, c);
                if (permutation == 3) value = cube_get(X, n, b, c, a);
                if (permutation == 4) value = cube_get(X, n, c, a, b);
                if (permutation == 5) value = cube_get(X, n, c, b, a);
                cube_set(result, n, a, b, c, value);
            }
        }
    }
    return result;
}

void RO_DLPNOCCSD_T::triples_spin_enforcer(SharedMatrix& X, TripleSpinCase spin, int i, int j, int k) {
    const int naocc = nalpha_ - nfrzc();
    const int nbocc = nbeta_ - nfrzc();
    const int nsomo = naocc - nbocc;
    const auto spins = get_spin_triple(spin);
    const std::array<int, 3> occupied = {i, j, k};

    if (i == j || ((spin == TripleSpinCase::AAA || spin == TripleSpinCase::BBB) && (i == k || j == k))) {
        X->zero();
        return;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (spins[axis] == SpinCase::Beta && occupied[axis] >= nbocc) {
            X->zero();
            return;
        }
    }

    if (nsomo == 0) return;
    const int n = X->nrow();
    const int alpha_virtual_end = n - nsomo;
    auto data = X->get_pointer();
    const size_t nn = static_cast<size_t>(n) * n;
    // Clear only forbidden contiguous slabs/rows/tails; do not scan every
    // allowed element of the cube to discover that it needs no change.
    const int a_end = spins[0] == SpinCase::Alpha ? alpha_virtual_end : n;
    const int b_end = spins[1] == SpinCase::Alpha ? alpha_virtual_end : n;
    if (a_end < n) std::fill(data + a_end * nn, data + n * nn, 0.0);
    for (int a = 0; a < a_end; ++a) {
        auto slab = data + a * nn;
        if (b_end < n) std::fill(slab + b_end * n, slab + nn, 0.0);
        if (spins[2] == SpinCase::Alpha) {
            for (int b = 0; b < b_end; ++b)
                std::fill(slab + b * n + alpha_virtual_end, slab + (b + 1) * n, 0.0);
        }
    }
}

void RO_DLPNOCCSD_T::ro_triples_sparsity(bool prescreening) {
    timer_on("RO Triples Sparsity");

    const int naocc = nalpha_ - nfrzc();
    const int nbocc = nbeta_ - nfrzc();
    const int nbf = basisset_->nbf();
    const int npao = C_pao_->ncol();
    const int natom = molecule_->natom();

    auto rebuild_maps = [&]() {
        for (auto& map : i_j_k_to_ijk_spin_) map.clear();
        for (int ijk = 0; ijk < static_cast<int>(ijk_to_i_j_k_ro_.size()); ++ijk) {
            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_ro_[ijk];
            for (TripleSpinCase spin : kTripleSpins) {
                const int ts = static_cast<int>(spin);
                if (!active_ijk_spin_[ts][ijk]) continue;
                auto& map = i_j_k_to_ijk_spin_[ts];
                if (spin == TripleSpinCase::AAB || spin == TripleSpinCase::BBA) {
                    map[triple_key(i, j, k, naocc)] = ijk;
                    map[triple_key(j, i, k, naocc)] = ijk;
                } else {
                    const std::array<std::array<int, 3>, 6> permutations = {
                        std::array<int, 3>{i, j, k}, std::array<int, 3>{i, k, j}, std::array<int, 3>{j, i, k},
                        std::array<int, 3>{j, k, i}, std::array<int, 3>{k, i, j}, std::array<int, 3>{k, j, i}};
                    for (const auto& p : permutations) map[triple_key(p[0], p[1], p[2], naocc)] = ijk;
                }
            }
        }
    };

    if (prescreening) {
        ijk_to_i_j_k_ro_.clear();
        for (auto& active : active_ijk_spin_) active.clear();

        const int max_weak_pairs = options_.get_int("TRIPLES_MAX_WEAK_PAIRS");
        for (int i = 0; i < naocc; ++i) {
            for (int j = i + 1; j < naocc; ++j) {
                const int ij = i_j_to_ij_[i][j];
                if (ij == -1) continue;
                for (int k : lmopair_to_lmos_[ij]) {
                    const int ik = i_j_to_ij_[i][k];
                    const int jk = i_j_to_ij_[j][k];
                    if (ik == -1 || jk == -1) continue;

                    int weak_pair_count = 0;
                    if (i_j_to_ij_weak_[i][j] != -1) ++weak_pair_count;
                    if (i_j_to_ij_weak_[i][k] != -1) ++weak_pair_count;
                    if (i_j_to_ij_weak_[j][k] != -1) ++weak_pair_count;
                    if (weak_pair_count > max_weak_pairs) continue;

                    std::array<bool, 4> active = {j < k,                   // AAA: i < j < k
                                                  k < nbocc,               // AAB: i < j, k beta occupied
                                                  i < nbocc && j < nbocc,  // BBA: i < j beta occupied, k alpha occupied
                                                  j < k && k < nbocc};     // BBB: i < j < k beta occupied

                    if (!std::any_of(active.begin(), active.end(), [](bool value) { return value; })) continue;
                    ijk_to_i_j_k_ro_.emplace_back(i, j, k);
                    for (int ts = 0; ts < 4; ++ts) active_ijk_spin_[ts].push_back(active[ts]);
                }
            }
        }
    } else {
        const double energy_cutoff = options_.get_double("T_CUT_TRIPLES_WEAK");
        de_lccsd_t_screened_ro_ = 0.0;
        de_lccsd_t_screened_spin_ro_.fill(0.0);

        std::vector<std::tuple<int, int, int>> retained_triplets;
        std::array<std::vector<bool>, 4> retained_active;
        for (int old_ijk = 0; old_ijk < static_cast<int>(ijk_to_i_j_k_ro_.size()); ++old_ijk) {
            if (std::fabs(e_ijk_ro_[old_ijk]) < energy_cutoff) {
                for (TripleSpinCase spin : kTripleSpins) {
                    const int ts = static_cast<int>(spin);
                    if (!active_ijk_spin_[ts][old_ijk]) continue;
                    de_lccsd_t_screened_ro_ += e_ijk_spin_[ts][old_ijk];
                    de_lccsd_t_screened_spin_ro_[ts] += e_ijk_spin_[ts][old_ijk];
                }
                continue;
            }

            retained_triplets.push_back(ijk_to_i_j_k_ro_[old_ijk]);
            for (int ts = 0; ts < 4; ++ts) retained_active[ts].push_back(active_ijk_spin_[ts][old_ijk]);
        }
        ijk_to_i_j_k_ro_ = std::move(retained_triplets);
        active_ijk_spin_ = std::move(retained_active);
    }

    rebuild_maps();

    const int ntriplets = ijk_to_i_j_k_ro_.size();
    tno_scale_ro_.assign(ntriplets, 1.0);

    SparseMap lmo_to_ribfs(naocc);
    SparseMap lmo_to_paos(naocc);
    const double mkn_cutoff = options_.get_double(prescreening ? "T_CUT_MKN_TRIPLES_PRE" : "T_CUT_MKN_TRIPLES");
    const double doi_cutoff = options_.get_double(prescreening ? "T_CUT_DO_TRIPLES_PRE" : "T_CUT_DO_TRIPLES");

    for (int i = 0; i < naocc; ++i) {
        std::vector<double> mulliken_population(natom, 0.0);
        auto orbital_density = reference_wavefunction_->S()->clone();
        for (int mu = 0; mu < nbf; ++mu) {
            orbital_density->scale_row(0, mu, C_lmo_->get(mu, i));
            orbital_density->scale_column(0, mu, C_lmo_->get(mu, i));
        }
        for (int mu = 0; mu < nbf; ++mu) {
            const int center_mu = basisset_->function_to_center(mu);
            const double p_mu_mu = orbital_density->get(mu, mu);
            for (int nu = 0; nu < nbf; ++nu) {
                const int center_nu = basisset_->function_to_center(nu);
                const double p_nu_nu = orbital_density->get(nu, nu);
                const double denominator = p_mu_mu + p_nu_nu;
                if (std::fabs(denominator) < 1.0e-16) continue;
                const double p_mu_nu = orbital_density->get(mu, nu);
                mulliken_population[center_mu] += p_mu_nu * p_mu_mu / denominator;
                mulliken_population[center_nu] += p_mu_nu * p_nu_nu / denominator;
            }
        }
        for (int atom = 0; atom < natom; ++atom) {
            if (std::fabs(mulliken_population[atom]) <= mkn_cutoff) continue;
            lmo_to_ribfs[i] = merge_lists(lmo_to_ribfs[i], atom_to_ribf_[atom]);
        }

        std::vector<int> selected_paos;
        for (int u = 0; u < nbf; ++u) {
            if (std::fabs(DOI_iu_->get(i, u)) > doi_cutoff) selected_paos.push_back(u);
        }
        lmo_to_paos[i] = contract_lists(selected_paos, atom_to_bf_);
        // The appended SOMOs are mandatory beta virtuals and must be present in every triples domain.
        for (int u = nbf; u < npao; ++u) lmo_to_paos[i].push_back(u);
    }

    lmotriplet_to_ribfs_ro_.assign(ntriplets, {});
    lmotriplet_to_lmos_ro_.assign(ntriplets, {});
    lmotriplet_to_paos_ro_.assign(ntriplets, {});

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < ntriplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_ro_[ijk];
        lmotriplet_to_ribfs_ro_[ijk] = merge_lists(lmo_to_ribfs[i], merge_lists(lmo_to_ribfs[j], lmo_to_ribfs[k]));
        lmotriplet_to_paos_ro_[ijk] = merge_lists(lmo_to_paos[i], merge_lists(lmo_to_paos[j], lmo_to_paos[k]));
        for (int l = 0; l < naocc; ++l) {
            if (i_j_to_ij_[i][l] != -1 && i_j_to_ij_[j][l] != -1 && i_j_to_ij_[k][l] != -1)
                lmotriplet_to_lmos_ro_[ijk].push_back(l);
        }
    }

    std::array<int, 4> case_counts{};
    for (int ts = 0; ts < 4; ++ts) {
        case_counts[ts] = std::count(active_ijk_spin_[ts].begin(), active_ijk_spin_[ts].end(), true);
    }
    outfile->Printf("    Retained oriented RO triplets: AAA %d, AAB %d, BBA %d, BBB %d (%d shared domains)\n",
                    case_counts[0], case_counts[1], case_counts[2], case_counts[3], ntriplets);

    timer_off("RO Triples Sparsity");
}

void RO_DLPNOCCSD_T::ro_sort_triplets(double total_energy) {
    timer_on("RO Sort Triplets");
    const int ntriplets = ijk_to_i_j_k_ro_.size();
    std::vector<std::pair<int, double>> ranked(ntriplets);
    for (int ijk = 0; ijk < ntriplets; ++ijk) ranked[ijk] = {ijk, e_ijk_ro_[ijk]};
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& lhs, const auto& rhs) { return std::fabs(lhs.second) > std::fabs(rhs.second); });

    const double strong_scale = options_.get_double("T_CUT_TNO_STRONG_SCALE");
    const double weak_scale = options_.get_double("T_CUT_TNO_WEAK_SCALE");
    is_strong_triplet_ro_.assign(ntriplets, false);
    tno_scale_ro_.assign(ntriplets, weak_scale);

    double accumulated = 0.0;
    int nstrong = 0;
    const double target = 0.9 * std::fabs(total_energy);
    for (const auto& [ijk, energy] : ranked) {
        is_strong_triplet_ro_[ijk] = true;
        tno_scale_ro_[ijk] = strong_scale;
        accumulated += std::fabs(energy);
        ++nstrong;
        if (accumulated >= target) break;
    }
    outfile->Printf("    Number of strong RO triplet domains: %d of %d\n\n", nstrong, ntriplets);
    timer_off("RO Sort Triplets");
}

void RO_DLPNOCCSD_T::ro_tno_transform(double tno_tolerance) {
    timer_on("RO TNO Transform");

    const int naocc = nalpha_ - nfrzc();
    const int nbocc = nbeta_ - nfrzc();
    const int nsomo = naocc - nbocc;
    const int ntriplets = ijk_to_i_j_k_ro_.size();
    const int min_tnos = options_.get_int("MIN_TNOS");

    X_tno_ro_.assign(ntriplets, nullptr);
    n_tno_ro_.assign(ntriplets, 0);
    for (auto& fock : F_tno_spin_) fock.assign(ntriplets, nullptr);

    auto build_pair_density = [&](int pair) {
        const int npno = n_pno_[pair];
        auto density = std::make_shared<Matrix>(npno, npno);
        density->zero();

        auto add_density = [&](const SharedMatrix& amplitudes, double scale) {
            auto left = linalg::doublet(amplitudes, amplitudes, false, true);
            auto right = linalg::doublet(amplitudes, amplitudes, true, false);
            left->add(right);
            left->scale(scale);
            density->add(left);
        };

        // The spin trace includes both mixed-spin orientations and exactly recovers the established restricted
        // density D(T) + 1/2 D(T - T^T) when the alpha and beta amplitudes coincide.
        add_density(T_iajb_spin_[static_cast<int>(DoubleSpinCase::AA)][pair], 0.25);
        add_density(T_iajb_spin_[static_cast<int>(DoubleSpinCase::AB)][pair], 0.5);
        // D(T) = T T^T + T^T T is invariant to transposing T.
        // Use AB(ji) directly for BA(ij), avoiding an amplitude copy.
        add_density(T_iajb_spin_[static_cast<int>(DoubleSpinCase::AB)][ij_to_ji_[pair]], 0.5);
        add_density(T_iajb_spin_[static_cast<int>(DoubleSpinCase::BB)][pair], 0.25);
        auto& pair_indices = ij_to_i_j_[pair];
        if (pair_indices.first == pair_indices.second) density->scale(0.5);
        return density;
    };

    // Reuse frequently requested pair densities across triplets. Bound this
    // optional cache to 5% of the memory budget (and available headroom);
    // uncached pairs use the identical on-demand construction below.
    const int npairs = ij_to_i_j_.size();
    std::vector<size_t> uses(npairs, 0);
    for (const auto& triplet : ijk_to_i_j_k_ro_) {
        int i, j, k;
        std::tie(i, j, k) = triplet;
        ++uses[i_j_to_ij_[i][j]];
        ++uses[i_j_to_ij_[j][k]];
        ++uses[i_j_to_ij_[i][k]];
    }
    std::vector<int> ranked_pairs;
    for (int pair = 0; pair < npairs; ++pair) if (uses[pair] > 1) ranked_pairs.push_back(pair);
    std::stable_sort(ranked_pairs.begin(), ranked_pairs.end(),
                     [&](int left, int right) { return uses[left] > uses[right]; });
    const size_t budget = static_cast<size_t>(0.9 * memory_ / sizeof(double));
    const size_t fixed = ro_triples_fixed_memory();
    size_t cache_budget = fixed < budget ? std::min(static_cast<size_t>(memory_ / (20 * sizeof(double))),
                                                    (budget - fixed) / 2) : 0;
    std::vector<int> cached_pairs;
    for (int pair : ranked_pairs) {
        const size_t words = static_cast<size_t>(n_pno_[pair]) * n_pno_[pair];
        if (words > cache_budget) continue;
        cached_pairs.push_back(pair);
        cache_budget -= words;
    }
    std::vector<SharedMatrix> pair_densities(npairs);
#pragma omp parallel for schedule(dynamic, 1)
    for (int index = 0; index < static_cast<int>(cached_pairs.size()); ++index) {
        const int pair = cached_pairs[index];
        pair_densities[pair] = build_pair_density(pair);
    }

    auto F_pao_average = F_pao_a_->clone();
    F_pao_average->add(F_pao_b_);
    F_pao_average->scale(0.5);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < ntriplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_ro_[ijk];
        const auto& domain = lmotriplet_to_paos_ro_[ijk];
        const int npao = domain.size();
        const int nexternal_pao = npao - nsomo;

        auto retain_somo_block = [&]() {
            if (nsomo == 0 || npao < nsomo) return;
            auto X_tno = std::make_shared<Matrix>(npao, nsomo);
            for (int s = 0; s < nsomo; ++s) X_tno->set(npao - nsomo + s, s, 1.0);
            X_tno_ro_[ijk] = X_tno;
            n_tno_ro_[ijk] = nsomo;
            const std::array<SharedMatrix, 2> F_pao_spin = {F_pao_a_, F_pao_b_};
            for (SpinCase sigma : kSpins) {
                const int s = static_cast<int>(sigma);
                auto F_domain = submatrix_rows_and_cols(*F_pao_spin[s], domain, domain);
                F_tno_spin_[s][ijk] = linalg::triplet(X_tno, F_domain, X_tno, true, false, false);
                matrix_spin_enforcer_vv(F_tno_spin_[s][ijk], sigma);
            }
        };

        if (nexternal_pao <= 0) {
            retain_somo_block();
            continue;
        }

        const std::vector<int> external_domain(domain.begin(), domain.begin() + nexternal_pao);
        auto S_external = submatrix_rows_and_cols(*S_pao_, external_domain, external_domain);
        auto F_external = submatrix_rows_and_cols(*F_pao_average, external_domain, external_domain);

        SharedMatrix X_canonical_pao;
        SharedVector e_canonical_pao;
        std::tie(X_canonical_pao, e_canonical_pao) = orthocanonicalizer(S_external, F_external);
        auto F_canonical_pao = linalg::triplet(X_canonical_pao, F_external, X_canonical_pao, true, false, false);
        const int ncanonical = X_canonical_pao->ncol();
        if (ncanonical == 0) {
            retain_somo_block();
            continue;
        }

        auto project_pair_density = [&](int pair) {
            auto density = pair_densities[pair] ? pair_densities[pair] : build_pair_density(pair);
            auto S_pair_external = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[pair], external_domain);
            S_pair_external = linalg::doublet(S_pair_external, X_canonical_pao);
            auto S_pno_external = linalg::doublet(X_pno_[pair], S_pair_external, true, false);
            return linalg::triplet(S_pno_external, density, S_pno_external, true, false, false);
        };

        const int ij = i_j_to_ij_[i][j];
        const int jk = i_j_to_ij_[j][k];
        const int ik = i_j_to_ij_[i][k];
        auto triplet_density = project_pair_density(ij);
        triplet_density->add(project_pair_density(jk));
        triplet_density->add(project_pair_density(ik));
        triplet_density->scale(1.0 / 3.0);

        auto X_density = std::make_shared<Matrix>(ncanonical, ncanonical);
        auto occupations = std::make_shared<Vector>(ncanonical);
        triplet_density->diagonalize(*X_density, *occupations, descending);

        const int min_external = std::max(1, min_tnos - nsomo);
        int nexternal_tno = 0;
        for (int a = 0; a < ncanonical; ++a) {
            if (std::fabs(occupations->get(a)) >= tno_scale_ro_[ijk] * tno_tolerance || a < min_external)
                ++nexternal_tno;
        }
        nexternal_tno = std::min(ncanonical, std::max(1, nexternal_tno));

        Dimension zero(1);
        Dimension selected_dimension(1);
        selected_dimension.fill(nexternal_tno);
        auto X_selected = X_density->get_block({zero, X_density->rowspi()}, {zero, selected_dimension});

        SharedMatrix canonical_rotation;
        SharedVector selected_energies;
        std::tie(canonical_rotation, selected_energies) = canonicalizer(X_selected, F_canonical_pao);
        X_selected = linalg::doublet(X_selected, canonical_rotation);
        auto X_external_tno = linalg::doublet(X_canonical_pao, X_selected);

        const int ntno = nexternal_tno + nsomo;
        auto X_tno = std::make_shared<Matrix>(npao, ntno);
        for (int u = 0; u < nexternal_pao; ++u) {
            for (int a = 0; a < nexternal_tno; ++a) X_tno->set(u, a, X_external_tno->get(u, a));
        }
        // Preserve the SOMO block as explicit trailing orbitals so all existing spin masks remain exact.
        for (int s = 0; s < nsomo; ++s) X_tno->set(nexternal_pao + s, nexternal_tno + s, 1.0);

        X_tno_ro_[ijk] = X_tno;
        n_tno_ro_[ijk] = ntno;
        const std::array<SharedMatrix, 2> F_pao_spin = {F_pao_a_, F_pao_b_};
        for (SpinCase sigma : kSpins) {
            const int s = static_cast<int>(sigma);
            auto F_domain = submatrix_rows_and_cols(*F_pao_spin[s], domain, domain);
            F_tno_spin_[s][ijk] = linalg::triplet(X_tno, F_domain, X_tno, true, false, false);
            matrix_spin_enforcer_vv(F_tno_spin_[s][ijk], sigma);
        }
    }

    if (ntriplets == 0) {
        outfile->Printf("    No RO triples survived screening.\n\n");
        timer_off("RO TNO Transform");
        return;
    }
    int total = 0;
    int minimum = C_pao_->ncol();
    int maximum = 0;
    for (int ntno : n_tno_ro_) {
        total += ntno;
        minimum = std::min(minimum, ntno);
        maximum = std::max(maximum, ntno);
    }
    outfile->Printf("    RO TNOs per shared triplet domain: avg %d, min %d, max %d\n\n", total / ntriplets, minimum,
                    maximum);
    timer_off("RO TNO Transform");
}

double RO_DLPNOCCSD_T::compute_ro_lccsd_t0(bool save_memory) {
    timer_on("RO LCCSD(T0)");

    const int naocc = nalpha_ - nfrzc();
    const int nbocc = nbeta_ - nfrzc();
    const int nsomo = naocc - nbocc;
    const int ntriplets = ijk_to_i_j_k_ro_.size();

    for (int ts = 0; ts < 4; ++ts) {
        e_ijk_spin_[ts].assign(ntriplets, 0.0);
        if (save_memory) {
            W_iajbkc_spin_[ts].assign(ntriplets, nullptr);
            V_iajbkc_spin_[ts].assign(ntriplets, nullptr);
            T_iajbkc_spin_[ts].assign(ntriplets, nullptr);
            t_disk_generation_ro_[ts].assign(ntriplets, 0);
        }
    }
    e_ijk_ro_.assign(ntriplets, 0.0);

    const std::array<SharedMatrix, 2> F_lmo_spin = {F_lmo_a_, F_lmo_b_};
    const std::array<SharedMatrix, 2> F_lmo_pao_spin = {
        linalg::triplet(C_lmo_, F_reference_a_, C_pao_, true, false, false),
        linalg::triplet(C_lmo_, F_reference_b_, C_pao_, true, false, false)};

    std::vector<std::pair<int, size_t>> ranked(ntriplets);
#pragma omp parallel for
    for (int ijk = 0; ijk < ntriplets; ++ijk) {
        const size_t npao = lmotriplet_to_paos_ro_[ijk].size();
        const size_t naux = lmotriplet_to_ribfs_ro_[ijk].size();
        const size_t ntno = n_tno_ro_[ijk];
        ranked[ijk] = {ijk, naux * ntno * npao * (npao + ntno)};
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    double total_energy = 0.0;
    std::time_t start_time = std::time(nullptr);

    int init_threads = 1;
#ifdef _OPENMP
    init_threads = std::min(omp_get_max_threads(), std::max(1, ntriplets));
#endif
    if (toggle_memory_) {
        size_t resident = ro_triples_fixed_memory() +
                          2 * static_cast<size_t>(naocc) * C_pao_->ncol();
        if (save_memory) {
            const size_t generations = (!write_intermediates_ro_ ? 2 : 0) + (!write_amplitudes_ro_ ? 1 : 0);
            for (int ijk = 0; ijk < ntriplets; ++ijk) {
                const size_t n = n_tno_ro_[ijk];
                for (int ts = 0; ts < 4; ++ts)
                    if (active_ijk_spin_[ts][ijk]) resident += generations * n * n * n;
            }
        }
        const size_t budget = static_cast<size_t>(0.9 * memory_ / sizeof(double));
        const size_t workspace = ro_triples_init_workspace();
        if (workspace > 0) {
            if (resident >= budget || workspace > budget - resident)
                throw PSIEXCEPTION("Too little memory for one RO-DLPNO-(T0) initialization worker");
            const int allowed = static_cast<int>(std::min(static_cast<size_t>(init_threads),
                                                          (budget - resident) / workspace));
            if (allowed < init_threads)
                outfile->Printf("    Limiting RO triples initialization to %d threads (memory budget).\n", allowed);
            init_threads = allowed;
        }
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(init_threads) reduction(+ : total_energy)
    for (int rank = 0; rank < ntriplets; ++rank) {
        const int ijk = ranked[rank].first;
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_ro_[ijk];
        const std::array<int, 3> occupied = {i, j, k};
        const int ntno = n_tno_ro_[ijk];
        if (ntno == 0) continue;

        const int nlmo = lmotriplet_to_lmos_ro_[ijk].size();
        const int npao = lmotriplet_to_paos_ro_[ijk].size();
        const int naux = lmotriplet_to_ribfs_ro_[ijk].size();

        std::array<SharedMatrix, 3> q_ov;
        std::array<SharedMatrix, 3> q_oo;
        for (int p = 0; p < 3; ++p) {
            q_ov[p] = std::make_shared<Matrix>(naux, npao);
            q_oo[p] = std::make_shared<Matrix>(naux, nlmo);
        }
        auto q_vv = std::make_shared<Matrix>(naux, ntno * ntno);

        // Reuse the PAO work buffer for every auxiliary function.  The old
        // path allocated a new npao-by-npao matrix for every q_local.
        auto q_vv_pao = std::make_shared<Matrix>(npao, npao);
        for (int q_local = 0; q_local < naux; ++q_local) {
            const int q = lmotriplet_to_ribfs_ro_[ijk][q_local];
            const int center = ribasis_->function_to_center(q);
            for (int l_local = 0; l_local < nlmo; ++l_local) {
                const int l = lmotriplet_to_lmos_ro_[ijk][l_local];
                for (int p = 0; p < 3; ++p) {
                    q_oo[p]->set(q_local, l_local,
                                 qij_[q]->get(riatom_to_lmos_ext_dense_[center][occupied[p]],
                                              riatom_to_lmos_ext_dense_[center][l]));
                }
            }
            for (int u_local = 0; u_local < npao; ++u_local) {
                const int u = lmotriplet_to_paos_ro_[ijk][u_local];
                for (int p = 0; p < 3; ++p) {
                    q_ov[p]->set(q_local, u_local,
                                 qia_[q]->get(riatom_to_lmos_ext_dense_[center][occupied[p]],
                                              riatom_to_paos_ext_dense_[center][u]));
                }
            }

            q_vv_pao->zero();
            for (int u_local = 0; u_local < npao; ++u_local) {
                const int u = lmotriplet_to_paos_ro_[ijk][u_local];
                for (int v_local = 0; v_local < npao; ++v_local) {
                    const int v = lmotriplet_to_paos_ro_[ijk][v_local];
                    const int uv = riatom_to_pao_pairs_dense_[center][u][v];
                    if (uv != -1) q_vv_pao->set(u_local, v_local, qab_[q]->get(uv, 0));
                }
            }
            auto q_vv_tno =
                linalg::triplet(X_tno_ro_[ijk], q_vv_pao, X_tno_ro_[ijk], true, false, false);
            ::memcpy(q_vv->get_pointer() + q_local * ntno * ntno, q_vv_tno->get_pointer(),
                     ntno * ntno * sizeof(double));
        }

        q_vv_pao.reset();
        for (int p = 0; p < 3; ++p) q_ov[p] = linalg::doublet(q_ov[p], X_tno_ro_[ijk]);
        auto metric =
            submatrix_rows_and_cols(*full_metric_, lmotriplet_to_ribfs_ro_[ijk], lmotriplet_to_ribfs_ro_[ijk]);

        // Factor the Coulomb metric once for all three occupied rows instead
        // of repeating the same DGESV factorization for each right-hand side.
        auto q_ov_metric_rhs = std::make_shared<Matrix>(naux, 3 * ntno);
        for (int q_local = 0; q_local < naux; ++q_local) {
            for (int p = 0; p < 3; ++p) {
                ::memcpy(q_ov_metric_rhs->get_pointer() + q_local * 3 * ntno + p * ntno,
                         q_ov[p]->get_pointer() + q_local * ntno, ntno * sizeof(double));
            }
        }
        C_DGESV_wrapper(metric->clone(), q_ov_metric_rhs);
        std::array<SharedMatrix, 3> q_ov_metric;
        for (int p = 0; p < 3; ++p) q_ov_metric[p] = std::make_shared<Matrix>(naux, ntno);
        for (int q_local = 0; q_local < naux; ++q_local) {
            for (int p = 0; p < 3; ++p) {
                ::memcpy(q_ov_metric[p]->get_pointer() + q_local * ntno,
                         q_ov_metric_rhs->get_pointer() + q_local * 3 * ntno + p * ntno,
                         ntno * sizeof(double));
            }
        }

        q_ov_metric_rhs.reset();
        metric->power(0.5, 1.0e-14);

        // The inverse-square-root metric transform has six right-hand-side
        // blocks.  Solve them together so DGESV performs one factorization.
        const int metric_rhs_stride = ntno + nlmo;
        auto metric_rhs = std::make_shared<Matrix>(naux, 3 * metric_rhs_stride);
        for (int q_local = 0; q_local < naux; ++q_local) {
            for (int p = 0; p < 3; ++p) {
                const int offset = p * metric_rhs_stride;
                ::memcpy(metric_rhs->get_pointer() + q_local * 3 * metric_rhs_stride + offset,
                         q_ov[p]->get_pointer() + q_local * ntno, ntno * sizeof(double));
                ::memcpy(metric_rhs->get_pointer() + q_local * 3 * metric_rhs_stride + offset + ntno,
                         q_oo[p]->get_pointer() + q_local * nlmo, nlmo * sizeof(double));
            }
        }
        C_DGESV_wrapper(metric->clone(), metric_rhs);
        for (int q_local = 0; q_local < naux; ++q_local) {
            for (int p = 0; p < 3; ++p) {
                const int offset = p * metric_rhs_stride;
                ::memcpy(q_ov[p]->get_pointer() + q_local * ntno,
                         metric_rhs->get_pointer() + q_local * 3 * metric_rhs_stride + offset,
                         ntno * sizeof(double));
                ::memcpy(q_oo[p]->get_pointer() + q_local * nlmo,
                         metric_rhs->get_pointer() + q_local * 3 * metric_rhs_stride + offset + ntno,
                         nlmo * sizeof(double));
            }
        }

        metric_rhs.reset();
        metric.reset();

        // These twelve spatial integral blocks are formed once and reused by every spin case.
        std::array<SharedMatrix, 3> K_ovvv;
        for (int p = 0; p < 3; ++p) K_ovvv[p] = linalg::doublet(q_ov_metric[p], q_vv, true, false);

        std::array<std::array<SharedMatrix, 3>, 3> K_ooov{};
        K_ooov[0][1] = linalg::doublet(q_oo[0], q_ov[1], true, false);
        K_ooov[1][0] = linalg::doublet(q_oo[1], q_ov[0], true, false);
        K_ooov[2][1] = linalg::doublet(q_oo[2], q_ov[1], true, false);
        K_ooov[1][2] = linalg::doublet(q_oo[1], q_ov[2], true, false);
        K_ooov[0][2] = linalg::doublet(q_oo[0], q_ov[2], true, false);
        K_ooov[2][0] = linalg::doublet(q_oo[2], q_ov[0], true, false);

        std::array<std::array<SharedMatrix, 3>, 3> K_ovov{};
        K_ovov[0][1] = linalg::doublet(q_ov[0], q_ov[1], true, false);
        K_ovov[1][0] = K_ovov[0][1]->transpose();
        K_ovov[0][2] = linalg::doublet(q_ov[0], q_ov[2], true, false);
        K_ovov[2][0] = K_ovov[0][2]->transpose();
        K_ovov[1][2] = linalg::doublet(q_ov[1], q_ov[2], true, false);
        K_ovov[2][1] = K_ovov[1][2]->transpose();

        // These auxiliary buffers need not coexist with the spin moments,
        // projected pair cache, and packed occupied-contraction workspace.
        q_vv.reset();
        for (int p = 0; p < 3; ++p) {
            q_ov[p].reset();
            q_oo[p].reset();
            q_ov_metric[p].reset();
        }

        std::unordered_map<long long, SharedMatrix> projected_pairs;
        auto project_pair = [&](int p, int q, SpinCase spin_p, SpinCase spin_q) -> SharedMatrix {
            const long long key = (((static_cast<long long>(p) * naocc + q) * 2 + static_cast<int>(spin_p)) * 2 +
                                   static_cast<int>(spin_q));
            auto found = projected_pairs.find(key);
            if (found != projected_pairs.end()) return found->second;

            auto projected = std::make_shared<Matrix>(ntno, ntno);
            if ((spin_p == SpinCase::Beta && p >= nbocc) || (spin_q == SpinCase::Beta && q >= nbocc)) {
                projected_pairs[key] = projected;
                return projected;
            }
            const int pair = i_j_to_ij_[p][q];
            if (pair == -1) {
                projected_pairs[key] = projected;
                return projected;
            }
            auto S_pair_triplet = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[pair], lmotriplet_to_paos_ro_[ijk]);
            S_pair_triplet = linalg::doublet(S_pair_triplet, X_tno_ro_[ijk]);
            auto S_pno_triplet = linalg::doublet(X_pno_[pair], S_pair_triplet, true, false);
            projected = linalg::triplet(S_pno_triplet, T_iajb_spin_helper(pair, spin_p, spin_q), S_pno_triplet, true,
                                        false, false);
            const int alpha_end = ntno - nsomo;
            if (spin_p == SpinCase::Alpha) {
                for (int a = alpha_end; a < ntno; ++a) projected->zero_row(0, a);
            }
            if (spin_q == SpinCase::Alpha) {
                for (int b = alpha_end; b < ntno; ++b) projected->zero_column(0, b);
            }
            projected_pairs[key] = projected;
            return projected;
        };

        std::unordered_map<long long, SharedMatrix> projected_singles;
        auto project_single = [&](int p, SpinCase spin) -> SharedMatrix {
            const long long key = static_cast<long long>(p) * 2 + static_cast<int>(spin);
            auto found = projected_singles.find(key);
            if (found != projected_singles.end()) return found->second;
            auto projected = std::make_shared<Matrix>(ntno, 1);
            if (spin == SpinCase::Beta && p >= nbocc) {
                projected_singles[key] = projected;
                return projected;
            }
            const int pp = i_j_to_ij_[p][p];
            auto S_pair_triplet = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[pp], lmotriplet_to_paos_ro_[ijk]);
            S_pair_triplet = linalg::doublet(S_pair_triplet, X_tno_ro_[ijk]);
            auto S_pno_triplet = linalg::doublet(X_pno_[pp], S_pair_triplet, true, false);
            projected = linalg::doublet(S_pno_triplet, T_ia_spin_[static_cast<int>(spin)][p], true, false);
            if (spin == SpinCase::Alpha) {
                for (int a = ntno - nsomo; a < ntno; ++a) projected->set(a, 0, 0.0);
            }
            projected_singles[key] = projected;
            return projected;
        };

        std::unordered_map<long long, SharedMatrix> projected_fia;
        auto project_fia = [&](int p, SpinCase spin) -> SharedMatrix {
            const long long key = static_cast<long long>(p) * 2 + static_cast<int>(spin);
            auto found = projected_fia.find(key);
            if (found != projected_fia.end()) return found->second;
            auto projected = std::make_shared<Matrix>(ntno, 1);
            if (spin == SpinCase::Beta && p >= nbocc) {
                projected_fia[key] = projected;
                return projected;
            }
            auto Fia_domain = submatrix_rows_and_cols(*F_lmo_pao_spin[static_cast<int>(spin)], std::vector<int>{p},
                                                      lmotriplet_to_paos_ro_[ijk]);
            Fia_domain = linalg::doublet(Fia_domain, X_tno_ro_[ijk]);
            projected = Fia_domain->transpose();
            if (spin == SpinCase::Alpha) {
                for (int a = ntno - nsomo; a < ntno; ++a) projected->set(a, 0, 0.0);
            }
            projected_fia[key] = projected;
            return projected;
        };

        auto contraction_work = std::make_shared<Matrix>(ntno, ntno * ntno);
        auto pair_stack = std::make_shared<Matrix>(nlmo, ntno * ntno);
        // Pack projected pairs as (l,a*b), then sum the local occupied index
        // with GEMM as well. project_pair() already caches each spin block.
        auto add_ooov_t2 = [&](const SharedMatrix& result, const SharedMatrix& K, int fixed,
                               SpinCase first_spin, SpinCase second_spin, bool local_first,
                               const std::array<int, 3>& axes, double scale) {
            if (nlmo == 0) return;
            const size_t nn = static_cast<size_t>(ntno) * ntno;
            for (int l_local = 0; l_local < nlmo; ++l_local) {
                const int l = lmotriplet_to_lmos_ro_[ijk][l_local];
                const auto T = local_first ? project_pair(l, fixed, first_spin, second_spin)
                                           : project_pair(fixed, l, first_spin, second_spin);
                ::memcpy(pair_stack->get_pointer() + l_local * nn, T->get_pointer(), nn * sizeof(double));
            }
            C_DGEMM('T', 'N', ntno, ntno * ntno, nlmo, 1.0, K->get_pointer(), ntno,
                    pair_stack->get_pointer(), ntno * ntno, 0.0, contraction_work->get_pointer(), ntno * ntno);
            ro_add_permuted_cube(result, contraction_work, ntno, axes, scale);
        };

        auto pure_parent = [&](int p, int q, int r, SpinCase spin) {
            auto parent = std::make_shared<Matrix>(ntno, ntno * ntno);
            const auto T_qr = project_pair(occupied[q], occupied[r], spin, spin);
            // -K_p(c,b,d) T_qr(d,a) - K_p(b,c,d) T_qr(a,d)
            ro_contract_vvv_t2(K_ovvv[p], T_qr, contraction_work, ntno, false);
            ro_add_permuted_cube(parent, contraction_work, ntno, {2, 1, 0}, -1.0);
            ro_contract_vvv_t2(K_ovvv[p], T_qr, contraction_work, ntno, true);
            ro_add_permuted_cube(parent, contraction_work, ntno, {1, 2, 0}, -1.0);
            add_ooov_t2(parent, K_ooov[q][r], occupied[p], spin, spin, false, {0, 2, 1}, 1.0);
            add_ooov_t2(parent, K_ooov[r][q], occupied[p], spin, spin, false, {0, 1, 2}, 1.0);
            return parent;
        };

        auto form_pure_w = [&](SpinCase spin) {
            const std::array<std::array<int, 3>, 3> occupied_permutations = {
                std::array<int, 3>{0, 1, 2}, std::array<int, 3>{1, 0, 2}, std::array<int, 3>{2, 1, 0}};
            const std::array<double, 3> signs = {1.0, -1.0, -1.0};
            std::array<SharedMatrix, 3> parents;
            for (int p = 0; p < 3; ++p) {
                parents[p] = pure_parent(occupied_permutations[p][0], occupied_permutations[p][1],
                                         occupied_permutations[p][2], spin);
            }
            auto W = std::make_shared<Matrix>(ntno, ntno * ntno);
            for (int a = 0; a < ntno; ++a) {
                for (int b = 0; b < ntno; ++b) {
                    for (int c = 0; c < ntno; ++c) {
                        double value = 0.0;
                        for (int p = 0; p < 3; ++p) {
                            value +=
                                signs[p] * (cube_get(parents[p], ntno, a, b, c) - cube_get(parents[p], ntno, b, a, c) -
                                            cube_get(parents[p], ntno, c, b, a));
                        }
                        cube_set(W, ntno, a, b, c, value);
                    }
                }
            }
            return W;
        };

        auto pure_v_parent = [&](int p, int q, int r, SpinCase spin) {
            auto parent = std::make_shared<Matrix>(ntno, ntno * ntno);
            const auto T_p = project_single(occupied[p], spin);
            const auto F_p = project_fia(occupied[p], spin);
            const auto T_qr = project_pair(occupied[q], occupied[r], spin, spin);
            for (int a = 0; a < ntno; ++a) {
                for (int b = 0; b < ntno; ++b) {
                    for (int c = 0; c < ntno; ++c) {
                        const double value = (K_ovov[q][r]->get(b, c) - K_ovov[q][r]->get(c, b)) * T_p->get(a, 0) +
                                             F_p->get(a, 0) * T_qr->get(b, c);
                        cube_set(parent, ntno, a, b, c, value);
                    }
                }
            }
            return parent;
        };

        auto form_pure_v = [&](SpinCase spin) {
            const std::array<std::array<int, 3>, 3> occupied_permutations = {
                std::array<int, 3>{0, 1, 2}, std::array<int, 3>{1, 0, 2}, std::array<int, 3>{2, 1, 0}};
            const std::array<double, 3> signs = {1.0, -1.0, -1.0};
            std::array<SharedMatrix, 3> parents;
            for (int p = 0; p < 3; ++p) {
                parents[p] = pure_v_parent(occupied_permutations[p][0], occupied_permutations[p][1],
                                           occupied_permutations[p][2], spin);
            }
            auto V = std::make_shared<Matrix>(ntno, ntno * ntno);
            for (int a = 0; a < ntno; ++a) {
                for (int b = 0; b < ntno; ++b) {
                    for (int c = 0; c < ntno; ++c) {
                        double value = 0.0;
                        for (int p = 0; p < 3; ++p) {
                            value +=
                                signs[p] * (cube_get(parents[p], ntno, a, b, c) - cube_get(parents[p], ntno, b, a, c) -
                                            cube_get(parents[p], ntno, c, b, a));
                        }
                        cube_set(V, ntno, a, b, c, value);
                    }
                }
            }
            return V;
        };

        auto mixed_pij_block = [&](int p, int q, SpinCase like, SpinCase opposite) {
            constexpr int r = 2;
            auto block = std::make_shared<Matrix>(ntno, ntno * ntno);
            const auto T_qr = project_pair(occupied[q], occupied[r], like, opposite);
            const auto T_pr = project_pair(occupied[p], occupied[r], like, opposite);
            ro_contract_vvv_t2(K_ovvv[p], T_qr, contraction_work, ntno, false);
            ro_add_permuted_cube(block, contraction_work, ntno, {0, 1, 2}, 1.0);
            ro_contract_vvv_t2(K_ovvv[q], T_pr, contraction_work, ntno, true);
            ro_add_permuted_cube(block, contraction_work, ntno, {1, 2, 0}, 1.0);
            ro_contract_vvv_t2(K_ovvv[p], T_qr, contraction_work, ntno, true);
            ro_add_permuted_cube(block, contraction_work, ntno, {0, 2, 1}, 1.0);
            ro_contract_vvv_t2(K_ovvv[q], T_pr, contraction_work, ntno, false);
            ro_add_permuted_cube(block, contraction_work, ntno, {1, 0, 2}, 1.0);
            add_ooov_t2(block, K_ooov[q][r], occupied[p], like, like, false, {2, 0, 1}, -1.0);
            add_ooov_t2(block, K_ooov[r][p], occupied[q], like, opposite, false, {0, 1, 2}, -1.0);
            add_ooov_t2(block, K_ooov[r][q], occupied[p], like, opposite, false, {1, 0, 2}, -1.0);
            return block;
        };

        auto mixed_pab_block = [&](SpinCase like, SpinCase opposite) {
            constexpr int p = 0, q = 1, r = 2;
            auto block = std::make_shared<Matrix>(ntno, ntno * ntno);
            const auto T_pq = project_pair(occupied[p], occupied[q], like, like);
            ro_contract_vvv_t2(K_ovvv[r], T_pq, contraction_work, ntno, true);
            ro_add_permuted_cube(block, contraction_work, ntno, {2, 1, 0}, 1.0);
            add_ooov_t2(block, K_ooov[q][p], occupied[r], like, opposite, true, {0, 1, 2}, -1.0);
            add_ooov_t2(block, K_ooov[p][q], occupied[r], like, opposite, true, {1, 0, 2}, -1.0);
            return block;
        };

        auto form_mixed_w = [&](SpinCase like, SpinCase opposite) {
            const auto direct = mixed_pij_block(0, 1, like, opposite);
            const auto exchanged = mixed_pij_block(1, 0, like, opposite);
            const auto particle = mixed_pab_block(like, opposite);
            auto W = std::make_shared<Matrix>(ntno, ntno * ntno);
            for (int a = 0; a < ntno; ++a) {
                for (int b = 0; b < ntno; ++b) {
                    for (int c = 0; c < ntno; ++c) {
                        cube_set(W, ntno, a, b, c,
                                 cube_get(direct, ntno, a, b, c) - cube_get(exchanged, ntno, a, b, c) +
                                     cube_get(particle, ntno, a, b, c) - cube_get(particle, ntno, b, a, c));
                    }
                }
            }
            return W;
        };

        auto mixed_v_pij_block = [&](int p, int q, SpinCase like, SpinCase opposite) {
            constexpr int r = 2;
            auto block = std::make_shared<Matrix>(ntno, ntno * ntno);
            const auto T_p = project_single(occupied[p], like);
            const auto T_q = project_single(occupied[q], like);
            const auto T_r = project_single(occupied[r], opposite);
            const auto F_p = project_fia(occupied[p], like);
            const auto F_q = project_fia(occupied[q], like);
            const auto T_qr = project_pair(occupied[q], occupied[r], like, opposite);
            const auto T_pr = project_pair(occupied[p], occupied[r], like, opposite);
            for (int a = 0; a < ntno; ++a) {
                for (int b = 0; b < ntno; ++b) {
                    for (int c = 0; c < ntno; ++c) {
                        const double value = K_ovov[q][r]->get(b, c) * T_p->get(a, 0) +
                                             K_ovov[p][r]->get(a, c) * T_q->get(b, 0) +
                                             K_ovov[p][q]->get(a, b) * T_r->get(c, 0) +
                                             F_p->get(a, 0) * T_qr->get(b, c) + F_q->get(b, 0) * T_pr->get(a, c);
                        cube_set(block, ntno, a, b, c, value);
                    }
                }
            }
            return block;
        };

        auto form_mixed_v = [&](SpinCase like, SpinCase opposite) {
            constexpr int p = 0, q = 1, r = 2;
            const auto direct = mixed_v_pij_block(p, q, like, opposite);
            const auto exchanged = mixed_v_pij_block(q, p, like, opposite);
            const auto F_r = project_fia(occupied[r], opposite);
            const auto T_pq = project_pair(occupied[p], occupied[q], like, like);
            auto V = std::make_shared<Matrix>(ntno, ntno * ntno);
            // The direct-minus-exchanged blocks already contain all four
            // P(ij)P(ab) [t_i^a (jb|kc)] terms. Adding an unpermuted copy
            // would double count one term and break occupied-orbital invariance.
            for (int a = 0; a < ntno; ++a) {
                for (int b = 0; b < ntno; ++b) {
                    for (int c = 0; c < ntno; ++c) {
                        cube_set(V, ntno, a, b, c,
                                 cube_get(direct, ntno, a, b, c) - cube_get(exchanged, ntno, a, b, c) +
                                     F_r->get(c, 0) * T_pq->get(a, b));
                    }
                }
            }
            return V;
        };

        for (TripleSpinCase spin : kTripleSpins) {
            const int ts = static_cast<int>(spin);
            if (!active_ijk_spin_[ts][ijk]) continue;
            const auto spins = get_spin_triple(spin);

            SharedMatrix W;
            SharedMatrix V;
            if (spin == TripleSpinCase::AAA) {
                W = form_pure_w(SpinCase::Alpha);
                V = form_pure_v(SpinCase::Alpha);
            } else if (spin == TripleSpinCase::BBB) {
                W = form_pure_w(SpinCase::Beta);
                V = form_pure_v(SpinCase::Beta);
            } else {
                const SpinCase like = spins[0];
                const SpinCase opposite = spins[2];
                W = form_mixed_w(like, opposite);
                V = form_mixed_v(like, opposite);
            }
            W->set_name(std::string("RO W ") + triple_spin_label(spin) + " " + std::to_string(ijk));
            V->set_name(std::string("RO V ") + triple_spin_label(spin) + " " + std::to_string(ijk));
            triples_spin_enforcer(W, spin, i, j, k);
            triples_spin_enforcer(V, spin, i, j, k);

            auto T = W->clone();
            T->set_name(std::string("RO T ") + triple_spin_label(spin) + " " + std::to_string(ijk));
            for (int a = 0; a < ntno; ++a) {
                for (int b = 0; b < ntno; ++b) {
                    for (int c = 0; c < ntno; ++c) {
                        const std::array<int, 3> virtuals = {a, b, c};
                        bool allowed = true;
                        for (int axis = 0; axis < 3; ++axis) {
                            if (spins[axis] == SpinCase::Alpha && virtuals[axis] >= ntno - nsomo) allowed = false;
                        }
                        if (!allowed) {
                            cube_set(T, ntno, a, b, c, 0.0);
                            continue;
                        }
                        const double denominator = F_lmo_spin[static_cast<int>(spins[0])]->get(i, i) +
                                                   F_lmo_spin[static_cast<int>(spins[1])]->get(j, j) +
                                                   F_lmo_spin[static_cast<int>(spins[2])]->get(k, k) -
                                                   F_tno_spin_[static_cast<int>(spins[0])][ijk]->get(a, a) -
                                                   F_tno_spin_[static_cast<int>(spins[1])][ijk]->get(b, b) -
                                                   F_tno_spin_[static_cast<int>(spins[2])][ijk]->get(c, c);
                        cube_set(T, ntno, a, b, c,
                                 std::fabs(denominator) > 1.0e-14 ? cube_get(W, ntno, a, b, c) / denominator : 0.0);
                    }
                }
            }
            triples_spin_enforcer(T, spin, i, j, k);

            const double prefactor =
                (spin == TripleSpinCase::AAA || spin == TripleSpinCase::BBB) ? 1.0 / 6.0 : 1.0 / 2.0;
            e_ijk_spin_[ts][ijk] = prefactor * (W->vector_dot(T) + V->vector_dot(T));
            e_ijk_ro_[ijk] += e_ijk_spin_[ts][ijk];
            total_energy += e_ijk_spin_[ts][ijk];

            if (!save_memory) continue;
            if (write_intermediates_ro_) {
#pragma omp critical(ro_triples_psio)
                {
                    W->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                    V->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                }
            } else {
                W_iajbkc_spin_[ts][ijk] = W;
                V_iajbkc_spin_[ts][ijk] = V;
            }
            if (write_amplitudes_ro_) {
#pragma omp critical(ro_triples_psio)
                T->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
            } else {
                T_iajbkc_spin_[ts][ijk] = T;
            }
        }
    }

    if (save_memory) {
        outfile->Printf("    RO iterative triples initialization completed in %d s\n\n",
                        static_cast<int>(std::time(nullptr) - start_time));
    } else {
        outfile->Printf("    RO semicanonical (T0) completed in %d s\n\n",
                        static_cast<int>(std::time(nullptr) - start_time));
    }
    timer_off("RO LCCSD(T0)");
    return total_energy;
}

size_t RO_DLPNOCCSD_T::ro_triples_fixed_memory() const {
    // Count resident matrices by allocation, not by ordered-pair alias.
    // CCSD-only intermediates have already been released at this point.
    size_t words = qij_memory_ + qia_memory_ + qab_memory_;
    std::unordered_set<const Matrix*> counted;
    auto add = [&](const SharedMatrix& matrix) {
        if (matrix && counted.insert(matrix.get()).second)
            for (int h = 0; h < matrix->nirrep(); ++h) words += matrix->size(h);
    };
    for (const auto& blocks : T_ia_spin_) for (const auto& matrix : blocks) add(matrix);
    for (const auto& blocks : T_iajb_spin_) for (const auto& matrix : blocks) add(matrix);
    for (const auto& matrix : X_pno_) add(matrix);
    for (const auto& matrix : X_tno_ro_) add(matrix);
    for (const auto& blocks : F_tno_spin_) for (const auto& matrix : blocks) add(matrix);
    for (const auto& matrix : {C_lmo_, C_lmo_previous_, C_pao_, S_pao_, F_pao_, F_pao_a_, F_pao_b_,
                               F_lmo_, F_lmo_a_, F_lmo_b_, F_lmo_pao_, F_ao_, F_reference_a_, F_reference_b_,
                               full_metric_, DOI_ij_, DOI_iu_, DOI_uv_, dipole_pair_e_, dipole_pair_e_bound_,
                               er_thc_x_ao_, er_thc_Z_}) add(matrix);
    // AO reference data can be shared with the wavefunction above.
    for (const auto& matrix : {reference_wavefunction_->Ca(), reference_wavefunction_->Cb(),
                               reference_wavefunction_->Fa(), reference_wavefunction_->Fb(),
                               reference_wavefunction_->Da(), reference_wavefunction_->Db(),
                               reference_wavefunction_->S(), reference_wavefunction_->H()}) add(matrix);
    return words;
}

size_t RO_DLPNOCCSD_T::ro_triples_init_workspace() const {
    size_t maximum = 0;
    for (int ijk = 0; ijk < static_cast<int>(n_tno_ro_.size()); ++ijk) {
        const size_t n = n_tno_ro_[ijk];
        if (n == 0) continue;
        const size_t p = lmotriplet_to_paos_ro_[ijk].size();
        const size_t q = lmotriplet_to_ribfs_ro_[ijk].size();
        const size_t l = lmotriplet_to_lmos_ro_[ijk].size();
        // DF/metric transforms and numerator construction are separate peaks:
        // the auxiliary work arrays are released before building W/V/T.
        const size_t integrals = q * (n * n + 3 * p + 9 * n + 6 * l) + 3 * q * q +
                                 3 * p * p + 6 * p * n + 3 * n * n * n;
        // Three K_ovvv, permutation parents, W/V/T, one BLAS work cube,
        // packed local pairs, and all cached projected spin pairs. The
        // bound also covers pair-projection and singles/Fia work arrays.
        const size_t moments = 10 * n * n * n + (13 * l + 36) * n * n +
                               3 * p * p + 6 * p * n + 12 * l * n;
        maximum = std::max(maximum, std::max(integrals, moments));
    }
    return maximum;
}

void RO_DLPNOCCSD_T::ro_estimate_memory() {
    size_t cube_elements = 0;
    size_t max_cube_elements = 0;
    size_t active_spin_blocks = 0;
    for (int ijk = 0; ijk < static_cast<int>(ijk_to_i_j_k_ro_.size()); ++ijk) {
        const size_t n = n_tno_ro_[ijk];
        const size_t elements = n * n * n;
        for (int ts = 0; ts < 4; ++ts) {
            if (active_ijk_spin_[ts][ijk]) {
                cube_elements += elements;
                max_cube_elements = std::max(max_cube_elements, elements);
                ++active_spin_blocks;
            }
        }
    }

    write_intermediates_ro_ = options_.get_bool("WRITE_TRIPLES_INTERMEDIATES");
    write_amplitudes_ro_ = options_.get_bool("WRITE_TRIPLES_AMPLITUDES");
    // An explicit request to write amplitudes places both Jacobi generations
    // on disk.  Automatic memory control may instead select the hybrid mode
    // with current T resident and only next T on disk.
    write_next_amplitudes_ro_ = write_amplitudes_ro_;

    const size_t integral_memory = qij_memory_ + qia_memory_ + qab_memory_;
    const size_t fixed_memory = ro_triples_fixed_memory();
    const size_t w_memory = cube_elements;
    const size_t v_memory = cube_elements;
    const size_t current_t_memory = cube_elements;
    const size_t next_t_memory = cube_elements;

    size_t nthreads = 1;
#ifdef _OPENMP
    nthreads = static_cast<size_t>(omp_get_max_threads());
#endif
    // Per worker, allow room for W, a cloned T, the residual, neighbor
    // permutation/projection buffers, and the transient promoted T block.
    // This is deliberately conservative; persistent generations are counted
    // separately below.
    constexpr size_t workspace_cubes_per_thread = 8;
    const size_t iteration_workspace = workspace_cubes_per_thread * max_cube_elements;
    // Initialization can use fewer workers than the iterative solver. Reserve
    // at least one initialization worker here; compute_ro_lccsd_t0 selects
    // the largest concurrency that fits after its resident W/V/T are counted.
    const size_t init_workspace = ro_triples_init_workspace();
    const size_t workspace_memory = std::max(nthreads * iteration_workspace,
                                             (toggle_memory_ ? 1 : nthreads) * init_workspace);
    // The two bare LMO-PAO blocks are local to compute_ro_lccsd_t0().
    const size_t fock_workspace = 2 * static_cast<size_t>(nalpha_ - nfrzc()) * C_pao_->ncol();

    const size_t all_in_core_memory =
        fixed_memory + w_memory + v_memory + current_t_memory + next_t_memory + workspace_memory + fock_workspace;
    size_t resident_memory = fixed_memory + workspace_memory + fock_workspace;
    if (!write_intermediates_ro_) resident_memory += w_memory + v_memory;
    if (!write_amplitudes_ro_) resident_memory += current_t_memory;
    if (!write_next_amplitudes_ro_) resident_memory += next_t_memory;

    const auto exceeds_memory = [&]() {
        return static_cast<double>(resident_memory) * sizeof(double) > 0.9 * static_cast<double>(memory_);
    };

    // Immutable W/V are consumed once per block and are therefore the
    // highest-throughput tensors to spill.  Preserve current T in memory as
    // long as possible because every residual can request several neighbor
    // amplitudes.
    if (toggle_memory_ && !write_intermediates_ro_ && exceeds_memory()) {
        write_intermediates_ro_ = true;
        resident_memory -= w_memory + v_memory;
    }
    if (toggle_memory_ && !write_next_amplitudes_ro_ && exceeds_memory()) {
        write_next_amplitudes_ro_ = true;
        resident_memory -= next_t_memory;
    }
    if (toggle_memory_ && !write_amplitudes_ro_ && exceeds_memory()) {
        write_amplitudes_ro_ = true;
        write_next_amplitudes_ro_ = true;
        resident_memory -= current_t_memory;
    }
    if (toggle_memory_ && exceeds_memory())
        throw PSIEXCEPTION("Too little memory for the RO-DLPNO-(T) intermediates");

    constexpr double bytes_to_gb = 1.0e-9;
    const auto words_to_gb = [&](size_t words) { return words * sizeof(double) * bytes_to_gb; };
    outfile->Printf("\n  ==> RO-DLPNO-(T) Memory Requirements <==\n\n");
    outfile->Printf("    Active spin-resolved T cubes : %zu\n", active_spin_blocks);
    outfile->Printf("    Integral storage              : %8.3f [GB]\n", words_to_gb(integral_memory));
    outfile->Printf("    Resident orbitals/amplitudes   : %8.3f [GB]\n", words_to_gb(fixed_memory - integral_memory));
    outfile->Printf("    W generation                  : %8.3f [GB]\n", words_to_gb(w_memory));
    outfile->Printf("    V generation                  : %8.3f [GB]\n", words_to_gb(v_memory));
    outfile->Printf("    Current T generation          : %8.3f [GB]\n", words_to_gb(current_t_memory));
    outfile->Printf("    Next T generation             : %8.3f [GB]\n", words_to_gb(next_t_memory));
    outfile->Printf("    Initialization/iteration work : %8.3f [GB] (%zu iteration threads)\n",
                    words_to_gb(workspace_memory + fock_workspace), nthreads);
    outfile->Printf("    All-in-core requirement       : %8.3f [GB]\n", words_to_gb(all_in_core_memory));
    outfile->Printf("    Selected resident memory      : %8.3f [GB]\n", words_to_gb(resident_memory));
    outfile->Printf("    Automatic memory limit (90%%)  : %8.3f [GB]\n", 0.9 * memory_ * bytes_to_gb);
    outfile->Printf("    Total memory available        : %8.3f [GB]\n\n", memory_ * bytes_to_gb);
    if (write_intermediates_ro_) outfile->Printf("    Writing RO W and V intermediates to disk.\n");
    if (write_amplitudes_ro_) {
        outfile->Printf("    Writing current and next RO T generations to disk.\n");
    } else if (write_next_amplitudes_ro_) {
        outfile->Printf("    Keeping current RO T amplitudes in memory.\n");
        outfile->Printf("    Writing only the next RO T Jacobi generation to disk.\n");
    } else {
        outfile->Printf("    Keeping both RO T Jacobi generations in memory.\n");
    }
    outfile->Printf("\n");
}

std::string RO_DLPNOCCSD_T::ro_t_amplitude_name(TripleSpinCase spin, int index, bool next) const {
    const int ts = static_cast<int>(spin);
    const bool alternate = write_amplitudes_ro_ && t_disk_generation_ro_[ts][index];
    return std::string("RO T ") + ((alternate != next) ? "next " : "") +
           triple_spin_label(spin) + " " + std::to_string(index);
}

double RO_DLPNOCCSD_T::compute_ro_t_iteration_energy() {
    timer_on("RO Compute (T) Energy");
    const int ntriplets = ijk_to_i_j_k_ro_.size();
    for (auto& energies : e_ijk_spin_) energies.assign(ntriplets, 0.0);
    e_ijk_ro_.assign(ntriplets, 0.0);

    double total_energy = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total_energy)
    for (int ijk = 0; ijk < ntriplets; ++ijk) {
        const int ntno = n_tno_ro_[ijk];
        if (ntno == 0) continue;
        for (TripleSpinCase spin : kTripleSpins) {
            const int ts = static_cast<int>(spin);
            if (!active_ijk_spin_[ts][ijk]) continue;

            auto load_matrix = [&](const char* kind, bool from_disk,
                                   const std::array<std::vector<SharedMatrix>, 4>& matrices) {
                if (!from_disk) return matrices[ts][ijk];
                auto matrix = std::make_shared<Matrix>(
                    std::strcmp(kind, "T") == 0 ? ro_t_amplitude_name(spin, ijk) :
                        std::string("RO ") + kind + " " + triple_spin_label(spin) + " " + std::to_string(ijk), ntno,
                    ntno * ntno);
#pragma omp critical(ro_triples_psio)
                matrix->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                return matrix;
            };

            auto W = load_matrix("W", write_intermediates_ro_, W_iajbkc_spin_);
            auto V = load_matrix("V", write_intermediates_ro_, V_iajbkc_spin_);
            auto T = load_matrix("T", write_amplitudes_ro_, T_iajbkc_spin_);
            const double prefactor =
                (spin == TripleSpinCase::AAA || spin == TripleSpinCase::BBB) ? 1.0 / 6.0 : 1.0 / 2.0;
            e_ijk_spin_[ts][ijk] = prefactor * (W->vector_dot(T) + V->vector_dot(T));
            e_ijk_ro_[ijk] += e_ijk_spin_[ts][ijk];
            total_energy += e_ijk_spin_[ts][ijk];
        }
    }
    timer_off("RO Compute (T) Energy");
    return total_energy;
}

double RO_DLPNOCCSD_T::ro_lccsd_t_iterations() {
    timer_on("RO LCCSD(T) Iterations");

    const int naocc = nalpha_ - nfrzc();
    const int nbocc = nbeta_ - nfrzc();
    const int nsomo = naocc - nbocc;
    const int ntriplets = ijk_to_i_j_k_ro_.size();
    const std::array<SharedMatrix, 2> F_lmo_spin = {F_lmo_a_, F_lmo_b_};
    const double fock_cutoff = options_.get_double("F_CUT_T");
    const double iteration_cutoff = options_.get_double("T_CUT_ITER");

    outfile->Printf("\n  ==> Iterative RO-DLPNO-(T) <==\n\n");
    outfile->Printf("                         (T) Energy      Delta E       Max R     Time (s)\n");

    std::array<std::vector<double>, 4> old_spin_energies;
    for (auto& energies : old_spin_energies) energies.assign(ntriplets, 0.0);
    double current_energy = 0.0;
    for (const auto& energies : e_ijk_spin_)
        for (double energy : energies) current_energy += energy;

    const int max_iterations = options_.get_int("DLPNO_MAXITER");
    bool energy_converged = false;
    bool residual_converged = false;
    int iteration = 1;

    while (!(energy_converged && residual_converged)) {
        const std::time_t iteration_start = std::time(nullptr);
        std::array<std::vector<double>, 4> residual_rms;
        for (auto& residuals : residual_rms) residuals.assign(ntriplets, 0.0);
        std::array<std::vector<SharedMatrix>, 4> next_amplitudes;
        if (!write_next_amplitudes_ro_) next_amplitudes = T_iajbkc_spin_;
        std::array<std::vector<char>, 4> amplitude_updated;
        for (auto& updated : amplitude_updated) updated.assign(ntriplets, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int ijk = 0; ijk < ntriplets; ++ijk) {
            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_ro_[ijk];
            const std::array<int, 3> occupied = {i, j, k};
            const int ntno = n_tno_ro_[ijk];
            if (ntno == 0) continue;

            for (TripleSpinCase spin : kTripleSpins) {
                const int ts = static_cast<int>(spin);
                if (!active_ijk_spin_[ts][ijk]) continue;
                if (iteration > 1 && std::fabs(e_ijk_spin_[ts][ijk] - old_spin_energies[ts][ijk]) <
                                         std::fabs(old_spin_energies[ts][ijk] * iteration_cutoff))
                    continue;

                auto load_matrix = [&](const char* kind, bool from_disk,
                                       const std::array<std::vector<SharedMatrix>, 4>& matrices, int index) {
                    if (!from_disk) return matrices[ts][index];
                    const int n = n_tno_ro_[index];
                    auto matrix = std::make_shared<Matrix>(
                        std::strcmp(kind, "T") == 0 ? ro_t_amplitude_name(spin, index) :
                            std::string("RO ") + kind + " " + triple_spin_label(spin) + " " + std::to_string(index), n,
                        n * n);
#pragma omp critical(ro_triples_psio)
                    matrix->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                    return matrix;
                };

                auto W = load_matrix("W", write_intermediates_ro_, W_iajbkc_spin_, ijk);
                auto T = load_matrix("T", write_amplitudes_ro_, T_iajbkc_spin_, ijk)->clone();
                auto residual = W->clone();
                residual->set_name("RO triples residual");
                const auto spins = get_spin_triple(spin);

                // Full virtual Fock action in the common RO TNO basis.  Its diagonal gives the T0 denominator;
                // retaining the off-diagonal part makes the iterative result invariant to the common spatial basis.
                const auto F_a = F_tno_spin_[static_cast<int>(spins[0])][ijk];
                const auto F_b = F_tno_spin_[static_cast<int>(spins[1])][ijk];
                const auto F_c = F_tno_spin_[static_cast<int>(spins[2])][ijk];
                const double occupied_diagonal = F_lmo_spin[static_cast<int>(spins[0])]->get(i, i) +
                                                 F_lmo_spin[static_cast<int>(spins[1])]->get(j, j) +
                                                 F_lmo_spin[static_cast<int>(spins[2])]->get(k, k);
                ro_add_virtual_fock(residual, T, {F_a, F_b, F_c}, occupied_diagonal, ntno);

                // Off-diagonal occupied Fock couplings preserve the spin carried by the replaced slot.
                for (int axis = 0; axis < 3; ++axis) {
                    const int s = static_cast<int>(spins[axis]);
                    const int noccupied = spins[axis] == SpinCase::Alpha ? naocc : nbocc;
                    for (int l = 0; l < noccupied; ++l) {
                        if (l == occupied[axis]) continue;
                        const double coupling = F_lmo_spin[s]->get(l, occupied[axis]);
                        if (std::fabs(coupling) < fock_cutoff) continue;

                        auto requested = occupied;
                        requested[axis] = l;
                        const int key = triple_key(requested[0], requested[1], requested[2], naocc);
                        const auto neighbor_position = i_j_k_to_ijk_spin_[ts].find(key);
                        if (neighbor_position == i_j_k_to_ijk_spin_[ts].end()) continue;
                        const int neighbor = neighbor_position->second;
                        if (n_tno_ro_[neighbor] == 0) continue;

                        auto neighbor_T = load_matrix("T", write_amplitudes_ro_, T_iajbkc_spin_, neighbor);
                        neighbor_T = ro_triples_permuter(neighbor_T, spin, requested[0], requested[1], requested[2]);

                        auto overlap = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_ro_[ijk],
                                                               lmotriplet_to_paos_ro_[neighbor]);
                        overlap = linalg::triplet(X_tno_ro_[ijk], overlap, X_tno_ro_[neighbor], true, false, false);
                        auto projected_neighbor =
                            ro_matmul_3d(neighbor_T, overlap, n_tno_ro_[neighbor], n_tno_ro_[ijk]);
                        C_DAXPY(static_cast<size_t>(ntno) * ntno * ntno, -coupling, projected_neighbor->get_pointer(), 1,
                                residual->get_pointer(), 1);
                    }
                }

                triples_spin_enforcer(residual, spin, i, j, k);
                double residual_squared = 0.0;
                size_t allowed_elements = 0;
                for (int a = 0; a < ntno; ++a) {
                    for (int b = 0; b < ntno; ++b) {
                        for (int c = 0; c < ntno; ++c) {
                            const std::array<int, 3> virtuals = {a, b, c};
                            bool allowed = true;
                            for (int axis = 0; axis < 3; ++axis) {
                                if (spins[axis] == SpinCase::Alpha && virtuals[axis] >= ntno - nsomo) allowed = false;
                            }
                            if (!allowed) continue;
                            const double residual_element = cube_get(residual, ntno, a, b, c);
                            residual_squared += residual_element * residual_element;
                            ++allowed_elements;
                            const double denominator =
                                F_a->get(a, a) + F_b->get(b, b) + F_c->get(c, c) - occupied_diagonal;
                            if (std::fabs(denominator) > 1.0e-14)
                                cube_add(T, ntno, a, b, c, -residual_element / denominator);
                        }
                    }
                }
                triples_spin_enforcer(T, spin, i, j, k);
                residual_rms[ts][ijk] = allowed_elements == 0 ? 0.0 : std::sqrt(residual_squared / allowed_elements);
                amplitude_updated[ts][ijk] = 1;

                if (write_next_amplitudes_ro_) {
                    T->set_name(ro_t_amplitude_name(spin, ijk, true));
#pragma omp critical(ro_triples_psio)
                    T->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                } else {
                    next_amplitudes[ts][ijk] = T;
                }
            }
        }

        // Jacobi semantics require every residual to read the same amplitude generation.  Promote the completed
        // generation only after all triplets have finished reading their neighbors.
        if (write_amplitudes_ro_) {
            // Swap disk slots only after the parallel residual build finishes.
            // This replaces a serial read/rewrite of every updated cube. A
            // skipped block retains its slot, so neighbors never read stale T.
            for (int ts = 0; ts < 4; ++ts)
                for (int ijk = 0; ijk < ntriplets; ++ijk)
                    if (amplitude_updated[ts][ijk]) t_disk_generation_ro_[ts][ijk] ^= 1;
        } else if (write_next_amplitudes_ro_) {
            for (TripleSpinCase spin : kTripleSpins) {
                const int ts = static_cast<int>(spin);
                for (int ijk = 0; ijk < ntriplets; ++ijk) {
                    if (!amplitude_updated[ts][ijk]) continue;
                    const int ntno = n_tno_ro_[ijk];
                    auto T = std::make_shared<Matrix>(
                        ro_t_amplitude_name(spin, ijk, true), ntno,
                        ntno * ntno);
                    T->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::SubBlocks);
                    T->set_name(std::string("RO T ") + triple_spin_label(spin) + " " + std::to_string(ijk));
                    // All residuals have completed. Replace one current block
                    // at a time without materializing a second in-core generation.
                    T_iajbkc_spin_[ts][ijk] = T;
                }
            }
        } else {
            T_iajbkc_spin_ = std::move(next_amplitudes);
        }

        const double previous_energy = current_energy;
        old_spin_energies = e_ijk_spin_;
        current_energy = compute_ro_t_iteration_energy();

        double max_residual = 0.0;
        for (const auto& residuals : residual_rms)
            for (double value : residuals) max_residual = std::max(max_residual, value);

        energy_converged = std::fabs(current_energy - previous_energy) < options_.get_double("E_CONVERGENCE");
        residual_converged = max_residual < options_.get_double("R_CONVERGENCE");
        outfile->Printf("  @RO-LCCSD(T) iter %3d: %16.12f %10.3e %10.3e %8d\n", iteration, current_energy,
                        current_energy - previous_energy, max_residual,
                        static_cast<int>(std::time(nullptr) - iteration_start));

        if (++iteration > max_iterations + 1) throw PSIEXCEPTION("Maximum RO-DLPNO-(T) iterations exceeded");
    }

    timer_off("RO LCCSD(T) Iterations");
    return current_energy;
}

double RO_DLPNOCCSD_T::compute_energy() { return DLPNOCCSD::compute_energy(); }

void RO_DLPNOCCSD_T::post_ccsd_correction(DLPNOCCSDPhase phase) {
    compute_ro_triples_correction();
    const std::string suffix = options_.get_bool("T0_APPROXIMATION") ? "(T0)" : "(T)";
    if (phase == DLPNOCCSDPhase::FinalBrueckner) {
        set_scalar_variable("BCCD" + suffix + " TOTAL ENERGY", scalar_variable("CURRENT ENERGY"));
        set_scalar_variable("BCCD" + suffix + " CORRELATION ENERGY", scalar_variable("CURRENT CORRELATION ENERGY"));
        set_scalar_variable("DLPNO-BCCD" + suffix + " TOTAL ENERGY", scalar_variable("CURRENT ENERGY"));
        set_scalar_variable("DLPNO-BCCD" + suffix + " CORRELATION ENERGY", scalar_variable("CURRENT CORRELATION ENERGY"));
        outfile->Printf("\n  @Total RO-DLPNO-BCCD%s Energy: %16.12f\n", suffix.c_str(), scalar_variable("CURRENT ENERGY"));
    }
}

double RO_DLPNOCCSD_T::compute_ro_triples_correction() {
    require_converged_brueckner_reference(3, "RO-CCSD(T)");
    timer_on("RO-DLPNO-CCSD(T)");

    // Release the CCSD-only integral, overlap, and dressed-intermediate caches before allocating triples cubes.
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
    for (auto& block : T_n_ij_spin_) block.clear();
    for (auto& block : i_Qk_t1_spin_) block.clear();
    for (auto& block : i_Qa_t1_spin_) block.clear();
    for (auto& block : F_pno_spin_) block.clear();
    for (auto& block : Fkc_tilde_spin_) block.clear();
    for (auto& block : Fai_tilde_spin_) block.clear();
    for (auto& block : Fab_tilde_spin_) block.clear();
    for (auto& block : Fki_tilde_spin_) block.reset();
    K_iajb_.clear();
    T_iajb_.clear();
    Tt_iajb_.clear();
    T_ia_.clear();
    for (auto& block : T_iajb_srolmp2_spin_) block.clear();
    Fkj_.reset();

    psio_->open(PSIF_DLPNO_TRIPLES, PSIO_OPEN_NEW);
    ro_print_header();

    const double tno_pre = options_.get_double("T_CUT_TNO_PRE");
    const double tno_cutoff = options_.get_double("T_CUT_TNO");

    outfile->Printf("\n   Starting spin-resolved RO triplet prescreening...\n");
    ro_triples_sparsity(true);
    ro_tno_transform(tno_pre);
    compute_ro_lccsd_t0();

    outfile->Printf("\n   Continuing with RO triplets above %6.3e Eh...\n", options_.get_double("T_CUT_TRIPLES_WEAK"));
    ro_triples_sparsity(false);
    outfile->Printf("    Screened RO triplet contribution: %.12f\n\n", de_lccsd_t_screened_ro_);
    ro_tno_transform(tno_cutoff);
    const double accurate_t0 = compute_ro_lccsd_t0();

    auto sum_current_spin_energies = [&]() {
        std::array<double, 4> sums{};
        for (int ts = 0; ts < 4; ++ts) {
            sums[ts] = de_lccsd_t_screened_spin_ro_[ts];
            for (double energy : e_ijk_spin_[ts]) sums[ts] += energy;
        }
        return sums;
    };
    std::array<double, 4> final_spin_energy = sum_current_spin_energies();
    e_lccsd_t_ro_ = e_lccsd_ + accurate_t0 + de_lccsd_t_screened_ro_;

    outfile->Printf("    RO-DLPNO-CCSD(T0) correlation energy: %16.12f\n", e_lccsd_t_ro_);
    outfile->Printf("      DLPNO-CCSD contribution:            %16.12f\n", e_lccsd_);
    outfile->Printf("      RO-DLPNO-(T0) contribution:         %16.12f\n", accurate_t0);
    outfile->Printf("      Screened triplet contribution:      %16.12f\n\n", de_lccsd_t_screened_ro_);

    if (!options_.get_bool("T0_APPROXIMATION")) {
        outfile->Printf("\n  ==> Computing Full Iterative RO-DLPNO-(T) <==\n\n");
        ro_sort_triplets(accurate_t0);
        ro_tno_transform(tno_cutoff);
        ro_estimate_memory();

        const double loose_t0 = compute_ro_lccsd_t0(true);
        std::array<double, 4> loose_t0_spin{};
        for (int ts = 0; ts < 4; ++ts)
            for (double energy : e_ijk_spin_[ts]) loose_t0_spin[ts] += energy;

        E_T_ro_ = ro_lccsd_t_iterations();
        std::array<double, 4> iterative_spin{};
        for (int ts = 0; ts < 4; ++ts)
            for (double energy : e_ijk_spin_[ts]) iterative_spin[ts] += energy;

        const double iterative_increment = E_T_ro_ - loose_t0;
        e_lccsd_t_ro_ += iterative_increment;
        for (int ts = 0; ts < 4; ++ts) final_spin_energy[ts] += iterative_spin[ts] - loose_t0_spin[ts];

        outfile->Printf("    RO-DLPNO-(T0), iterative TNO space: %16.12f\n", loose_t0);
        outfile->Printf("    RO-DLPNO-(T),  iterative TNO space: %16.12f\n", E_T_ro_);
        outfile->Printf("    Net iterative triples increment:    %16.12f\n\n", iterative_increment);
    }

    const double triples_energy = e_lccsd_t_ro_ - e_lccsd_;
    const double correlation_energy = e_lccsd_t_ro_ + de_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;
    const double total_energy = reference_energy_ + correlation_energy;

    if (options_.get_bool("T0_APPROXIMATION")) {
        set_scalar_variable("CCSD(T0) TOTAL ENERGY", total_energy);
        set_scalar_variable("CCSD(T0) CORRELATION ENERGY", correlation_energy);
        set_scalar_variable("(T0) CORRECTION ENERGY", triples_energy);
    }
    set_scalar_variable("CURRENT REFERENCE ENERGY", reference_energy_);
    set_scalar_variable("CCSD(T) CORRELATION ENERGY", correlation_energy);
    set_scalar_variable("CURRENT CORRELATION ENERGY", correlation_energy);
    set_scalar_variable("CCSD(T) TOTAL ENERGY", total_energy);
    set_scalar_variable("CURRENT ENERGY", total_energy);
    set_scalar_variable("(T) CORRECTION ENERGY", triples_energy);
    set_scalar_variable("AAA (T) CORRECTION ENERGY", final_spin_energy[0]);
    set_scalar_variable("AAB (T) CORRECTION ENERGY", final_spin_energy[1]);
    set_scalar_variable("ABB (T) CORRECTION ENERGY", final_spin_energy[2]);
    set_scalar_variable("BBB (T) CORRECTION ENERGY", final_spin_energy[3]);
    set_scalar_variable("DLPNO SEMICANONICAL (T0) ENERGY", accurate_t0 + de_lccsd_t_screened_ro_);
    set_scalar_variable("DLPNO SCREENED TRIPLETS ENERGY", de_lccsd_t_screened_ro_);

    ro_print_results();
    psio_->close(PSIF_DLPNO_TRIPLES, 0);
    timer_off("RO-DLPNO-CCSD(T)");
    return total_energy;
}

void RO_DLPNOCCSD_T::ro_print_results() {
    const double triples = e_lccsd_t_ro_ - e_lccsd_;
    const double ccsd_correlation = e_lccsd_ + de_weak_ + de_lmp2_eliminated_ + de_pno_total_ + de_dipole_;
    const double total_correlation = ccsd_correlation + triples;
    outfile->Printf("\n  Total RO-DLPNO-CCSD(T) Correlation Energy: %16.12f\n", total_correlation);
    outfile->Printf("    RO-DLPNO-CCSD contribution:              %16.12f\n", ccsd_correlation);
    outfile->Printf("    RO-DLPNO-(T) contribution:               %16.12f\n", triples);
    outfile->Printf("      AAA: %16.12f\n", scalar_variable("AAA (T) CORRECTION ENERGY"));
    outfile->Printf("      AAB: %16.12f\n", scalar_variable("AAB (T) CORRECTION ENERGY"));
    outfile->Printf("      ABB: %16.12f\n", scalar_variable("ABB (T) CORRECTION ENERGY"));
    outfile->Printf("      BBB: %16.12f\n", scalar_variable("BBB (T) CORRECTION ENERGY"));
    outfile->Printf("    Screened triplet contribution:           %16.12f\n", de_lccsd_t_screened_ro_);
    outfile->Printf("\n  @Total RO-DLPNO-CCSD(T) Energy: %16.12f\n\n", reference_energy_ + total_correlation);
}

}  // namespace dlpno
}  // namespace psi
