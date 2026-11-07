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

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/twobody.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace psi {
namespace dlpno {
// One Cartesian component on ordered subsets of AO and auxiliary functions.
// The two-/three-center engines omit the zero shells in their derivative buffers.
std::pair<SharedMatrix, SharedMatrix> df_mixed_integrals(SharedWavefunction wfn, const std::string& aux_name,
        const std::vector<int>& aux_ids, const std::vector<int>& ao_ids, int u, int v) {
    if (!wfn) throw PSIEXCEPTION("A wavefunction is required for mixed DF integrals.");
    auto basis = wfn->basisset(), aux = wfn->get_basisset(aux_name);
    const int ncoord = 3 * wfn->molecule()->natom();
    if (u < 0 || u >= ncoord || v < 0 || v >= ncoord)
        throw PSIEXCEPTION("Invalid Cartesian coordinate for a mixed DF derivative.");
    using ShellMap = std::map<int, std::vector<std::pair<int, int>>>;
    auto shells = [](const std::shared_ptr<BasisSet>& bs, const std::vector<int>& ids) {
        ShellMap selected;
        std::unordered_set<int> seen;
        for (size_t i = 0; i < ids.size(); ++i) {
            const int bf = ids[i];
            if (bf < 0 || bf >= bs->nbf() || !seen.insert(bf).second)
                throw PSIEXCEPTION("Expected distinct in-range basis indices for a mixed DF derivative.");
            const int shell = bs->function_to_shell(bf);
            selected[shell].emplace_back(bf - bs->shell(shell).function_index(), i);
        }
        return selected;
    };
    const auto qs = shells(aux, aux_ids), ms = shells(basis, ao_ids);
    const int nq = aux_ids.size(), nb = ao_ids.size();
    if (nb && nb > std::numeric_limits<int>::max() / nb)
        throw PSIEXCEPTION("Mixed DF AO block exceeds the matrix column limit.");
    auto metric = std::make_shared<Matrix>(nq,nq);
    auto raw = std::make_shared<Matrix>(nq,nb * nb);
    raw->set_numpy_shape({nq,nb,nb});
    if (!nq) return {metric,raw};
    auto zero = BasisSet::zero_ao_basis_set();
    IntegralFactory mf(aux,zero,aux,zero), qf(aux,zero,basis,basis);
    std::vector<ShellMap::const_iterator> rows;
    for (auto it = qs.begin(); it != qs.end(); ++it) rows.push_back(it);
    int threads = 1;
#ifdef _OPENMP
    threads = std::max(1, std::min(int(rows.size()), Process::environment.get_n_threads()));
#endif
    std::vector<std::unique_ptr<TwoBodyAOInt>> metrics(threads), triples(threads);
    // Construct engines serially; each worker owns its buffers and scratch.
    for (int t = 0; t < threads; ++t) {
        metrics[t] = mf.eri(2);
        triples[t] = qf.eri(2);
    }
    std::vector<std::exception_ptr> failures(rows.size());
    auto components = [u,v](std::initializer_list<int> centers) {
        std::vector<int> indices;
        // Equal physical centers must be summed over all ordered slots,
        // including both orders when the two nuclear coordinates coincide.
        const int dim = 3 * centers.size();
        if (std::all_of(centers.begin(), centers.end(), [&](int c){ return c == *centers.begin(); }))
            return indices;  // rigid translation of all centers
        for (auto i = centers.begin(); i != centers.end(); ++i) {
            if (*i != u / 3) continue;
            for (auto j = centers.begin(); j != centers.end(); ++j) {
                if (*j != v / 3) continue;
                int a = 3 * (i-centers.begin()) + u%3, b = 3 * (j-centers.begin()) + v%3;
                if (a > b) std::swap(a,b);
                indices.push_back(a*(2*dim-a-1)/2+b);
            }
        }
        return indices;
    };
    // Shell P owns distinct auxiliary output rows, even for reordered subsets.
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (int row = 0; row < int(rows.size()); ++row) {
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        const auto& mi = metrics[thread];
        const auto& qi = triples[thread];
        const auto& [P,pids] = *rows[row];
        try {
            const int cp = aux->shell_to_center(P);
            for (const auto& [Q,qids] : qs) {
                const auto parts = components({cp,aux->shell_to_center(Q)});
                if (parts.empty() || !mi->compute_shell_deriv2(P,0,Q,0)) continue;
                for (int part : parts) {
                    const auto* buf = mi->buffers()[part];
                    for (const auto& [p,i] : pids)
                        for (const auto& [q,j] : qids)
                            (*metric)(i,j) += buf[p*aux->shell(Q).nfunction()+q];
                }
            }
            for (const auto& [M,mids] : ms) {
                for (const auto& [N,nids] : ms) {
                    const auto parts = components({cp,basis->shell_to_center(M),basis->shell_to_center(N)});
                    if (parts.empty() || !qi->compute_shell_deriv2(P,0,M,N)) continue;
                    const int nm = basis->shell(M).nfunction(), nn = basis->shell(N).nfunction();
                    for (int part : parts) {
                        const auto* buf = qi->buffers()[part];
                        for (const auto& [p,i] : pids)
                            for (const auto& [m,j] : mids)
                                for (const auto& [n,k] : nids)
                                    (*raw)(i,j*nb+k) += buf[(p*nm+m)*nn+n];
                    }
                }
            }
        } catch (...) { failures[row] = std::current_exception(); }
    }
    for (const auto& failure : failures)
        if (failure) std::rethrow_exception(failure);
    return {metric,raw};
}


SharedMatrix DLPNOCCSD_T::triples_df_response(int r, const SharedMatrix& w_bar, const SharedMatrix& v_bar,
                                              std::vector<ThreeCenterDensity>& three_center,
                                              const SharedMatrix& x_bar) {
    const auto [i, j, k] = ijk_to_i_j_k_[r];
    const std::array<int, 3> occ{i, j, k};
    const int n = n_tno_[r];
    const auto& lmos = lmotriplet_to_lmos_[r];
    const auto& aux = lmotriplet_to_ribfs_[r];
    struct PairProjection {
        SharedMatrix map, overlap, t, seed;
    };
    std::map<int, PairProjection> pairs;
    auto pair = [&](int ij) -> PairProjection& {
        auto& p = pairs[ij];
        if (!p.overlap) {
            auto metric = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmotriplet_to_paos_[r]);
            p.map = linalg::doublet(X_pno_[ij], metric, true, false);
            p.overlap = linalg::doublet(p.map, X_tno_[r]);
            p.t = linalg::triplet(p.overlap, T_iajb_[ij], p.overlap, true, false, false);
            p.seed = std::make_shared<Matrix>(n, n);
        }
        return p;
    };
    auto pull_back_pairs = [&]() {
        for (auto& [ij, p] : pairs) {
            triples_source2_[ij]->add(linalg::triplet(p.overlap, p.seed, p.overlap, false, false, true));
            auto ds = linalg::triplet(T_iajb_[ij], p.overlap, p.seed, false, false, true);
            ds->add(linalg::triplet(T_iajb_[ij], p.overlap, p.seed, true, false, false));
            x_bar->add(linalg::doublet(p.map, ds, true, false));
            reverse_pair_triplet_overlap(ij, r, X_tno_[r], ds);
        }
    };
    // The forward V also contains f_i^a t_jk^bc and its two permutations.
    // The (i,k) pair for the middle term occurs in precisely that order.
    for (int p = 0; p < 3; ++p) {
        const int q = p == 0 ? 1 : 0, s = p == 2 ? 1 : 2;
        auto& block = pair(i_j_to_ij_[occ[q]][occ[s]]);
        auto fp = submatrix_rows_and_cols(*F_lmo_pao_, {occ[p]}, lmotriplet_to_paos_[r]);
        auto f = linalg::doublet(fp, X_tno_[r]);
        auto df = std::make_shared<Matrix>(1, n);
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                for (int c = 0; c < n; ++c) {
                    const std::array<int, 3> x{a, b, c};
                    const double v = (*v_bar)(a, b * n + c);
                    (*block.seed)(x[q], x[s]) += v * (*f)(0, x[p]);
                    (*df)(0, x[p]) += v * (*block.t)(x[q], x[s]);
                }
        x_bar->add(linalg::doublet(fp, df, true, false));
        add_local_block(triples_fov_bar_, {occ[p]}, lmotriplet_to_paos_[r],
                        linalg::doublet(df, X_tno_[r], false, true));
    }
    if (aux.empty()) {
        pull_back_pairs();
        return std::make_shared<Matrix>(0, 0);
    }
    auto raw = triples_df_blocks(r);
    auto full_bar = std::make_shared<Matrix>(aux.size(), aux.size());
    auto cutoff_bar = full_bar->clone();

    auto metric = submatrix_rows_and_cols(*full_metric_, aux, aux);
    auto U = metric->clone();
    auto eval = std::make_shared<Vector>(aux.size());
    metric->diagonalize(U, eval);
    const double cut = 1.e-14 * std::max(std::abs((*eval)(0)), std::abs((*eval)(aux.size() - 1)));
    auto weighted = U->clone();
    for (int a = 0; a < aux.size(); ++a) {
        double v = (*eval)(a);
        if (v == 0.0) throw PSIEXCEPTION("Singular full Coulomb metric for a DLPNO (T) triplet.");
        weighted->scale_column(0, a, v > 0.0 && v >= cut ? 1.0 / v : 0.0);
    }
    auto inverse_cut = linalg::doublet(weighted, U, false, true);

    TriplesDFBlocks bar, fitted;
    std::array<SharedMatrix, 3> full_ov;
    auto full_vv = raw.vv->clone();
    C_DGESV_wrapper(metric->clone(), full_vv);
    for (int p = 0; p < 3; ++p) {
        full_ov[p] = raw.ov[p]->clone();
        C_DGESV_wrapper(metric->clone(), full_ov[p]);
        fitted.ov[p] = linalg::doublet(inverse_cut, raw.ov[p]);
        fitted.oo[p] = linalg::doublet(inverse_cut, raw.oo[p]);
        bar.ov[p] = std::make_shared<Matrix>(aux.size(), n);
        bar.oo[p] = std::make_shared<Matrix>(aux.size(), lmos.size());
    }
    bar.vv = std::make_shared<Matrix>(aux.size(), n * n);

    // Accumulate adjoints of the full inverse used by (ia|bd), and of the
    // rank-filtered inverse used by the symmetric half transforms, separately.
    // Undo the six explicit permutations of W (Eq. 109). They act on
    // virtual slots even when two occupied labels are identical.
    std::array<int, 3> perm{0, 1, 2};
    do {
        const auto [p, q, s] = perm;
        auto g = std::make_shared<Matrix>(n * n, n);
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                for (int c = 0; c < n; ++c) {
                    const std::array<int, 3> x{a, b, c};
                    (*g)(x[p] * n + x[q], x[s]) = (*w_bar)(a, b * n + c);
                }

        auto& kj = pair(i_j_to_ij_[occ[s]][occ[q]]);
        auto K = linalg::doublet(full_ov[p], raw.vv, true, false);
        K->reshape(n * n, n);
        kj.seed->add(linalg::doublet(g, K, true, false));
        auto ovvv = linalg::doublet(g, kj.t);
        ovvv->reshape(n, n * n);
        full_bar->add(linalg::triplet(raw.ov[p], ovvv, raw.vv, false, false, true));
        bar.ov[p]->add(linalg::doublet(full_vv, ovvv, false, true));
        bar.vv->add(linalg::doublet(full_ov[p], ovvv));

        auto ooov = std::make_shared<Matrix>(lmos.size(), n);
        auto Klc = linalg::doublet(raw.oo[q], fitted.ov[s], true, false);
        for (int l = 0; l < lmos.size(); ++l) {
            auto& il = pair(i_j_to_ij_[occ[p]][lmos[l]]);
            auto grad = linalg::doublet(g, submatrix_rows(*Klc, {l}), false, true);
            grad->reshape(n, n);
            il.seed->subtract(grad);
            auto t = il.t->clone();
            t->reshape(1, n * n);
            auto row = linalg::doublet(t, g);
            for (int c = 0; c < n; ++c) (*ooov)(l, c) = -(*row)(0, c);
        }
        cutoff_bar->add(linalg::triplet(raw.oo[q], ooov, raw.ov[s], false, false, true));
        bar.oo[q]->add(linalg::doublet(fitted.ov[s], ooov, false, true));
        bar.ov[s]->add(linalg::doublet(fitted.oo[q], ooov));
    } while (std::next_permutation(perm.begin(), perm.end()));

    // The three t1 * (ov|ov) terms of V (Eq. 110). F_ov * t2 is independent
    // of this auxiliary metric when the Fock matrix and orbital bases are fixed.
    for (int p = 0; p < 3; ++p) {
        const int q = (p + 1) % 3, s = (p + 2) % 3;
        auto& ii = pair(i_j_to_ij_[occ[p]][occ[p]]);
        auto singles = linalg::doublet(ii.overlap, T_ia_[occ[p]], true, false);
        auto ds = std::make_shared<Matrix>(n, 1);
        auto K = linalg::doublet(raw.ov[q], fitted.ov[s], true, false);
        auto ovov = std::make_shared<Matrix>(n, n);
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                for (int c = 0; c < n; ++c) {
                    const std::array<int, 3> x{a, b, c};
                    (*ovov)(x[q], x[s]) += (*singles)(x[p], 0) * (*v_bar)(a, b * n + c);
                    (*ds)(x[p], 0) += (*K)(x[q], x[s]) * (*v_bar)(a, b * n + c);
                }
        triples_source1_[occ[p]]->add(linalg::doublet(ii.overlap, ds));
        auto overlap_bar = linalg::doublet(T_ia_[occ[p]], ds, false, true);
        x_bar->add(linalg::doublet(ii.map, overlap_bar, true, false));
        reverse_pair_triplet_overlap(i_j_to_ij_[occ[p]][occ[p]], r, X_tno_[r], overlap_bar);
        cutoff_bar->add(linalg::triplet(raw.ov[q], ovov, raw.ov[s], false, false, true));
        bar.ov[q]->add(linalg::doublet(fitted.ov[s], ovov, false, true));
        bar.ov[s]->add(linalg::doublet(fitted.ov[q], ovov));
    }

    pull_back_pairs();
    accumulate_three_center(r, bar, three_center, x_bar);
    auto full = linalg::triplet(U, full_bar, U, true, false, false);
    auto filtered = linalg::triplet(U, cutoff_bar, U, true, false, false);
    // Matrix::power(-0.5, 1.e-14) retains positive metric eigenvalues
    // relative to the spectral radius. On a fixed-rank branch the derivative
    // of f(J) uses divided differences, including kept/discarded rotations.
    for (int u = 0; u < aux.size(); ++u) {
        const double a = (*eval)(u);
        for (int v = 0; v < aux.size(); ++v) {
            const double b = (*eval)(v);
            const bool ka = a > 0.0 && a >= cut, kb = b > 0.0 && b >= cut;
            double factor = 0.0;
            if (ka && kb)
                factor = -1.0 / (a * b);
            else if (ka != kb)
                factor = ((ka ? 1.0 / a : 0.0) - (kb ? 1.0 / b : 0.0)) / (a - b);
            (*full)(u, v) = -(*full)(u, v) / (a * b) + factor * (*filtered)(u, v);
        }
    }
    auto density = linalg::triplet(U, full, U, false, false, true);
    density->hermitivitize();
    return density;
}

void DLPNOCCSD_T::accumulate_three_center(int r, const TriplesDFBlocks& bar, std::vector<ThreeCenterDensity>& density,
                                          const SharedMatrix& x_bar) {
    const auto [i, j, k] = ijk_to_i_j_k_[r];
    const std::array<int, 3> occ{i, j, k};
    const auto& paos = lmotriplet_to_paos_[r];
    const auto& lmos = lmotriplet_to_lmos_[r];
    const auto& aux = lmotriplet_to_ribfs_[r];
    const int n = n_tno_[r];
    std::array<SharedMatrix, 3> ov;
    for (int p = 0; p < 3; ++p) ov[p] = linalg::doublet(bar.ov[p], X_tno_[r], false, true);

    for (int q = 0; q < aux.size(); ++q) {
        const int atom = ribasis_->function_to_center(aux[q]);
        const auto& li = riatom_to_lmos_ext_dense_[atom];
        const auto& vi = riatom_to_paos_ext_dense_[atom];
        if (riatom_to_lmos_ext_[atom].empty() && riatom_to_paos_ext_[atom].empty()) continue;
        auto& d = density[aux[q]];
        if (!d.oo) {
            const int no = riatom_to_lmos_ext_[atom].size(), nv = riatom_to_paos_ext_[atom].size();
            d.oo = std::make_shared<Matrix>(no, no);
            d.ov = std::make_shared<Matrix>(no, nv);
            d.vv = std::make_shared<Matrix>(nv, nv);
        }
        for (int p = 0; p < 3; ++p) {
            const int mu = li[occ[p]];
            if (mu < 0) continue;
            for (int l = 0; l < lmos.size(); ++l) {
                const int m = li[lmos[l]];
                if (m >= 0) (*d.oo)(mu, m) += (*bar.oo[p])(q, l);
            }
            for (int u = 0; u < paos.size(); ++u) {
                const int a = vi[paos[u]];
                if (a >= 0) {
                    (*d.ov)(mu, a) += (*ov[p])(q, u);
                    C_DAXPY(n, (*qia_[aux[q]])(mu, a), &(*bar.ov[p])(q, 0), 1, &(*x_bar)(u, 0), 1);
                }
            }
        }

        auto vv_tno = std::make_shared<Matrix>(n, n);
        std::copy_n(&(*bar.vv)(q, 0), static_cast<size_t>(n) * n, vv_tno->get_pointer());
        auto vv = linalg::triplet(X_tno_[r], vv_tno, X_tno_[r], false, false, true);
        vv_tno->add(vv_tno->transpose());
        x_bar->add(linalg::triplet(triplet_pao_vv(r, aux[q]), X_tno_[r], vv_tno));
        for (int u = 0; u < paos.size(); ++u) {
            for (int v = 0; v < paos.size(); ++v) {
                const int pair = riatom_to_pao_pairs_dense_[atom][paos[u]][paos[v]];
                if (pair < 0) continue;
                // Both uv and vu read the same stored upper-triangle element.
                const auto [a, b] = riatom_to_pao_pairs_[atom][pair];
                (*d.vv)(vi[a], vi[b]) += (*vv)(u, v);
            }
        }
    }
}

SharedMatrix DLPNOCCSD_T::contract_three_center(const std::vector<ThreeCenterDensity>& density,
                                                const std::string& name) {
    auto grad = std::make_shared<Matrix>(name, molecule_->natom(), 3);
    auto zero = BasisSet::zero_ao_basis_set();
    IntegralFactory factory(ribasis_, zero, basisset_, basisset_);
    auto values = factory.eri();
    auto deriv = factory.eri(1);
    const double tol = options_.get_double("DLPNO_AO_INTS_TOL");

    for (int Q = 0; Q < ribasis_->nshell(); ++Q) {
        const int q0 = ribasis_->shell(Q).function_index(), nq = ribasis_->shell(Q).nfunction();
        if (!density[q0].oo) continue;  // auxiliary domains contain whole atoms
        const int atomQ = ribasis_->shell_to_center(Q);
        const auto& bfs1 = riatom_to_bfs1_[atomQ];
        const auto& bfs2 = riatom_to_bfs2_[atomQ];
        const auto& lmo = gradient_lmo_slices_[atomQ];
        auto pao = submatrix_rows_and_cols(*C_pao_, bfs2, riatom_to_paos_ext_[atomQ]);
        auto bfs = merge_lists(bfs1, bfs2);
        const auto i1 = index_list(bfs, bfs1), i2 = index_list(bfs, bfs2);
        std::vector<SharedMatrix> blocks(nq), raw(nq);
        for (int q = 0; q < nq; ++q) {
            const auto& d = density[q0 + q];
            auto oo = linalg::triplet(lmo, d.oo, lmo, false, false, true);
            auto ov = linalg::triplet(lmo, d.ov, pao, false, false, true);
            auto vv = linalg::triplet(pao, d.vv, pao, false, false, true);
            auto D = std::make_shared<Matrix>(bfs.size(), bfs.size());
            auto scatter = [&](const SharedMatrix& x, const std::vector<int>& rows, const std::vector<int>& cols) {
                for (int u = 0; u < rows.size(); ++u)
                    for (int v = 0; v < cols.size(); ++v) (*D)(rows[u], cols[v]) += (*x)(u, v);
            };
            scatter(oo, i1, i1);
            scatter(ov, i1, i2);
            scatter(vv, i2, i2);
            blocks[q] = std::move(D);
            raw[q] = std::make_shared<Matrix>(bfs.size(), bfs.size());
        }

        // No global Q-by-AO-pair density. Each shell triplet is differentiated
        // once, with precisely the same Schwarz screen as the three forward transforms.
        auto shells = merge_lists(riatom_to_shells1_[atomQ], riatom_to_shells2_[atomQ]);
        for (int M : shells) {
            const int atomM = basisset_->shell_to_center(M), nm = basisset_->shell(M).nfunction();
            const int m0 = std::lower_bound(bfs.begin(), bfs.end(), basisset_->shell(M).function_index()) - bfs.begin();
            for (int N : shells) {
                if (N > M) break;
                const int atomN = basisset_->shell_to_center(N), nn = basisset_->shell(N).nfunction();
                if (J_metric_shell_diag_[Q] * values->shell_pair_value(M, N) < tol * tol) continue;
                const int n0 =
                    std::lower_bound(bfs.begin(), bfs.end(), basisset_->shell(N).function_index()) - bfs.begin();
                // Coefficient motion contributes even if all shells have one center.
                values->compute_shell(Q, 0, M, N);
                const auto* ints = values->buffer();
                for (int q = 0, index = 0; q < nq; ++q)
                    for (int m = 0; m < nm; ++m)
                        for (int n = 0; n < nn; ++n, ++index) {
                            (*raw[q])(m0 + m, n0 + n) = ints[index];
                            if (M != N) (*raw[q])(n0 + n, m0 + m) = ints[index];
                        }
                if (atomQ == atomM && atomM == atomN) continue;
                deriv->compute_shell_deriv1(Q, 0, M, N);
                const auto& buf = deriv->buffers();
                for (int q = 0, index = 0; q < nq; ++q)
                    for (int m = 0; m < nm; ++m)
                        for (int n = 0; n < nn; ++n, ++index) {
                            double d = (*blocks[q])(m0 + m, n0 + n);
                            if (N != M) d += (*blocks[q])(n0 + n, m0 + m);
                            for (int xyz = 0; xyz < 3; ++xyz) {
                                (*grad)(atomQ, xyz) += d * buf[xyz][index];
                                (*grad)(atomM, xyz) += d * buf[3 + xyz][index];
                                (*grad)(atomN, xyz) += d * buf[6 + xyz][index];
                            }
                        }
            }
        }
        auto refit_bar = std::make_shared<Matrix>(lmo->nrow(), lmo->ncol());
        auto pao_bar = std::make_shared<Matrix>(pao->nrow(), pao->ncol());
        SharedMatrix dl, dp;
        if (debug_) {
            auto probe = [](const SharedMatrix& x, int salt) {
                auto d = x->clone();
                for (int a = 0; a < d->nrow(); ++a)
                    for (int b = 0; b < d->ncol(); ++b)
                        (*d)(a, b) = std::sin(.51 * (1 + a) + .37 * (1 + b) + salt) /
                                     std::sqrt(1. + double(d->nrow()) * d->ncol());
                return d;
            };
            dl = probe(lmo, 1 + atomQ);
            dp = probe(pao, 3 + atomQ);
        }
        double numeric = 0.;
        for (int q = 0; q < nq; ++q) {
            const auto& d = density[q0 + q];
            auto oo = submatrix_rows_and_cols(*raw[q], i1, i1), ov = submatrix_rows_and_cols(*raw[q], i1, i2),
                 vv = submatrix_rows_and_cols(*raw[q], i2, i2);
            auto zoo = d.oo->clone();
            zoo->add(d.oo->transpose());
            auto zvv = d.vv->clone();
            zvv->add(d.vv->transpose());
            refit_bar->add(linalg::triplet(oo, lmo, zoo));
            refit_bar->add(linalg::triplet(ov, pao, d.ov, false, false, true));
            pao_bar->add(linalg::triplet(vv, pao, zvv));
            pao_bar->add(linalg::triplet(ov, lmo, d.ov, true, false, false));
            if (debug_) {
                auto sample = [&](double h) {
                    auto c = lmo->clone(), p = pao->clone();
                    c->axpy(h, dl);
                    p->axpy(h, dp);
                    return d.oo->vector_dot(linalg::triplet(c, oo, c, true, false, false)) +
                           d.ov->vector_dot(linalg::triplet(c, ov, p, true, false, false)) +
                           d.vv->vector_dot(linalg::triplet(p, vv, p, true, false, false));
                };
                const int Qfun = q0 + q;
                double forward = d.ov->vector_dot(qia_[Qfun]);
                // Crude excluded-pair response has only an OV density; OO/VV
                // forward arrays do not exist yet at that stage.
                if (d.oo->vector_dot(d.oo) != 0.) forward += d.oo->vector_dot(qij_[Qfun]);
                const auto& pset = riatom_to_paos_ext_[atomQ];
                if (d.vv->vector_dot(d.vv) != 0.) for (int u = 0; u < pset.size(); ++u)
                    for (int v = 0; v < pset.size(); ++v) {
                        int uv = riatom_to_pao_pairs_dense_[atomQ][pset[u]][pset[v]];
                        if (uv >= 0) forward += (*d.vv)(u, v) * (*qab_[Qfun])(uv, 0);
                    }
                const double replay = std::abs(sample(0.) - forward);
                if (!std::isfinite(replay)) throw PSIEXCEPTION("Nonfinite three-center transform replay.");
                df_replay_error_ = std::max(df_replay_error_, replay);
                constexpr double h = 1.e-5;
                numeric += (sample(h) - sample(-h)) / (2 * h);
            }
        }
        if (debug_) {
            double err = std::abs(numeric - refit_bar->vector_dot(dl) - pao_bar->vector_dot(dp));
            if (!std::isfinite(err)) throw PSIEXCEPTION("Nonfinite DF orbital coefficient probe.");
            df_coefficient_error_ = std::max(df_coefficient_error_, err);
        }
        if (!fitted_coefficient_bar_[atomQ])
            fitted_coefficient_bar_[atomQ] = refit_bar;
        else
            fitted_coefficient_bar_[atomQ]->add(refit_bar);
        add_local_block(df_pao_coefficient_bar_, bfs2, riatom_to_paos_ext_[atomQ], pao_bar);
    }
    return grad;
}

}  // namespace dlpno
}  // namespace psi
