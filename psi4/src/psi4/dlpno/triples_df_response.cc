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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace psi {
namespace dlpno {
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
            if (x_bar) {
                auto ds = linalg::triplet(T_iajb_[ij], p.overlap, p.seed, false, false, true);
                ds->add(linalg::triplet(T_iajb_[ij], p.overlap, p.seed, true, false, false));
                x_bar->add(linalg::doublet(p.map, ds, true, false));
            }
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
        if (x_bar) x_bar->add(linalg::doublet(fp, df, true, false));
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
        if (x_bar) x_bar->add(linalg::triplet(ii.map, T_ia_[occ[p]], ds, true, false, true));
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
                    if (x_bar) C_DAXPY(n, (*qia_[aux[q]])(mu, a), &(*bar.ov[p])(q, 0), 1, &(*x_bar)(u, 0), 1);
                }
            }
        }

        auto vv_tno = std::make_shared<Matrix>(n, n);
        std::copy_n(&(*bar.vv)(q, 0), static_cast<size_t>(n) * n, vv_tno->get_pointer());
        auto vv = linalg::triplet(X_tno_[r], vv_tno, X_tno_[r], false, false, true);
        if (x_bar) {
            vv_tno->add(vv_tno->transpose());
            x_bar->add(linalg::triplet(triplet_pao_vv(r, aux[q]), X_tno_[r], vv_tno));
        }
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

SharedMatrix DLPNOCCSD_T::contract_three_center(const std::vector<ThreeCenterDensity>& density) {
    auto grad = std::make_shared<Matrix>("DLPNO (T) THREE-CENTER GRADIENT", molecule_->natom(), 3);
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
        std::vector<SharedMatrix> blocks(nq);
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
                if (atomQ == atomM && atomM == atomN) continue;
                if (J_metric_shell_diag_[Q] * values->shell_pair_value(M, N) < tol * tol) continue;
                const int n0 =
                    std::lower_bound(bfs.begin(), bfs.end(), basisset_->shell(N).function_index()) - bfs.begin();
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
    }
    return grad;
}

}  // namespace dlpno
}  // namespace psi
