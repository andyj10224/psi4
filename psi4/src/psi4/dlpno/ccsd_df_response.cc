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
#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include <algorithm>
#include <cmath>

namespace psi {
namespace dlpno {
namespace {
// A local three-index tensor with occupied labels or PNO columns on each leg.
struct PairDFSpace {
    const std::vector<int> labels;
    SharedMatrix coefficients;
    int pair = -1;
    int size() const { return coefficients ? coefficients->ncol() : labels.size(); }
};
struct PairDFBlock {
    PairDFSpace left, right;
    SharedMatrix raw, bar;
};
}  // namespace

SharedMatrix DLPNOCCSD_T::pair_df_response(int p, std::vector<ThreeCenterDensity>& density) {
    const auto [i, j] = ij_to_i_j_[p];
    const auto& aux = lmopair_to_ribfs_[p];
    const auto& dom = lmopair_to_lmos_[p];
    const auto& response = ccsd_pair_response_[p];
    const int nq = aux.size(), n = n_pno_[p];
    auto J = submatrix_rows_and_cols(*full_metric_, aux, aux);
    auto Jbar = std::make_shared<Matrix>(nq, nq);
    if (!nq || !n) return Jbar;

    auto U = J->clone();
    auto eig = std::make_shared<Vector>(nq);
    J->diagonalize(U, eig);
    const double cut = 1.e-14 * std::max(std::abs((*eig)(0)), std::abs((*eig)(nq - 1)));
    std::vector<double> h(nq);
    auto W = U->clone();
    for (int q = 0; q < nq; ++q) {
        double e = (*eig)(q);
        h[q] = e > 0. && e >= cut ? 1. / std::sqrt(e) : 0.;
        W->scale_column(0, q, h[q]);
    }
    auto half = linalg::doublet(W, U, false, true);
    auto half_bar = Jbar->clone();
    // DGESV contractions (mixed J/K and MP2-origin exchange) use the unfiltered inverse.
    auto inv = J->clone();
    inv->identity();
    C_DGESV_wrapper(J->clone(), inv);
    auto pno = [&](int pair) { return PairDFSpace{lmopair_to_paos_[pair], X_pno_[pair], pair}; };
    PairDFSpace oi{{i}, {}}, oj{{j}, {}}, occ{dom, {}}, vir = pno(p);
    auto untransformed = [&](const PairDFSpace& l, const PairDFSpace& r, int q) {
        int atom = ribasis_->function_to_center(q);
        const auto& li = riatom_to_lmos_ext_dense_[atom];
        const auto& vi = riatom_to_paos_ext_dense_[atom];
        auto v = std::make_shared<Matrix>(l.labels.size(), r.labels.size());
        for (int u = 0; u < l.labels.size(); ++u)
            for (int t = 0; t < r.labels.size(); ++t) {
                int x = l.labels[u], y = r.labels[t];
                if (l.coefficients) {
                    int xy = riatom_to_pao_pairs_dense_[atom][x][y];
                    if (xy >= 0) (*v)(u, t) = (*qab_[q])(xy, 0);
                } else if (li[x] >= 0 && r.coefficients && vi[y] >= 0) {
                    (*v)(u, t) = (*qia_[q])(li[x], vi[y]);
                } else if (!r.coefficients && li[x] >= 0 && li[y] >= 0) {
                    (*v)(u, t) = (*qij_[q])(li[x], li[y]);
                }
            }
        return v;
    };
    auto raw_block = [&](const PairDFSpace& l, const PairDFSpace& r) {
        auto raw = std::make_shared<Matrix>(nq, l.size() * r.size());
        for (int a = 0; a < nq; ++a) {
            auto v = untransformed(l, r, aux[a]);
            if (l.coefficients) v = linalg::doublet(l.coefficients, v, true, false);
            if (r.coefficients) v = linalg::doublet(v, r.coefficients);
            std::copy_n(v->get_pointer(), raw->ncol(), &(*raw)(a, 0));
        }
        return PairDFBlock{l, r, raw, std::make_shared<Matrix>(nq, raw->ncol())};
    };
    auto scatter = [&](const PairDFBlock& b) {
        const auto& l = b.left;
        const auto& r = b.right;
        for (int a = 0; a < nq; ++a) {
            int q = aux[a], atom = ribasis_->function_to_center(q);
            const auto& li = riatom_to_lmos_ext_dense_[atom];
            const auto& vi = riatom_to_paos_ext_dense_[atom];
            auto& d = density[q];
            if (!d.oo) {
                int no = riatom_to_lmos_ext_[atom].size(), nv = riatom_to_paos_ext_[atom].size();
                d.oo = std::make_shared<Matrix>(no, no);
                d.ov = std::make_shared<Matrix>(no, nv);
                d.vv = std::make_shared<Matrix>(nv, nv);
            }
            auto v = std::make_shared<Matrix>(l.size(), r.size());
            std::copy_n(&(*b.bar)(a, 0), b.bar->ncol(), v->get_pointer());
            if (r.coefficients) {
                auto bare = untransformed(l, r, q);
                if (l.coefficients) {
                    ccsd_pno_coefficient_response_[l.pair]->add(
                        linalg::triplet(bare, r.coefficients, v, false, false, true));
                    bare = linalg::doublet(l.coefficients, bare, true, false);
                }
                ccsd_pno_coefficient_response_[r.pair]->add(linalg::doublet(bare, v, true, false));
            }
            if (l.coefficients) v = linalg::doublet(l.coefficients, v);
            if (r.coefficients) v = linalg::doublet(v, r.coefficients, false, true);
            for (int u = 0; u < l.labels.size(); ++u)
                for (int t = 0; t < r.labels.size(); ++t) {
                    int x = l.labels[u], y = r.labels[t];
                    if (l.coefficients) {
                        int xy = riatom_to_pao_pairs_dense_[atom][x][y];
                        if (xy >= 0) {
                            auto [s, b] = riatom_to_pao_pairs_[atom][xy];
                            (*d.vv)(vi[s], vi[b]) += (*v)(u, t);
                        }
                    } else if (li[x] >= 0 && r.coefficients && vi[y] >= 0) {
                        (*d.ov)(li[x], vi[y]) += (*v)(u, t);
                    } else if (!r.coefficients && li[x] >= 0 && li[y] >= 0) {
                        (*d.oo)(li[x], li[y]) += (*v)(u, t);
                    }
                }
        }
    };
    // For C=A^T J^-1 B use both solve adjoints. For symmetrically
    // whitened products reverse the actual fixed-rank J^-1/2 transform.
    auto whiten_bar = [&](PairDFBlock& b, const SharedMatrix& seed) {
        b.bar->add(linalg::doublet(half, seed));
        half_bar->add(linalg::doublet(seed, b.raw, false, true));
    };
    auto integral = [&](PairDFBlock& a, PairDFBlock& b, const SharedMatrix& z, bool full) {
        if (full) {
            auto fa = linalg::doublet(inv, a.raw), fb = linalg::doublet(inv, b.raw);
            auto da = linalg::doublet(fb, z, false, true);
            a.bar->add(da);
            b.bar->add(linalg::doublet(fa, z));
            Jbar->subtract(linalg::doublet(fa, da, false, true));
        } else {
            whiten_bar(a, linalg::triplet(half, b.raw, z, false, false, true));
            whiten_bar(b, linalg::triplet(half, a.raw, z));
        }
    };
    auto iv = raw_block(oi, vir), jv = raw_block(oj, vir), io = raw_block(oi, occ);
    auto ov = raw_block(occ, vir), vv = raw_block(vir, vir), ij = raw_block(oi, oj);
    integral(iv, jv, response.exchange, true);
    integral(io, jv, response.ooov_k, false);
    auto jseed = response.ooov_j->clone();
    jseed->reshape(1, dom.size() * n);
    integral(ij, ov, jseed, false);
    integral(iv, vv, response.ovvv, false);
    whiten_bar(iv, response.qi_vir);
    whiten_bar(io, response.qi_occ);
    if (i <= j) {
        auto packed = [&](const std::vector<SharedMatrix>& factors) {
            auto z = std::make_shared<Matrix>(nq, factors[0]->nrow() * factors[0]->ncol());
            for (int a = 0; a < nq; ++a) std::copy_n(factors[a]->get_pointer(), z->ncol(), &(*z)(a, 0));
            return z;
        };
        whiten_bar(ov, packed(response.qma));
        whiten_bar(vv, packed(response.qab));
    }
    for (int m = 0; m < dom.size(); ++m) {
        int k = dom[m], kj = i_j_to_ij_[k][j];
        auto vk = pno(kj);
        if (!vk.size()) continue;
        PairDFSpace ok{{k}, {}};
        auto ik = raw_block(oi, ok), cross = raw_block(vir, vk), kv = raw_block(ok, vk);
        auto z = response.mixed_j[m]->clone();
        z->reshape(1, n * vk.size());
        integral(ik, cross, z, true);
        integral(iv, kv, response.mixed_k[m], true);
        scatter(ik);
        scatter(cross);
        scatter(kv);
    }
    for (const auto* b : {&iv, &jv, &io, &ov, &vv, &ij}) scatter(*b);

    // Stable divided differences of the inverse square root, including a
    // transition between kept and discarded eigenvectors at fixed metric rank.
    auto z = linalg::triplet(U, half_bar, U, true, false, false);
    for (int a = 0; a < nq; ++a)
        for (int b = 0; b < nq; ++b) {
            double f = 0.;
            if (h[a] && h[b])
                f = -h[a] * h[b] / (std::sqrt((*eig)(a)) + std::sqrt((*eig)(b)));
            else if (h[a] != h[b])
                f = (h[a] - h[b]) / ((*eig)(a) - (*eig)(b));
            (*z)(a, b) *= f;
        }
    Jbar->add(linalg::triplet(U, z, U, false, false, true));
    Jbar->hermitivitize();
    return Jbar;
}

void DLPNOCCSD_T::contract_ccsd_df_response() {
    MetricShellBlocks metric(ribasis_->nshell());
    std::vector<ThreeCenterDensity> density(ribasis_->nbf());
    double scale = 0.;
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        auto bar = pair_df_response(p, density);
        scale += add_metric_density(metric, lmopair_to_ribfs_[p], bar);
    }
    // Only the backtransformed one-body and local-basis covectors are needed downstream.
    ccsd_pair_response_.clear();
    set_scalar_variable("DLPNO CCSD AUXILIARY METRIC SCALE DERIVATIVE", scale);
    set_array_variable("DLPNO CCSD AUXILIARY METRIC GRADIENT",
                       contract_metric(metric, "DLPNO CCSD AUXILIARY METRIC GRADIENT"));
    set_array_variable("DLPNO CCSD THREE-CENTER GRADIENT",
                       contract_three_center(density, "DLPNO CCSD THREE-CENTER GRADIENT"));
    if (debug_) check_ccsd_coefficient_response();
}

// Vary the shared ij/ji coefficients in an arbitrary rectangular direction,
// keeping the PAOs, Fock matrices, amplitudes, and all domain ranks fixed.
void DLPNOCCSD_T::check_ccsd_coefficient_response() {
    const auto original = X_pno_, original_k = K_iajb_;
    std::vector<SharedMatrix> d(original.size());
    double analytic = 0.;
    for (int p = 0; p < original.size(); ++p) {
        auto [i, j] = ij_to_i_j_[p];
        if (i > j) continue;
        auto v = original[p]->clone();
        for (int u = 0; u < v->nrow(); ++u)
            for (int a = 0; a < v->ncol(); ++a)
                (*v)(u, a) =
                    std::sin(.37 * (p + 1) + .29 * (u + 1) + .51 * (a + 1)) / std::sqrt(1. + v->nrow() * v->ncol());
        d[p] = d[ij_to_ji_[p]] = v;
    }
    for (int p = 0; p < original.size(); ++p) analytic += d[p]->vector_dot(ccsd_pno_coefficient_response_[p]);
    auto rebuild = [&]() {
        compute_pno_overlaps();
        compute_pno_integrals(true);
    };
    auto sample = [&](double step) {
        for (int p = 0; p < original.size(); ++p) {
            auto [i, j] = ij_to_i_j_[p];
            if (i > j) continue;
            auto v = original[p]->clone();
            v->axpy(step, d[p]);
            X_pno_[p] = X_pno_[ij_to_ji_[p]] = v;
            // The exchange integral originates in the MP2 setup and is subsequently
            // transformed twice. With both X matrices composed this is a single projection.
            const auto& aux = lmopair_to_ribfs_[p];
            const auto& pao = lmopair_to_paos_[p];
            auto qi = std::make_shared<Matrix>(aux.size(), pao.size()), qj = qi->clone();
            for (int a = 0; a < aux.size(); ++a) {
                int q = aux[a], atom = ribasis_->function_to_center(q);
                for (int u = 0; u < pao.size(); ++u) {
                    int t = riatom_to_paos_ext_dense_[atom][pao[u]];
                    (*qi)(a, u) = (*qia_[q])(riatom_to_lmos_ext_dense_[atom][i], t);
                    (*qj)(a, u) = (*qia_[q])(riatom_to_lmos_ext_dense_[atom][j], t);
                }
            }
            auto J = submatrix_rows_and_cols(*full_metric_, aux, aux);
            C_DGESV_wrapper(J, qi);
            K_iajb_[p] = linalg::triplet(v, linalg::doublet(qi, qj, true, false), v, true, false, false);
            if (i != j) K_iajb_[ij_to_ji_[p]] = K_iajb_[p]->transpose();
        }
        rebuild();
        return ccsd_lagrangian(lambda_ia_, lambda_iajb_);
    };
    double numeric;
    try {
        constexpr double h = 1.e-5;
        double plus = sample(h);
        numeric = (plus - sample(-h)) / (2 * h);
    } catch (...) {
        X_pno_ = original;
        K_iajb_ = original_k;
        throw;
    }
    X_pno_ = original;
    K_iajb_ = original_k;
    rebuild();
    prepare_ccsd_residual();
    outfile->Printf("    CCSD PNO coefficient direction: analytic % .12e, numerical % .12e, error %.3e\n", analytic,
                    numeric, std::abs(analytic - numeric));
    set_scalar_variable("DLPNO CCSD PNO COEFFICIENT DIRECTION ERROR", std::abs(analytic - numeric));
}

}  // namespace dlpno
}  // namespace psi
