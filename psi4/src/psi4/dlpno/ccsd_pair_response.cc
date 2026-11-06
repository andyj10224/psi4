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

#include "ccsd_reverse.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include <cmath>

namespace psi {
namespace dlpno {
using ccsd_detail::add_row;
using ccsd_detail::row_as_column;
using ccsd_detail::zero_blocks;

namespace {
// Differentiate seed : A B ... with the Frobenius convention. Release partial
// products after each local term. Transpose flags refer to the forward factor.
struct Factor {
    SharedMatrix value, bar;
    bool trans = false;
};
void product_bar(const SharedMatrix& seed, double weight, std::initializer_list<Factor> factors) {
    std::vector<Factor> f(factors);
    std::vector<SharedMatrix> left(f.size());
    left[0] = f[0].trans ? f[0].value->transpose() : f[0].value;
    for (size_t i = 1; i + 1 < f.size(); ++i) left[i] = linalg::doublet(left[i - 1], f[i].value, false, f[i].trans);
    auto d = seed;
    for (int i = static_cast<int>(f.size()) - 1; i >= 0; --i) {
        if (f[i].bar) {
            auto v = i ? linalg::doublet(left[i - 1], d, true, false) : d;
            f[i].bar->axpy(weight, f[i].trans ? v->transpose() : v);
        }
        if (i) d = linalg::doublet(d, f[i].value, false, !f[i].trans);
    }
}
SharedMatrix outer(const SharedMatrix& a, const SharedMatrix& b) { return linalg::doublet(a, b, false, true); }
SharedMatrix zeros_like(const SharedMatrix& m) { return std::make_shared<Matrix>(m->nrow(), m->ncol()); }
}  // namespace

DLPNOCCSD::DFPairState DLPNOCCSD::ccsd_df_pair_state(int i, int j) const {
    if (!gradient_model_ || i < 0 || j < i || j >= T_ia_.size() || i_j_to_ij_[i][j] < 0)
        throw PSIEXCEPTION("Expected a unique retained RHF CCSD pair.");
    const int p = i_j_to_ij_[i][j], rev = ij_to_ji_[p];
    const auto& dom = lmopair_to_lmos_[p];
    std::vector<SharedMatrix> singles;
    for (int m : dom) singles.push_back(T_ia_[m]->clone());
    auto aux = std::make_shared<Matrix>(lmopair_to_ribfs_[p].size(),1);
    for (int q = 0; q < aux->nrow(); ++q) (*aux)(q,0) = lmopair_to_ribfs_[p][q];
    return {dom, singles, {{"AUX",aux}, {"TN", T_n_ij_.at(p)->clone()}, {"T2", T_iajb_.at(p)->clone()},
                           {"OI", i_Qk_t1_.at(p)->clone()}, {"OJ", i_Qk_t1_.at(rev)->clone()},
                           {"VI", i_Qa_t1_.at(p)->clone()}, {"VJ", i_Qa_t1_.at(rev)->clone()}}};
}

// Partial derivative with respect to the projected bare Fock blocks. The CCSD
// amplitudes, interpair overlaps and two-electron inputs are independent here.
std::pair<std::vector<SharedMatrix>, DLPNOCCSD::PairBlocks> DLPNOCCSD::ccsd_fock_source(
    SharedMatrix oo, const PairBlocks& ov, const PairBlocks& vv) {
    const int no = T_ia_.size();
    if (!gradient_model_ || !oo || oo->nirrep() != 1 || oo->nrow() != no || oo->ncol() != no)
        throw PSIEXCEPTION("Invalid occupied Fock tangent for the RHF CCSD gradient model.");
    auto r1 = zero_blocks(T_ia_);
    PairBlocks r2;
    auto kj = oo->clone();
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        auto [i, j] = ij_to_i_j_[p];
        if (i > j) continue;
        const auto ij = std::make_pair(i, j);
        auto valid = [&](const PairBlocks& blocks, int rows) {
            auto it = blocks.find(ij);
            return it != blocks.end() && it->second && it->second->nirrep() == 1 &&
                   it->second->nrow() == rows && it->second->ncol() == n_pno_[p];
        };
        if (!valid(ov, lmopair_to_lmos_[p].size()) || !valid(vv, n_pno_[p]))
            throw PSIEXCEPTION("Invalid projected Fock tangent for a retained CCSD pair.");
        const auto& h = ov.at(ij);
        const auto& tn = T_n_ij_.at(p);
        auto fab = vv.at(ij)->clone();
        fab->subtract(linalg::doublet(tn, h, true, false));
        if (i == j) {
            auto ti = T_ia_[i];
            r1[i]->add(row_as_column(h, lmopair_to_lmos_dense_[p][i]));
            r1[i]->add(linalg::doublet(fab, ti));
            auto column = submatrix_rows_and_cols(*oo, lmopair_to_lmos_[p], {i});
            r1[i]->subtract(linalg::doublet(tn, column, true, false));
        }
        auto e = linalg::doublet(T_iajb_[p], fab, false, true);
        e->add(linalg::doublet(fab, T_iajb_[p]));
        r2[ij] = e;
    }
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        auto [i, j] = ij_to_i_j_[p];
        auto ij = std::make_pair(std::min(i,j), std::max(i,j));
        auto ii = i_j_to_ij_[i][i];
        auto h = row_as_column(ov.at(ij), lmopair_to_lmos_dense_[p][j]);
        r1[i]->add(linalg::triplet(S_PNO(p,ii), Tt_iajb_[p], h, true, false, false));
        (*kj)(i,j) += row_as_column(ov.at({j,j}), lmopair_to_lmos_dense_[i_j_to_ij_[j][j]][i])->vector_dot(T_ia_[j]);
    }
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        auto [i,j] = ij_to_i_j_[p];
        auto ij = std::make_pair(std::min(i,j),std::max(i,j));
        for (int k : lmopair_to_lmos_[i_j_to_ij_[i][i]]) {
            auto ik = i_j_to_ij_[i][k];
            auto s = S_PNO(p,ik);
            auto term = linalg::triplet(s,T_iajb_[ik],s,false,false,true);
            if (i == j) term->add(term->transpose());
            r2.at(ij)->axpy(-(*kj)(k,j), i > j ? term->transpose() : term);
        }
    }
    return {r1,r2};
}

void DLPNOCCSD::form_ccsd_pair_response() {
    CCAdjoints g(*this, lambda_ia_, lambda_iajb_);
    reverse_R1(g);
    reverse_R2(g);
    reverse_pair_intermediates(g);
    reverse_t1_transform(g);
    const int pairs = ij_to_i_j_.size();
    ccsd_pair_response_.assign(pairs, {});
    ccsd_foo_response_.assign(T_ia_.size(), {});
    auto& r = ccsd_pair_response_;
    auto dL = zero_blocks(L_iajb_);
    for (int p = 0; p < pairs; ++p) {
        auto& b = r[p];
        b.fov = zeros_like(T_n_ij_[p]);
        b.fvv = zeros_like(T_iajb_[p]);
        b.exchange = zeros_like(K_iajb_[p]);
        b.ooov_k = zeros_like(K_mibj_[p]);
        b.ooov_j = zeros_like(J_ijmb_[p]);
        b.ovvv = zeros_like(K_ivvv_[p]);
        b.qi_occ = g.qk[p]->clone();  // already includes the indirect path through dressed B_ia
        b.qi_vir = g.qa[p]->clone();
        b.mixed_j = zero_blocks(J_ikac_non_proj_[p]);
        b.mixed_k = zero_blocks(K_iakc_non_proj_[p]);
        const auto [i, j] = ij_to_i_j_[p];
        if (i <= j) {
            for (size_t q = 0; q < lmopair_to_ribfs_[p].size(); ++q) {
                b.qma.push_back(zeros_like(T_n_ij_[p]));
                b.qab.push_back(zeros_like(T_iajb_[p]));
            }
        }
    }
    auto overlap = [&](int p, int b, bool trans = false) -> Factor {
        auto s = S_PNO(p, b);
        auto& bar = r[p].overlap[b];
        if (!bar) {
            ccsd_response_words_ += size_t(s->nrow()) * s->ncol();
            if (toggle_memory_ && ccsd_response_words_ * sizeof(double) > 0.9 * memory_)
                throw PSIEXCEPTION("Insufficient memory for the active CCSD overlap adjoints.");
            bar = zeros_like(s);
        }
        return {s, bar, trans};
    };
    for (int p = 0; p < pairs; ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        const int rev = ij_to_ji_[p], ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];
        const int n = n_pno_[p];
        const auto& dom = lmopair_to_lmos_[p];
        auto& b = r[p];
        auto& fit = r[i <= j ? p : rev];
        const auto& T = T_iajb_[p];
        const auto& U = Tt_iajb_[p];
        const auto& tn = T_n_ij_[p];
        int mi = lmopair_to_lmos_dense_[p][i];
        // Energy and projected singles. T_n includes both singles in this pair.
        auto ti = row_as_column(tn, mi), tj = row_as_column(tn, lmopair_to_lmos_dense_[p][j]);
        dL[p]->add(T);
        dL[p]->add(outer(ti, tj));
        auto si = overlap(p, ii), sj = overlap(p, jj);
        si.bar->add(outer(linalg::doublet(L_iajb_[p], tj), T_ia_[i]));
        sj.bar->add(outer(linalg::doublet(L_iajb_[p], ti, true, false), T_ia_[j]));
        for (int m = 0; m < dom.size(); ++m)
            overlap(p, i_j_to_ij_[dom[m]][dom[m]]).bar->add(outer(row_as_column(g.tn[p], m), T_ia_[dom[m]]));

        // Bare Fock. Each pair contributes in its own PNO basis.
        ccsd_foo_response_[i][j] += g.fij_bar[p];
        add_row(b.fov, mi, g.fkc[p]);
        if (i <= j) {
            b.fvv->add(g.fab[p]);
            b.fov->add(g.fkc_bar[p]);
        }
        if (i == j) {
            add_row(b.fov, mi, T_ia_[i], 2.);
            add_row(b.fov, mi, g.z1[i]);
        }
        // R1 A,C and the flush of the B intermediate.
        auto uflat = Tt_iajb_[rev]->clone();
        uflat->reshape(n * n, 1);
        auto kv = K_ivvv_[rev]->clone();
        kv->reshape(n * n, n);
        product_bar(g.z1[i], 1., {overlap(rev, ii, true), {kv, {}, true}, {uflat, {}}});
        auto dk = outer(uflat, linalg::doublet(S_PNO(rev, ii), g.z1[i]));
        dk->reshape(n, n * n);
        r[rev].ovvv->add(dk);
        product_bar(g.z1[i], 1., {overlap(p, ii, true), {U, {}}, {Fkc_[rev], {}}});
        for (int l : dom) {
            int kl = i_j_to_ij_[j][l], ll = i_j_to_ij_[l][l];
            auto proj = linalg::triplet(S_PNO(kl, rev), Tt_iajb_[rev], S_PNO(rev, kl));
            double coeff = -g.z1[i]->vector_dot(linalg::doublet(S_PNO(ii, ll), T_ia_[l]));
            r[kl].exchange->axpy(coeff, proj);
            product_bar(K_iajb_[kl], coeff, {overlap(kl, rev), {Tt_iajb_[rev], {}}, overlap(rev, kl)});
            overlap(ii, ll).bar->axpy(-proj->vector_dot(K_iajb_[kl]), outer(g.z1[i], T_ia_[l]));
        }
        auto M = K_mibj_[p]->clone();
        M->add(linalg::doublet(tn, K_iajb_[p]));
        auto B = linalg::doublet(M, U, false, true);
        auto rows = zeros_like(tn);
        for (int m = 0; m < dom.size(); ++m) {
            auto s = overlap(p, i_j_to_ij_[dom[m]][dom[m]]);
            s.bar->subtract(outer(row_as_column(B, m), g.z1[dom[m]]));
            add_row(rows, m, linalg::doublet(s.value, g.z1[dom[m]]), -1.);
        }
        auto dm = linalg::doublet(rows, U);
        b.ooov_k->add(dm);
        b.exchange->add(linalg::doublet(tn, dm, true, false));

        auto y = g.z2[p]->clone();
        y->add(g.z2[rev]->transpose());
        y->scale(0.5);
        const auto qma = QIA_PNO(p), qab = QAB_PNO(p);
        if (i <= j) {
            auto z = y->clone();
            if (i == j) {
                z = g.z2[p]->clone();
                z->scale(0.5);
            }
            // A term and the B/E projections.
            for (int q = 0; q < qma.size(); ++q) {
                auto Q = qab[q]->clone();
                Q->subtract(linalg::doublet(tn, qma[q], true, false));
                auto dq = linalg::triplet(z, Q, T, false, false, true);
                dq->add(linalg::triplet(z, Q, T, true, false, false));
                fit.qab[q]->add(dq);
                fit.qma[q]->subtract(linalg::doublet(tn, dq));
            }
            auto df = linalg::doublet(z, T, true, false);
            df->add(linalg::doublet(z, T, false, true));
            for (int k = 0; k < dom.size(); ++k)
                for (int l = 0; l < dom.size(); ++l) {
                    int kl = i_j_to_ij_[dom[k]][dom[l]];
                    if (kl < 0 || !n_pno_[kl]) continue;
                    product_bar(z, (*beta_[p])(k, l), {overlap(kl, p, true), {T_iajb_[kl], {}}, overlap(kl, p)});
                    product_bar(df, -1.,
                                {overlap(kl, p, true),
                                 {Tt_iajb_[kl], {}},
                                 {K_iajb_[kl], r[kl].exchange, true},
                                 overlap(kl, p)});
                }
        }
        // Rectangular C and D integrals and overlaps.
        auto yc = y->transpose();
        yc->axpy(.5, y);
        for (int m = 0; m < dom.size(); ++m) {
            int k = dom[m], ki = i_j_to_ij_[k][i], kj = i_j_to_ij_[k][j];
            int ik = ij_to_ji_[ki], jk = ij_to_ji_[kj];
            auto mixed = [&](int a, int c, const SharedMatrix& dressed, const SharedMatrix& bare,
                             const SharedMatrix& amp, const SharedMatrix& z, double weight) {
                product_bar(z, weight, {overlap(p, a), {dressed, {}}, overlap(a, c), {amp, {}, true}, overlap(c, p)});
                product_bar(z, weight, {{bare, {}}, {amp, {}, true}, overlap(c, p)});
                return linalg::triplet(z, S_PNO(c, p), amp, false, true, false);
            };
            auto dj = mixed(ki, kj, gamma_[ki], J_ikac_non_proj_[p][m], T_iajb_[kj], yc, -1.);
            b.mixed_j[m]->subtract(dj);
            auto bare = K_iakc_non_proj_[p][m]->clone();
            bare->scale(2.);
            bare->subtract(J_ikac_non_proj_[p][m]);
            auto dd = mixed(ik, jk, delta_[ik], bare, Tt_iajb_[jk], y, .5);
            b.mixed_k[m]->add(dd);
            b.mixed_j[m]->axpy(-.5, dd);
        }
        for (int k : lmopair_to_lmos_[ii]) {
            int ik = i_j_to_ij_[i][k];
            product_bar(y, -(*F_im_double_tilde_)(k, j), {overlap(p, ik), {T_iajb_[ik], {}}, overlap(ik, p)});
            if (i_j_to_ij_[k][j] < 0) {
                auto t = linalg::triplet(S_PNO(p, ik), T_iajb_[ik], S_PNO(ik, p));
                ccsd_foo_response_[k][j] -= y->vector_dot(t);
            }
        }
        // F'' occupied and gamma/delta.
        for (int l : dom) {
            int li = i_j_to_ij_[l][i], lj = i_j_to_ij_[l][j];
            auto v = linalg::triplet(S_PNO(li, lj), Tt_iajb_[lj], S_PNO(lj, li));
            r[li].exchange->axpy(g.fkj[p], v);
            product_bar(K_iajb_[li], g.fkj[p], {overlap(li, lj), {Tt_iajb_[lj], {}}, overlap(lj, li)});
        }
        for (bool gamma : {true, false}) {
            int o = gamma ? j : i, k = gamma ? i : j, oo = i_j_to_ij_[o][o];
            const auto& z = gamma ? g.gamma[p] : g.delta[p];
            auto dz = linalg::doublet(tn, z);
            b.ooov_j->axpy(gamma ? -1. : 1., dz);
            if (!gamma) b.ooov_k->axpy(-2., dz);
            auto s = overlap(p, oo);
            auto tv = linalg::doublet(s.value, T_ia_[o]);
            auto flat = z->clone();
            flat->reshape(n * n, 1);
            int ip = gamma ? p : rev;
            double sign = gamma ? 1. : -1.;
            r[ip].ovvv->axpy(sign, outer(tv, flat));
            auto dv = linalg::doublet(K_ivvv_[ip], flat);
            dv->scale(sign);
            if (!gamma) {
                auto zt = z->transpose();
                zt->reshape(n * n, 1);
                auto d = outer(zt, tv);
                d->reshape(n, n * n);
                r[ip].ovvv->axpy(2., d);
                auto v = K_ivvv_[ip]->clone();
                v->reshape(n * n, n);
                dv->axpy(2., linalg::doublet(v, zt, true, false));
            }
            s.bar->add(outer(dv, T_ia_[o]));
            for (int l : dom) {
                int ll = i_j_to_ij_[l][l], other = gamma ? i_j_to_ij_[k][l] : i_j_to_ij_[l][k];
                int amp = gamma ? i_j_to_ij_[l][o] : i_j_to_ij_[o][l];
                Factor I = gamma ? Factor{K_iajb_[other], r[other].exchange} : Factor{L_iajb_[other], dL[other]};
                product_bar(z, -1.,
                            {overlap(p, ll),
                             {T_ia_[l], {}},
                             {T_ia_[o], {}, true},
                             overlap(other, oo, true),
                             I,
                             overlap(other, p)});
                product_bar(z, gamma ? -.5 : .5,
                            {overlap(p, amp),
                             {gamma ? T_iajb_[amp] : Tt_iajb_[amp], {}},
                             overlap(amp, other),
                             I,
                             overlap(other, p)});
            }
        }
        // Bare two-electron terms in dressed Fock and DF intermediates.
        b.ooov_j->axpy(2. * g.fij_bar[p], tn);
        r[rev].ooov_k->axpy(-g.fij_bar[p], tn);
        for (int k : dom) {
            int ik = i_j_to_ij_[i][k], kk = i_j_to_ij_[k][k];
            product_bar(g.fkc[p], 1., {overlap(p, ik), {L_iajb_[ik], dL[ik]}, overlap(ik, kk), {T_ia_[k], {}}});
        }
        for (int q = 0; q < qma.size(); ++q) {
            Factor V{qab[q], fit.qab[q]}, O{qma[q], fit.qma[q]};
            product_bar(g.beta[p], 1., {O, {T, {}}, {qma[q], fit.qma[q], true}});
            fit.qab[q]->add(outer(row_as_column(g.qa[p], q), ti));
            fit.qma[q]->add(outer(row_as_column(g.qk[p], q), ti));
            if (i > j) continue;
            double tr = tn->vector_dot(qma[q]);
            double dtr = g.fab[p]->vector_dot(qab[q]) + g.fkc_bar[p]->vector_dot(qma[q]);
            fit.qab[q]->axpy(2. * tr, g.fab[p]);
            fit.qma[q]->axpy(2. * tr, g.fkc_bar[p]);
            product_bar(g.fab[p], -1., {V, {tn, {}, true}, O});
            product_bar(g.fkc_bar[p], -1., {O, {tn, {}, true}, O});
            if (i == j) {
                const auto& z = g.z1[i];
                auto qa = row_as_column(i_Qa_ij_[p], q), qk = row_as_column(i_Qk_ij_[p], q);
                dtr += z->vector_dot(qa);
                add_row(b.qi_vir, q, z, 2. * tr);
                product_bar(z, -1., {V, {tn, {}, true}, {qk, {}}});
                add_row(b.qi_occ, q, linalg::triplet(tn, qab[q], z, false, true, false), -1.);
            }
            fit.qma[q]->axpy(2. * dtr, tn);
        }
    }
    // L=2K-K^T. All other blocks retain their forward orientation.
    for (int p = 0; p < pairs; ++p) {
        r[p].exchange->axpy(2., dL[p]);
        r[p].exchange->subtract(dL[p]->transpose());
    }
}

void DLPNOCCSD::backtransform_ccsd_one_body() {
    ccsd_fov_response_.assign(F_lmo_pao_->nrow(), {});
    ccsd_fvv_response_.assign(F_pao_->nrow(), {});
    ccsd_overlap_response_.assign(S_pao_->nrow(), {});
    ccsd_pno_coefficient_response_ = zero_blocks(X_pno_);
    auto scatter = [](SparseResponseRows& rows, const std::vector<int>& left, const std::vector<int>& right,
                      const SharedMatrix& b) {
        for (int u = 0; u < left.size(); ++u)
            for (int v = 0; v < right.size(); ++v) rows[left[u]][right[v]] += (*b)(u, v);
    };
    size_t edges = 0, kl_visits = 0;
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        auto& b = ccsd_pair_response_[p];
        const auto& paos = lmopair_to_paos_[p];
        const auto& X = X_pno_[p];
        auto& xbar = ccsd_pno_coefficient_response_[p];
        auto ov = submatrix_rows_and_cols(*F_lmo_pao_, lmopair_to_lmos_[p], paos);
        xbar->add(linalg::doublet(ov, b.fov, true, false));
        scatter(ccsd_fov_response_, lmopair_to_lmos_[p], paos, linalg::doublet(b.fov, X, false, true));
        if (ij_to_i_j_[p].first <= ij_to_i_j_[p].second) {
            const size_t m = lmopair_to_lmos_[p].size();
            kl_visits += m * m;
            auto vv = submatrix_rows_and_cols(*F_pao_, paos, paos);
            auto bar = b.fvv->clone();
            bar->add(b.fvv->transpose());
            xbar->add(linalg::triplet(vv, X, bar));
            scatter(ccsd_fvv_response_, paos, paos, linalg::triplet(X, b.fvv, X, false, false, true));
        }
        edges += b.overlap.size();
        for (const auto& [q, z] : b.overlap) {
            const auto& Y = X_pno_[q];
            const auto& other = lmopair_to_paos_[q];
            auto s = submatrix_rows_and_cols(*S_pao_, paos, other);
            xbar->add(linalg::triplet(s, Y, z, false, false, true));
            ccsd_pno_coefficient_response_[q]->add(linalg::triplet(s, X, z, true, false, false));
            scatter(ccsd_overlap_response_, paos, other, linalg::triplet(X, z, Y, false, false, true));
        }
        b.overlap.clear();
    }
    set_scalar_variable("DLPNO CCSD RESPONSE ORDERED PAIRS", ij_to_i_j_.size());
    set_scalar_variable("DLPNO CCSD RESPONSE OVERLAP BLOCKS", edges);
    set_scalar_variable("DLPNO CCSD RESPONSE KL CANDIDATES", kl_visits);
}

void DLPNOCCSD::check_ccsd_pair_response() {
    // Fixed-geometry partial derivatives: hold amplitudes, multipliers, PNO coefficients
    // and domain lists fixed. Rebuild the original right residual at both endpoints.
    struct Variation {
        SharedMatrix* slot;
        SharedMatrix original, direction;
    };
    auto direction = [](const SharedMatrix& m, int salt, bool symmetric = false) {
        auto d = zeros_like(m);
        for (int a = 0; a < d->nrow(); ++a)
            for (int b = 0; b < d->ncol(); ++b) {
                int x = symmetric ? std::min(a, b) : a, y = symmetric ? std::max(a, b) : b;
                (*d)(a, b) = std::sin(.71 * (1 + x) + .39 * (1 + y) + .17 * salt) /
                             std::sqrt(1. + double(d->nrow()) * d->ncol());
            }
        return d;
    };
    // Only diagnostic perturbations require holding disk-backed factors in core.
    const bool disk_ia = write_qia_pno_, disk_ab = write_qab_pno_;
    auto orig_ia = Qma_ij_, orig_ab = Qab_ij_;
    auto orig_l = L_iajb_;
    auto restore_fit = [&]() {
        Qma_ij_ = orig_ia;
        Qab_ij_ = orig_ab;
        write_qia_pno_ = disk_ia;
        write_qab_pno_ = disk_ab;
        L_iajb_ = orig_l;
    };
    try {
        Qma_ij_.resize(ij_to_i_j_.size());
        Qab_ij_.resize(ij_to_i_j_.size());
        for (int p = 0; p < ij_to_i_j_.size(); ++p) {
            auto [i, j] = ij_to_i_j_[p];
            if (i > j) continue;
            if (disk_ia) Qma_ij_[p] = QIA_PNO(p);
            if (disk_ab) Qab_ij_[p] = QAB_PNO(p);
        }
        write_qia_pno_ = write_qab_pno_ = false;
        for (const std::string group : {"INTEGRAL", "FOCK", "OVERLAP"}) {
            std::vector<Variation> vars;
            double analytic = 0.;
            auto add = [&](SharedMatrix& slot, const SharedMatrix& bar, bool sym = false) {
                auto d = direction(slot, static_cast<int>(vars.size()) + 1, sym);
                if (bar) analytic += d->vector_dot(bar);
                vars.push_back({&slot, slot, d});
                return d;
            };
            std::vector<SharedMatrix> fock1, plus1, minus1, plus2, minus2;
            PairBlocks fock2;
            double dual = 0.;
            if (group == "INTEGRAL") {
                for (int p = 0; p < ij_to_i_j_.size(); ++p) {
                    auto& b = ccsd_pair_response_[p];
                    add(K_iajb_[p], b.exchange);
                    add(K_mibj_[p], b.ooov_k);
                    add(J_ijmb_[p], b.ooov_j);
                    add(K_ivvv_[p], b.ovvv);
                    add(i_Qk_ij_[p], b.qi_occ);
                    add(i_Qa_ij_[p], b.qi_vir);
                    for (int m = 0; m < b.mixed_j.size(); ++m) {
                        add(J_ikac_non_proj_[p][m], b.mixed_j[m]);
                        add(K_iakc_non_proj_[p][m], b.mixed_k[m]);
                    }
                    for (int q = 0; q < b.qma.size(); ++q) {
                        add(Qma_ij_[p][q], b.qma[q]);
                        add(Qab_ij_[p][q], b.qab[q], true);
                    }
                }
            } else if (group == "FOCK") {
                auto oo = add(F_lmo_, {}, true), ov = add(F_lmo_pao_, {}), vv = add(F_pao_, {}, true);
                for (int i = 0; i < ccsd_foo_response_.size(); ++i)
                    for (auto [j, bar] : ccsd_foo_response_[i]) analytic += (*oo)(i, j) * bar;
                for (int i = 0; i < ccsd_fov_response_.size(); ++i)
                    for (const auto& [j, bar] : ccsd_fov_response_[i]) analytic += (*ov)(i, j) * bar;
                for (int i = 0; i < ccsd_fvv_response_.size(); ++i)
                    for (const auto& [j, bar] : ccsd_fvv_response_[i]) analytic += (*vv)(i, j) * bar;
                PairBlocks hov, hvv;
                for (int p = 0; p < ij_to_i_j_.size(); ++p) {
                    const auto ij = ij_to_i_j_[p];
                    if (ij.first > ij.second) continue;
                    const auto& paos = lmopair_to_paos_[p];
                    hov[ij] = linalg::doublet(submatrix_rows_and_cols(*ov, lmopair_to_lmos_[p], paos), X_pno_[p]);
                    hvv[ij] = linalg::triplet(X_pno_[p], submatrix_rows_and_cols(*vv,paos,paos),
                                              X_pno_[p],true,false,false);
                }
                std::tie(fock1,fock2) = ccsd_fock_source(oo,hov,hvv);
                for (int i = 0; i < fock1.size(); ++i) {
                    dual += lambda_ia_[i]->vector_dot(fock1[i]) +
                            2.*T_ia_[i]->vector_dot(row_as_column(hov.at({i,i}),lmopair_to_lmos_dense_[i_j_to_ij_[i][i]][i]));
                }
                for (int p = 0; p < ij_to_i_j_.size(); ++p) {
                    auto [i,j] = ij_to_i_j_[p];
                    auto b = fock2.at({std::min(i,j),std::max(i,j)});
                    dual += .5*lambda_iajb_[p]->vector_dot(i > j ? b->transpose() : b);
                }
            } else {
                auto ds = add(S_pao_, {}, true);
                for (int i = 0; i < ccsd_overlap_response_.size(); ++i)
                    for (const auto& [j, bar] : ccsd_overlap_response_[i]) analytic += (*ds)(i, j) * bar;
            }
            auto restore = [&]() {
                for (auto& v : vars) *v.slot = v.original;
                L_iajb_ = orig_l;
                if (group == "OVERLAP") compute_pno_overlaps();
            };
            auto sample = [&](double step) {
                for (auto& v : vars) {
                    *v.slot = v.original->clone();
                    (*v.slot)->axpy(step, v.direction);
                }
                if (group == "INTEGRAL")
                    for (int p = 0; p < K_iajb_.size(); ++p) {
                        L_iajb_[p] = K_iajb_[p]->clone();
                        L_iajb_[p]->scale(2.);
                        L_iajb_[p]->subtract(K_iajb_[p]->transpose());
                    }
                if (group == "OVERLAP") compute_pno_overlaps();
                if (group == "FOCK")
                    return ccsd_lagrangian(lambda_ia_, lambda_iajb_, step > 0 ? &plus1 : &minus1,
                                           step > 0 ? &plus2 : &minus2);
                return ccsd_lagrangian(lambda_ia_, lambda_iajb_);
            };
            double numeric;
            try {
                constexpr double h = 1.e-5;
                double plus = sample(h);
                numeric = (plus - sample(-h)) / (2 * h);
                if (group == "FOCK") {
                    auto diff = [h](SharedMatrix plus, SharedMatrix minus, SharedMatrix b) {
                        plus->subtract(minus);
                        plus->scale(.5/h);
                        plus->subtract(b);
                        return plus->absmax();
                    };
                    double e1 = 0., e2 = 0.;
                    for (int i = 0; i < fock1.size(); ++i)
                        e1 = std::max(e1, diff(plus1[i],minus1[i],fock1[i]));
                    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
                        auto [i,j] = ij_to_i_j_[p];
                        auto b = fock2.at({std::min(i,j),std::max(i,j)});
                        e2 = std::max(e2, diff(plus2[p],minus2[p],i > j ? b->transpose() : b));
                    }
                    set_scalar_variable("DLPNO CCSD FOCK R1 TANGENT ERROR",e1);
                    set_scalar_variable("DLPNO CCSD FOCK R2 TANGENT ERROR",e2);
                    set_scalar_variable("DLPNO CCSD FOCK TANGENT DUALITY ERROR",std::abs(dual-analytic));
                }
            } catch (...) {
                restore();
                throw;
            }
            restore();
            outfile->Printf("    CCSD %s direction: analytic % .12e, numerical % .12e, error %.3e\n", group.c_str(),
                            analytic, numeric, std::abs(analytic - numeric));
            set_scalar_variable("DLPNO CCSD " + group + " DIRECTION ERROR", std::abs(analytic - numeric));
        }
    } catch (...) {
        restore_fit();
        throw;
    }
    restore_fit();
    prepare_ccsd_residual();
}

}  // namespace dlpno
}  // namespace psi
