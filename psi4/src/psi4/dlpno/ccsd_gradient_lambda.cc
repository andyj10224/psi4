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
#include "psi4/libdiis/diismanager.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <algorithm>
#include <cmath>

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
            auto Kproj = linalg::triplet(S, K_iajb_[kl], S, true, false, false);
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
        auto M = K_mibj_[p]->clone();
        M->add(linalg::doublet(T_n_ij_[p], K_iajb_[p]));
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
            auto reverse_mixed = [&](int a, int b, const SharedMatrix& intermediate, SharedMatrix bare,
                                     const SharedMatrix& amp, const SharedMatrix& seed, double factor,
                                     const SharedMatrix& d_amp, const SharedMatrix& d_intermediate) {
                auto s1 = S_PNO(p, a), s2 = S_PNO(a, b), s3 = S_PNO(b, p);
                bare->add(linalg::triplet(s1, intermediate, s2));
                auto dz = linalg::triplet(seed, s3, amp, false, true, false);
                d_intermediate->axpy(factor, linalg::triplet(s1, dz, s2, true, false, true));
                d_amp->axpy(factor, linalg::triplet(s3, seed, bare, false, true, false));
            };
            reverse_mixed(ki, kj, gamma_[ki], J_ikac_non_proj_[p][m]->clone(), T_iajb_[kj], yc, -1., g.t2[kj],
                          g.gamma[ki]);
            auto M = K_iakc_non_proj_[p][m]->clone();
            M->scale(2.);
            M->subtract(J_ikac_non_proj_[p][m]);
            reverse_mixed(ik, jk, delta_[ik], M, Tt_iajb_[jk], y, 0.5, g.u[jk], g.delta[ik]);
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
            g.u[lb]->axpy(g.fkj[p], linalg::triplet(S, K_iajb_[la], S, true, false, false));
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
                auto tl = linalg::doublet(s1, T_ia_[l]);
                // - t_l^p (S_p,other K^T S_other,ii t_i)^T
                auto map = linalg::triplet(so, integral, s2, false, true, false);
                auto v = linalg::doublet(map, T_ia_[i]);
                g.t1[l]->subtract(linalg::triplet(s1, z, v, true, false, false));
                g.t1[i]->subtract(linalg::triplet(map, z, tl, true, true, false));
                // +/- 1/2 S_p,amp (t or u)_amp S_amp,other K S_other,p
                auto right = linalg::triplet(S_PNO(amp, other), integral, S_PNO(other, p));
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

void DLPNOCCSD::build_gradient_lambda_fixed() {
    timer_on("DLPNO Lambda fixed intermediates");
    auto fixed = std::make_unique<GradientLambdaFixed>();
    fixed->q.resize(T_iajb_.size());
    fixed->f.resize(T_iajb_.size());
    const size_t peak = ccsd_response_words_ * sizeof(double);
    const size_t headroom = 0.9 * memory_ > peak ? size_t(0.9 * memory_) - peak : 0;
    const size_t budget = std::min({size_t(64) * 1024 * 1024, size_t(memory_ / 32), headroom});
    size_t used = 0, cached_q = 0, cached_f = 0;
    for (size_t p = 0; p < T_iajb_.size(); ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        if (i > j) continue;
        const size_t n = n_pno_[p], words = n * n;
        if (words * sizeof(double) <= budget - used) {
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
            used += words * sizeof(double);
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
    gradient_lambda_fixed_ = std::move(fixed);
    outfile->Printf("    Lambda fixed intermediates: %zu Q pairs, %zu F pairs, %.1f MiB\n",
                    cached_q, cached_f, used / 1048576.);
    timer_off("DLPNO Lambda fixed intermediates");
}

void DLPNOCCSD::solve_gradient_lambda() {
    reset_lambda_state();
    estimate_gradient_lambda_memory();
    prepare_ccsd_residual();
    beta_ = compute_beta();
    gamma_ = compute_gamma();
    delta_ = compute_delta();
    F_im_double_tilde_ = compute_Fkj_double_tilde();
    lambda_ia_ = zero_blocks(T_ia_);
    lambda_iajb_ = zero_blocks(T_iajb_);
    std::vector<SharedMatrix> l1(T_ia_.size()), l2(T_iajb_.size());
    build_gradient_lambda_fixed();
    // Clear on both normal exit and exceptions, before any debug amplitude
    // perturbation. Cached matrices are read-only throughout this scope.
    struct FixedGuard {
        std::unique_ptr<GradientLambdaFixed>& fixed;
        ~FixedGuard() { fixed.reset(); }
    } fixed_guard{gradient_lambda_fixed_};
    std::vector<SharedMatrix> source1(T_ia_.size()), source2(T_iajb_.size());
    gradient_lambda_residual(lambda_ia_, lambda_iajb_, source1, source2, true);
    auto transpose_workspace = std::make_unique<CCAdjoints>(*this, lambda_ia_, lambda_iajb_);

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
    gradient_lambda_fixed_.reset();
    transpose_workspace.reset();
    source1.clear();
    source2.clear();
    form_ccsd_pair_response();
    backtransform_ccsd_one_body();
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
