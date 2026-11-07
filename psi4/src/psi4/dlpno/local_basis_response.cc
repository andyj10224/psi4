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
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include <algorithm>
#include <cmath>

namespace psi {
namespace dlpno {

void DLPNOCCSD_T::add_local_block(SparseResponseRows& dest, const std::vector<int>& rows, const std::vector<int>& cols,
                                  const SharedMatrix& block) {
    for (int u = 0; u < rows.size(); ++u)
        for (int v = 0; v < cols.size(); ++v) dest[rows[u]][cols[v]] += (*block)(u, v);
}

// On a fixed Cholesky pivot span an orthonormal parent B may be transported
// with dB=-1/2 B(B^T dS B). The orbital equations use full Fock blocks in
// this frame. No division by a kept-kept orbital-energy splitting is needed.
SharedMatrix DLPNOCCSD_T::parent_overlap_bar(const SharedMatrix& B, const SharedMatrix& bar) {
    // Partial Cholesky pads unselected AO rows with exact zeros. If the
    // subsequent eigenvalue filter also removed a direction inside the pivot
    // block, a fixed-span derivative alone would omit its rotation.
    int active_rows = 0;
    for (int u = 0; u < B->nrow(); ++u) {
        for (int a = 0; a < B->ncol(); ++a)
            if ((*B)(u, a) != 0.) {
                ++active_rows;
                break;
            }
    }
    if (active_rows > B->ncol())
        throw PSIEXCEPTION("PAO metric response requires a full-rank retained Cholesky pivot block.");
    auto inner = linalg::doublet(B, bar, true, false);
    inner->hermitivitize();
    inner->scale(-.5);
    return linalg::triplet(B, inner, B, false, false, true);
}

// Left coefficient and metric derivatives of S_pair,triplet = Xp^T M Y.
// The caller accumulates the right coefficient derivative separately.
void DLPNOCCSD_T::reverse_pair_triplet_overlap(int p, int r, const SharedMatrix& Y, const SharedMatrix& bar) {
    const auto& left = lmopair_to_paos_[p];
    const auto& right = lmotriplet_to_paos_[r];
    auto M = submatrix_rows_and_cols(*S_pao_, left, right);
    triples_pair_bar_[p]->add(linalg::triplet(M, Y, bar, false, false, true));
    add_local_block(triples_overlap_bar_, left, right, linalg::triplet(X_pno_[p], bar, Y, false, false, true));
}

namespace {
SharedMatrix parent_coefficient_bar(const SharedMatrix& op, const SharedMatrix& B, const SharedMatrix& g) {
    auto r = linalg::triplet(op, B, g, false, false, true);
    r->add(linalg::triplet(op, B, g, true, false, false));
    return r;
}
SharedMatrix basis_direction(const SharedMatrix& x, int seed, bool symmetric = false) {
    auto d = x->clone();
    for (int a = 0; a < d->nrow(); ++a)
        for (int b = 0; b < d->ncol(); ++b)
            (*d)(a, b) =
                std::sin(.32 * (1 + a) + .71 * (1 + b) + .59 * seed) / std::sqrt(1. + double(x->nrow()) * x->ncol());
    if (symmetric) d->hermitivitize();
    return d;
}
SharedMatrix basis_shift(const SharedMatrix& a, const SharedMatrix& d, double h) {
    auto v = a->clone();
    v->axpy(h, d);
    return v;
}

double probe_parent(const SharedMatrix& B, const SharedMatrix& bar, const SharedMatrix& S) {
    if (!B) return 0.;
    auto inv = linalg::triplet(B, S, B, true, false, false);
    inv->power(-.5);
    return bar->vector_dot(linalg::doublet(B, inv));
}
}  // namespace

// The pair coefficient derivatives from CCSD and (T) have already supplied
// the final projector's contribution to the LMP2 adjoint. Finish the direct
// first-space MP2 and parent metric terms without counting that source again.
void DLPNOCCSD_T::finish_local_basis_response() {
    auto merge = [](SparseResponseRows& dst, const SparseResponseRows& src) {
        for (int u = 0; u < src.size(); ++u)
            for (const auto& [v, g] : src[u]) dst[u][v] += g;
    };
    merge(ccsd_foo_response_, triples_foo_bar_);
    merge(ccsd_fov_response_, triples_fov_bar_);
    merge(ccsd_fvv_response_, triples_fvv_bar_);
    merge(ccsd_overlap_response_, triples_overlap_bar_);
    MetricShellBlocks metric(ribasis_->nshell());
    std::vector<ThreeCenterDensity> density(ribasis_->nbf());
    double projector_error = 0., projector_central_error = 0., metric_error = 0.;
    std::vector<std::vector<double>> projector_probe_rows;
    for (int p = 0; p < X_pno_.size(); ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        if (i > j) continue;
        const auto& first = initial_pno_selection_[p];
        const auto& last = final_pno_selection_[p];
        auto bar = ccsd_pno_coefficient_response_[p]->clone();
        bar->add(triples_pair_bar_[p]);
        if (i < j) {
            int rev = ij_to_ji_[p];
            bar->add(ccsd_pno_coefficient_response_[rev]);
            bar->add(triples_pair_bar_[rev]);
        }
        const auto& paos = lmopair_to_paos_[p];
        SharedMatrix parent = X_pno_[p], parent_bar = bar;
        if (first.parent) {
            parent = first.parent;
            auto b1 = linalg::doublet(bar, initial_pno_response_[p].final_transform, false, true);
            const int n = parent->ncol(), k = first.canonical_rotation->nrow();
            auto kbar = std::make_shared<Matrix>(n, n), fbar = kbar->clone();
            double ds = initial_pno_mp2_bar(p, b1, kbar, fbar);
            ccsd_foo_response_[i][i] += ds;
            ccsd_foo_response_[j][j] += ds;
            auto fit = pair_exchange_fit(p);
            auto F = submatrix_rows_and_cols(*F_pao_, paos, paos);
            auto dK = linalg::triplet(parent, kbar, parent, false, false, true);
            auto di = linalg::doublet(fit.bj, dK, false, true), dj = linalg::doublet(fit.bi, dK);
            auto dJ = linalg::doublet(fit.bi, di, false, true);
            dJ->scale(-1.);
            dJ->hermitivitize();
            if (debug_) {
                auto dm = basis_direction(fit.J, 1, true), dqi = basis_direction(fit.qi, 2 + i),
                     dqj = basis_direction(fit.qj, 2 + j);
                auto df = basis_direction(F, 4, true);
                double doi = .21, doj = i == j ? doi : -.17;
                double analytic = dJ->vector_dot(dm) + di->vector_dot(dqi) + dj->vector_dot(dqj) + ds * (doi + doj) +
                                  fbar->vector_dot(linalg::triplet(parent, df, parent, true, false, false));
                auto sample = [&](double h) {
                    auto a = basis_shift(fit.qi, dqi, h), b = basis_shift(fit.qj, dqj, h);
                    C_DGESV_wrapper(basis_shift(fit.J, dm, h), a);
                    auto K = linalg::doublet(a, b, true, false);
                    auto updated = transported_first_pno(p, K, basis_shift(F, df, h), (*F_lmo_)(i, i) + h * doi,
                                                         (*F_lmo_)(j, j) + h * doj);
                    return b1->vector_dot(updated);
                };
                constexpr double h = 1.e-7;
                const double coarse = (sample(h)-sample(-h))/(2.*h);
                const double fine = (sample(h/2.)-sample(-h/2.))/h;
                // Small PNO occupation gaps make the O(h^2) term visible at
                // the old spacing. Cancel it without relaxing the tolerance.
                const double err = std::abs(analytic-(4.*fine-coarse)/3.);
                if (!std::isfinite(err)) throw PSIEXCEPTION("Nonfinite combined PNO projector probe.");
                projector_error = std::max(projector_error, err);
                projector_central_error = std::max(projector_central_error,std::abs(analytic-coarse));
                if (debug_ > 1) {
                    for (double step : {1.e-5,3.e-6,1.e-6,3.e-7,1.e-7,3.e-8,1.e-8}) {
                        const double numeric = (sample(step)-sample(-step))/(2.*step);
                        projector_probe_rows.push_back({double(i),double(j),step,analytic,numeric});
                    }
                }
            }
            scatter_pair_fit(p, dJ, di, dj, metric, density);
            add_local_block(ccsd_fvv_response_, paos, paos, linalg::triplet(parent, fbar, parent, false, false, true));
            Dimension zero(1), keep(1);
            keep.fill(k);
            auto natural = first.eigenvectors->get_block({zero, first.eigenvectors->rowspi()}, {zero, keep});
            auto rot = linalg::doublet(natural, first.canonical_rotation);
            parent_bar = linalg::doublet(b1, rot, false, true);
            parent_bar->add(parent_coefficient_bar(fit.K, parent, kbar));
            parent_bar->add(parent_coefficient_bar(F, parent, fbar));
        } else if (last.parent) {
            parent = last.parent;
            Dimension zero(1), keep(1);
            keep.fill(last.canonical_rotation->nrow());
            auto natural = last.eigenvectors->get_block({zero, last.eigenvectors->rowspi()}, {zero, keep});
            auto rot = linalg::doublet(natural, last.canonical_rotation);
            parent_bar = linalg::doublet(bar, rot, false, true);
        }
        auto ds = parent_overlap_bar(parent, parent_bar);
        if (truncation_parent_[p]) ds->add(parent_overlap_bar(truncation_parent_[p], truncation_parent_bar_[p]));
        if (lmp2_parent_[p]) ds->add(parent_overlap_bar(lmp2_parent_[p], lmp2_parent_bar_[p]));
        if (debug_) {
            auto S = submatrix_rows_and_cols(*S_pao_, paos, paos);
            auto dS = linalg::triplet(S, basis_direction(S, 5, true), S);
            auto sample = [&](double h) {
                auto M = basis_shift(S, dS, h);
                return probe_parent(parent, parent_bar, M) +
                       probe_parent(truncation_parent_[p], truncation_parent_bar_[p], M) +
                       probe_parent(lmp2_parent_[p], lmp2_parent_bar_[p], M);
            };
            constexpr double h = 1.e-5;
            double err = std::abs(ds->vector_dot(dS) - (sample(h) - sample(-h)) / (2 * h));
            if (!std::isfinite(err)) throw PSIEXCEPTION("Nonfinite parent PAO metric probe.");
            metric_error = std::max(metric_error, err);
        }
        add_local_block(ccsd_overlap_response_, paos, paos, ds);
    }
    if (debug_) {
        set_scalar_variable("DLPNO PNO PROJECTOR OPERATOR DIRECTION ERROR", projector_error);
        set_scalar_variable("DLPNO PNO PROJECTOR OPERATOR CENTRAL ERROR", projector_central_error);
        set_scalar_variable("DLPNO PARENT PAO METRIC DIRECTION ERROR", metric_error);
        if (!projector_probe_rows.empty()) {
            auto rows = std::make_shared<Matrix>(projector_probe_rows.size(),5);
            for (int r = 0; r < rows->nrow(); ++r)
                for (int c = 0; c < 5; ++c) (*rows)(r,c) = projector_probe_rows[r][c];
            set_array_variable("DLPNO PNO PROJECTOR OPERATOR STEP STUDY",rows);
        }
    }
    const std::string m = "DLPNO PNO PROJECTOR AUXILIARY METRIC GRADIENT",
                      q = "DLPNO PNO PROJECTOR THREE-CENTER GRADIENT";
    set_array_variable(m, contract_metric(metric, m));
    set_array_variable(q, contract_three_center(density, q));
    triples_pair_bar_.clear();
    triples_foo_bar_.clear();
    triples_fov_bar_.clear();
    triples_fvv_bar_.clear();
    triples_overlap_bar_.clear();
    truncation_parent_.clear();
    truncation_parent_bar_.clear();
    lmp2_parent_.clear();
    lmp2_parent_bar_.clear();
    initial_pno_selection_.clear();
    final_pno_selection_.clear();
    initial_pno_response_.clear();
    ccsd_lmp2_projector_source_.clear();
    ccsd_pno_coefficient_response_.clear();
}

// Fitted LMOs satisfy S_bb L = (S C_lmo)_{b,i} independently on each
// auxiliary center. Accumulate both sides of the differentiated local solve.
void DLPNOCCSD_T::finish_df_refit_response() {
    df_refit_overlap_bar_.assign(C_pao_->nrow(), {});
    df_sclmo_bar_.assign(C_lmo_->nrow(), {});
    double error = 0.;
    for (int atom = 0; atom < fitted_coefficient_bar_.size(); ++atom) {
        const auto& bar = fitted_coefficient_bar_[atom];
        if (!bar) continue;
        const auto& bfs = riatom_to_bfs1_[atom];
        const auto& occ = riatom_to_lmos_ext_[atom];
        auto S = submatrix_rows_and_cols(*reference_wavefunction_->S(), bfs, bfs);
        auto z = bar->clone();
        C_DGESV_wrapper(S->clone(), z);
        const auto& C = gradient_lmo_slices_[atom];
        auto ds = linalg::doublet(z, C, false, true);
        ds->scale(-1.);
        add_local_block(df_refit_overlap_bar_, bfs, bfs, ds);
        add_local_block(df_sclmo_bar_, bfs, occ, z);
        if (debug_) {
            // A relative metric perturbation keeps ill-conditioned overlap blocks
            // on the same invertible branch.
            auto dm = S->clone();
            auto dr = z->clone();
            for (int a = 0; a < dm->nrow(); ++a)
                for (int b = 0; b < dm->ncol(); ++b) (*dm)(a, b) *= .05 * (std::sin(1. + a) + std::sin(1. + b));
            for (int a = 0; a < dr->nrow(); ++a)
                for (int b = 0; b < dr->ncol(); ++b)
                    (*dr)(a, b) =
                        std::cos(.43 * (1 + a) + .71 * (1 + b)) / std::sqrt(1. + double(dr->nrow()) * dr->ncol());
            auto sample = [&](double h) {
                auto M = S->clone(), rhs = linalg::doublet(S, C);
                M->axpy(h, dm);
                rhs->axpy(h, dr);
                C_DGESV_wrapper(M, rhs);
                return rhs->vector_dot(bar);
            };
            constexpr double h = 1.e-5;
            double e = std::abs((sample(h) - sample(-h)) / (2 * h) - z->vector_dot(dr) - ds->vector_dot(dm));
            if (!std::isfinite(e)) throw PSIEXCEPTION("Nonfinite LMO refit probe.");
            error = std::max(error, e);
        }
    }
    if (debug_) {
        set_scalar_variable("DLPNO DF ORBITAL REPLAY ERROR", df_replay_error_);
        set_scalar_variable("DLPNO DF ORBITAL COEFFICIENT DIRECTION ERROR", df_coefficient_error_);
        set_scalar_variable("DLPNO DF LMO REFIT DIRECTION ERROR", error);
    }
    fitted_coefficient_bar_.clear();
}

// Write P = B N, where B is the unnormalized occupied-projected AO matrix
// and N_uu = 1 / sqrt((B^T S B)_uu) on the active branch. Factor the result
// by operator, so the reverse pass visits only the local rows already used.
void DLPNOCCSD_T::reverse_pao_normalization() {
    const auto& N = gradient_pao_inv_norm_;
    std::vector<double> column(N.size(), 0.);
    raw_pao_response_ = {std::move(ccsd_fov_response_), std::move(ccsd_fvv_response_),
                         std::move(ccsd_overlap_response_), std::move(df_pao_coefficient_bar_)};
    double plus = 0., minus = 0.;
    constexpr double h = 1.e-5;
    auto direction = [](int u, int v, int seed, bool sym) {
        if (sym && u > v) std::swap(u, v);
        return .03 * std::sin(.37 * (u + 1) + .71 * (v + 1) + .59 * seed);
    };
    // Construct independent raw-block directions for a fixed-geometry check.
    // Scale them to give bounded variations in the normalized frame.
    auto reverse = [&](SparseResponseRows& rows, const SharedMatrix& value, bool both, int seed, bool symmetric) {
        for (int u = 0; u < rows.size(); ++u)
            for (auto& [v, g] : rows[u]) {
                const double y = (*value)(u, v);
                column[v] += g * y;
                if (both) column[u] += g * y;
                const double factor = (both ? N[u] : 1.) * N[v];
                if (debug_ && factor) {
                    const double d = direction(u, v, seed, symmetric);
                    auto sample = [&](double step) {
                        double norm = std::sqrt(1. + step * direction(v, v, 3, true));
                        if (both) norm *= std::sqrt(1. + step * direction(u, u, 3, true));
                        return g * (y + step * d) / norm;
                    };
                    plus += sample(h);
                    minus += sample(-h);
                }
                g *= factor;
            }
    };
    reverse(raw_pao_response_.fov, F_lmo_pao_, false, 1, false);
    reverse(raw_pao_response_.fvv, F_pao_, true, 2, true);
    reverse(raw_pao_response_.overlap, S_pao_, true, 3, true);
    reverse(raw_pao_response_.coefficients, C_pao_, false, 4, false);
    for (int u = 0; u < N.size(); ++u) {
        if (!N[u] || column[u] == 0.) continue;
        double dn = -.5 * column[u] * N[u] * N[u];
        raw_pao_response_.overlap[u][u] += dn;
    }
    size_t entries = 0;
    for (const auto* rows :
         {&raw_pao_response_.fov, &raw_pao_response_.fvv, &raw_pao_response_.overlap, &raw_pao_response_.coefficients})
        for (const auto& row : *rows) entries += row.size();
    set_scalar_variable("DLPNO LOCAL PAO RESPONSE ENTRIES", entries);
    if (debug_) {
        auto dot = [&](const SparseResponseRows& rows, bool both, int seed, bool sym) {
            double val = 0.;
            for (int u = 0; u < rows.size(); ++u)
                for (const auto& [v, g] : rows[u]) {
                    const double scale = N[v] * (both ? N[u] : 1.);
                    if (scale) val += g * (direction(u, v, seed, sym) / scale);
                }
            return val;
        };
        const double analytic =
            dot(raw_pao_response_.fov, false, 1, false) + dot(raw_pao_response_.fvv, true, 2, true) +
            dot(raw_pao_response_.overlap, true, 3, true) + dot(raw_pao_response_.coefficients, false, 4, false);
        double err = std::abs(analytic - (plus - minus) / (2 * h));
        if (!std::isfinite(err)) throw PSIEXCEPTION("Nonfinite PAO normalization probe.");
        set_scalar_variable("DLPNO PAO NORMALIZATION DIRECTION ERROR", err);
    }
}

void DLPNOCCSD_T::check_triples_local_basis(const std::function<double()>& evaluate, bool measure) {
    // Independent perturbations of the right iterative (T) energy. Rebuild
    // TNO densities, semicanonical orbitals, W/V and the T3 amplitudes.
    const auto X = X_pno_, tno = X_tno_;
    const auto eps = e_tno_;
    const auto selection = tno_selection_;
    const auto ranks = n_tno_;
    const auto fo = F_lmo_, fv = F_pao_, ov = F_lmo_pao_, sm = S_pao_;
    std::vector<SharedMatrix> dx(X.size());
    for (int p = 0; p < X.size(); ++p)
        if (ij_to_i_j_[p].first <= ij_to_i_j_[p].second) dx[p] = dx[ij_to_ji_[p]] = basis_direction(X[p], p + 1, false);
    auto dfo = basis_direction(fo, 2, true), dfv = basis_direction(fv, 3, true), dov = basis_direction(ov, 4, false);
    // A bounded bilinear perturbation within the range of the PAO metric.
    auto dsm = linalg::triplet(sm, basis_direction(sm, 5, true), sm);
    if (measure) {
        std::vector<SharedMatrix> parents(tno.size());
        for (int r = 0; r < tno.size(); ++r) parents[r] = selection[r].natural ? selection[r].natural : tno[r];
        auto restore = [&]() {
            X_pno_ = X;
            X_tno_ = tno;
            e_tno_ = eps;
            tno_selection_ = selection;
            n_tno_ = ranks;
            F_lmo_ = fo;
            F_pao_ = fv;
            F_lmo_pao_ = ov;
            S_pao_ = sm;
        };
        for (int group = 0; group < 3; ++group) {
            auto sample = [&](double h) {
                restore();
                if (group == 0) {
                    for (int p = 0; p < X.size(); ++p)
                        if (ij_to_i_j_[p].first <= ij_to_i_j_[p].second)
                            X_pno_[p] = X_pno_[ij_to_ji_[p]] = basis_shift(X[p], dx[p], h);
                } else if (group == 1) {
                    F_lmo_ = basis_shift(fo, dfo, h);
                    F_pao_ = basis_shift(fv, dfv, h);
                    F_lmo_pao_ = basis_shift(ov, dov, h);
                } else
                    S_pao_ = basis_shift(sm, dsm, h);
                tno_transform(options_.get_double("T_CUT_TNO"), false, &ranks, &parents);
                return evaluate();
            };
            try {
                constexpr double h = 2.e-6;
                double plus = sample(h);
                triples_local_probe_[group] = (plus - sample(-h)) / (2 * h);
            } catch (...) {
                restore();
                throw;
            }
            restore();
        }
        return;
    }
    auto dot = [](const SparseResponseRows& rows, const SharedMatrix& d) {
        double v = 0.;
        for (int i = 0; i < rows.size(); ++i)
            for (const auto& [j, z] : rows[i]) v += z * (*d)(i, j);
        return v;
    };
    std::array<double, 3> analytic{0.,
                                   dot(triples_foo_bar_, dfo) + dot(triples_fov_bar_, dov) + dot(triples_fvv_bar_, dfv),
                                   dot(triples_overlap_bar_, dsm)};
    for (int p = 0; p < X.size(); ++p) analytic[0] += triples_pair_bar_[p]->vector_dot(dx[p]);
    const std::array<std::string, 3> name{"PNO COEFFICIENT", "FOCK", "PAO OVERLAP"};
    for (int g = 0; g < 3; ++g) {
        double err = std::abs(analytic[g] - triples_local_probe_[g]);
        if (!std::isfinite(err)) throw PSIEXCEPTION("Nonfinite (T) local basis response.");
        outfile->Printf("    (T) %s direction: analytic % .12e, numerical % .12e, error %.3e\n", name[g].c_str(),
                        analytic[g], triples_local_probe_[g], err);
        set_scalar_variable("DLPNO (T) " + name[g] + " DIRECTION ERROR", err);
    }
}
}  // namespace dlpno
}  // namespace psi
