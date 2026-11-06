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

#include <algorithm>
#include <cmath>

namespace psi {
namespace dlpno {
namespace {
SharedMatrix project(const SharedMatrix& a, const SharedMatrix& x) {
    return linalg::triplet(x, a, x, true, false, false);
}
SharedMatrix lift(const SharedMatrix& a, const SharedMatrix& x) { return linalg::triplet(x, a, x, false, false, true); }
SharedMatrix spin_adapt(const SharedMatrix& t) {
    auto u = t->clone();
    u->scale(2.);
    u->subtract(t->transpose());
    return u;
}
// Coefficient derivative of bar : X^T A X, including nonsymmetric K and T.
SharedMatrix basis_bar(const SharedMatrix& a, const SharedMatrix& x, const SharedMatrix& bar) {
    auto g = linalg::triplet(a, x, bar, false, false, true);
    g->add(linalg::triplet(a, x, bar, true, false, false));
    return g;
}

struct SinglesLoss {
    double energy = 0., occ = 0.;
    SharedMatrix ov, vv;
};

// f^T (e_i I-F)^(-1) f with the same denominator screen as the energy.
// The divided difference includes rotations across the fixed spectral mask.
SinglesLoss singles(const SharedMatrix& f, const SharedMatrix& F, double oi) {
    const int n = F->nrow();
    auto V = F->clone();
    Vector eps(n);
    F->diagonalize(*V, eps);
    auto v = linalg::doublet(V, f, true, false);
    auto t = v->clone();
    std::vector<double> inv(n);
    for (int a = 0; a < n; ++a) {
        const double gap = oi - eps(a);
        inv[a] = std::abs(gap) > 1.e-12 ? 1. / gap : 0.;
        (*t)(a, 0) *= inv[a];
    }
    auto d = std::make_shared<Matrix>(n, n);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            double factor = inv[a] * inv[b];
            if ((inv[a] == 0.) != (inv[b] == 0.)) factor = (inv[a] - inv[b]) / (eps(a) - eps(b));
            (*d)(a, b) = (*v)(a, 0) * factor * (*v)(b, 0);
        }
    SinglesLoss r;
    r.energy = v->vector_dot(t);
    r.occ = -t->vector_dot(t);
    r.ov = linalg::doublet(V, t);
    r.ov->scale(2.);
    r.vv = lift(d, V);
    return r;
}

struct PairLoss {
    double energy, occ;
    SharedMatrix K, T, F, f, X;
};
// One orientation of E_full - E_kept. X contains retained density eigenvectors;
// a rotation within their span cancels from this scalar.
PairLoss truncated_pair(const SharedMatrix& K, const SharedMatrix& t, const SharedMatrix& F, const SharedMatrix& f,
                        const SharedMatrix& X, double oi, bool diagonal) {
    auto u = spin_adapt(t), k = project(K, X), v = project(u, X);
    auto dk = u->clone();
    dk->subtract(lift(v, X));
    auto du = K->clone();
    du->subtract(lift(k, X));
    auto dx = basis_bar(K, X, v);
    dx->add(basis_bar(u, X, k));
    dx->scale(-1.);
    PairLoss g{
        K->vector_dot(u) - k->vector_dot(v),    0., dk, spin_adapt(du), std::make_shared<Matrix>(F->nrow(), F->ncol()),
        std::make_shared<Matrix>(F->nrow(), 1), dx};
    if (diagonal) {
        auto full = singles(f, F, oi);
        auto small = singles(linalg::doublet(X, f, true, false), project(F, X), oi);
        g.energy += full.energy - small.energy;
        g.occ = full.occ - small.occ;
        g.f = full.ov;
        g.f->subtract(linalg::doublet(X, small.ov));
        g.F = full.vv;
        g.F->subtract(lift(small.vv, X));
        g.X->subtract(linalg::doublet(f, small.ov, false, true));
        g.X->subtract(basis_bar(F, X, small.vv));
    }
    return g;
}

SharedMatrix retained(const SharedMatrix& eig, int rank) {
    Dimension start(1), end(1);
    end.fill(rank);
    return eig->get_block({start, eig->rowspi()}, {start, end});
}

// Reverse the semicanonical MP2 equation in an orthonormal parent basis.
// Keeping the full Fock derivative also covers off-diagonal perturbations.
double reverse_mp2(PairLoss& g, const SharedMatrix& t, const SharedVector& eps, double occ_sum) {
    auto z = g.T->clone();
    for (int a = 0; a < t->nrow(); ++a)
        for (int b = 0; b < t->ncol(); ++b) (*z)(a, b) /= occ_sum - (*eps)(a) - (*eps)(b);
    g.K->add(z);
    auto df = linalg::doublet(z, t, false, true);
    df->add(linalg::doublet(t, z, true, false));
    df->hermitivitize();
    g.F->add(df);
    return -z->vector_dot(t);
}

// Independent forward calculations for the fixed-geometry probes.
SharedMatrix probe_mp2_amplitudes(const SharedMatrix& K, const SharedMatrix& F, double occ_sum) {
    auto V = F->clone();
    Vector eps(F->nrow());
    F->diagonalize(*V, eps);
    auto t = project(K, V);
    for (int a = 0; a < t->nrow(); ++a)
        for (int b = 0; b < t->ncol(); ++b) (*t)(a, b) /= occ_sum - eps(a) - eps(b);
    return lift(t, V);
}

SharedMatrix probe_selected_pnos(const SharedMatrix& t, int rank, bool diagonal) {
    auto u = spin_adapt(t);
    auto D = linalg::doublet(u, t, false, true);
    D->add(linalg::doublet(u, t, true, false));
    if (diagonal) D->scale(.5);
    auto V = D->clone();
    Vector occ(D->nrow());
    D->diagonalize(*V, occ, descending);
    return retained(V, rank);
}

SharedMatrix align_pnos(const SharedMatrix& X, const SharedMatrix& ref, const SharedMatrix& parent) {
    auto projected = linalg::triplet(X, X, ref, false, true, false);
    auto norm = linalg::doublet(projected, projected, true, false);
    norm->power(-.5);
    return linalg::triplet(parent, projected, norm);
}

double reference_loss(SharedMatrix K, SharedMatrix t, SharedMatrix F, SharedMatrix f, double oi, double oj, int rank,
                      bool initial, bool diagonal, SharedMatrix* selected = nullptr) {
    if (initial) t = probe_mp2_amplitudes(K, F, oi + oj);
    auto u = spin_adapt(t);
    auto x = probe_selected_pnos(t, rank, diagonal);
    if (selected) *selected = x;
    auto forward_singles = [&](SharedMatrix A, SharedMatrix b) {
        auto c = A->clone();
        Vector e(A->nrow());
        A->diagonalize(*c, e);
        b = linalg::doublet(c, b, true, false);
        double en = 0.;
        for (int a = 0; a < b->nrow(); ++a)
            if (std::abs(oi - e(a)) > 1.e-12) en += (*b)(a, 0) * (*b)(a, 0) / (oi - e(a));
        return en;
    };
    double en = K->vector_dot(u) - project(K, x)->vector_dot(project(u, x));
    if (diagonal) en += forward_singles(F, f) - forward_singles(project(F, x), linalg::doublet(x, f, true, false));
    return en;
}

SharedMatrix probe(const SharedMatrix& m, int seed, bool sym = false) {
    auto d = m->clone();
    for (int a = 0; a < d->nrow(); ++a)
        for (int b = 0; b < d->ncol(); ++b)
            (*d)(a, b) =
                std::sin(.39 * (1 + a) + .57 * (1 + b) + .67 * seed) / std::sqrt(1. + double(d->nrow()) * d->ncol());
    if (sym) d->hermitivitize();
    return d;
}
SharedMatrix shifted(const SharedMatrix& x, const SharedMatrix& d, double h) {
    auto y = x->clone();
    y->axpy(h, d);
    return y;
}
void record_error(double& maximum, double error) {
    if (!std::isfinite(error)) throw PSIEXCEPTION("Nonfinite PNO truncation response probe.");
    maximum = std::max(maximum, std::abs(error));
}
}  // namespace

DLPNOCCSD_T::PairExchangeFit DLPNOCCSD_T::pair_exchange_fit(int p) {
    const auto [i, j] = ij_to_i_j_[p];
    const auto& aux = lmopair_to_ribfs_[p];
    const auto& paos = lmopair_to_paos_[p];
    auto qi = std::make_shared<Matrix>(aux.size(), paos.size()), qj = qi->clone();
    for (int a = 0; a < aux.size(); ++a) {
        int q = aux[a], atom = ribasis_->function_to_center(q);
        const auto& l = riatom_to_lmos_ext_dense_[atom];
        for (int v = 0; v < paos.size(); ++v) {
            int u = riatom_to_paos_ext_dense_[atom][paos[v]];
            (*qi)(a, v) = (*qia_[q])(l[i], u);
            (*qj)(a, v) = (*qia_[q])(l[j], u);
        }
    }
    auto J = submatrix_rows_and_cols(*full_metric_, aux, aux);
    auto bi = qi->clone(), bj = qj->clone();
    C_DGESV_wrapper(J->clone(), bi);
    C_DGESV_wrapper(J->clone(), bj);
    return {J, qi, qj, bi, bj, linalg::doublet(bi, qj, true, false)};
}

void DLPNOCCSD_T::scatter_pair_fit(int p, const SharedMatrix& dJ, const SharedMatrix& di, const SharedMatrix& dj,
                                   MetricShellBlocks& metric, std::vector<ThreeCenterDensity>& density) {
    const auto [i, j] = ij_to_i_j_[p];
    const auto& aux = lmopair_to_ribfs_[p];
    const auto& paos = lmopair_to_paos_[p];
    add_metric_density(metric, aux, dJ);
    for (int a = 0; a < aux.size(); ++a) {
        int q = aux[a], atom = ribasis_->function_to_center(q);
        const auto& l = riatom_to_lmos_ext_dense_[atom];
        const auto& v = riatom_to_paos_ext_dense_[atom];
        auto& d = density[q];
        if (!d.oo) {
            size_t no = riatom_to_lmos_ext_[atom].size(), nv = riatom_to_paos_ext_[atom].size();
            d.oo = std::make_shared<Matrix>(no, no);
            d.ov = std::make_shared<Matrix>(no, nv);
            d.vv = std::make_shared<Matrix>(nv, nv);
        }
        for (int u = 0; u < paos.size(); ++u) {
            (*d.ov)(l[i], v[paos[u]]) += (*di)(a, u);
            (*d.ov)(l[j], v[paos[u]]) += (*dj)(a, u);
        }
    }
}

double DLPNOCCSD_T::initial_pno_mp2_bar(int p, const SharedMatrix& coefficient_bar, SharedMatrix& k, SharedMatrix& f) {
    const auto& first = initial_pno_selection_[p];
    const auto [i, j] = ij_to_i_j_[p];
    auto cov = linalg::triplet(first.parent, coefficient_bar, first.canonical_rotation, true, false, true);
    PairLoss child{0., 0., k, projector_amplitude_bar(first, cov, i == j), f, {}, {}};
    return reverse_mp2(child, first.amplitudes, initial_pno_response_[p].virtual_energy,
                       (*F_lmo_)(i, i) + (*F_lmo_)(j, j));
}

SharedMatrix DLPNOCCSD_T::transported_first_pno(int p, const SharedMatrix& K, const SharedMatrix& F, double oi,
                                                double oj) {
    const auto& first = initial_pno_selection_[p];
    if (!first.parent) return lmp2_state_.X[p];
    const auto& parent = first.parent;
    auto t = probe_mp2_amplitudes(project(K, parent), project(F, parent), oi + oj);
    auto X = probe_selected_pnos(t, first.canonical_rotation->nrow(), ij_to_i_j_[p].first == ij_to_i_j_[p].second);
    auto ref = linalg::doublet(retained(first.eigenvectors, X->ncol()), first.canonical_rotation);
    return align_pnos(X, ref, parent);
}

void DLPNOCCSD_T::form_pno_truncation_response() {
    MetricShellBlocks metric(ribasis_->nshell());
    std::vector<ThreeCenterDensity> density(ribasis_->nbf());
    const size_t pairs = ij_to_i_j_.size();
    truncation_parent_.assign(pairs, {});
    truncation_parent_bar_.assign(pairs, {});
    double energy[2]{}, error[2]{}, singles_error[2]{}, nested_df_error = 0.;
    for (int p = 0; p < pairs; ++p) {
        const auto& first = initial_pno_selection_[p];
        const auto& last = final_pno_selection_[p];
        if (!first.parent && !last.parent) continue;
        const auto [i, j] = ij_to_i_j_[p];
        const bool diag = i == j;
        const double weight = diag ? 1. : 2.;
        const double oi = (*F_lmo_)(i, i), oj = (*F_lmo_)(j, j);
        const auto& paos = lmopair_to_paos_[p];
        auto fit = pair_exchange_fit(p);
        const auto& [J, qi, qj, bi, bj, K0] = fit;
        auto Kbar = std::make_shared<Matrix>(paos.size(), paos.size());
        auto F0 = submatrix_rows_and_cols(*F_pao_, paos, paos);
        auto f0 = submatrix_rows_and_cols(*F_lmo_pao_, {i}, paos)->transpose();
        SharedMatrix child_bar;
        // The final loss depends on the first retained coefficients. Pass its
        // coefficient derivative back through the first selection before fitting.
        for (bool initial : {false, true}) {
            const auto& s = initial ? first : last;
            if (!s.parent) continue;
            const auto& A = s.parent;
            auto K = project(K0, A), F = project(F0, A), f = linalg::doublet(A, f0, true, false);
            auto x = retained(s.eigenvectors, s.canonical_rotation->nrow());
            auto g = truncated_pair(K, s.amplitudes, F, f, x, oi, diag);
            energy[initial] += weight * g.energy;
            g.T->add(projector_amplitude_bar(s, g.X, diag));
            double occupied_sum = 0.;
            if (initial) occupied_sum = reverse_mp2(g, s.amplitudes, initial_pno_response_[p].virtual_energy, oi + oj);
            if (debug_) {
                auto dK = probe(K, p + 1, diag), dT = probe(s.amplitudes, p + 2, diag), dF = probe(F, p + 3, true),
                     df = probe(f, p + 4);
                double di = .43, dj = diag ? di : -.29;
                auto check = [&](const SharedMatrix& source_f, const PairLoss& r, double dsum) {
                    double analytic =
                        r.K->vector_dot(dK) + r.F->vector_dot(dF) + r.f->vector_dot(df) + di * r.occ + (di + dj) * dsum;
                    if (!initial) analytic += r.T->vector_dot(dT);
                    auto sample = [&](double h) {
                        return reference_loss(shifted(K, dK, h), shifted(s.amplitudes, dT, h), shifted(F, dF, h),
                                              shifted(source_f, df, h), oi + h * di, oj + h * dj, x->ncol(), initial,
                                              diag);
                    };
                    constexpr double h = 1.e-7;
                    return weight * std::abs(analytic - (sample(h) - sample(-h)) / (2 * h));
                };
                record_error(error[initial], check(f, g, occupied_sum));
                if (diag) {
                    // RHF F_ia is nearly zero. Exercise the singles terms at a
                    // nonzero test value, without changing the physical density.
                    auto test_f = probe(f, 17);
                    test_f->scale(.02);
                    auto test = truncated_pair(K, s.amplitudes, F, test_f, x, oi, true);
                    test.T->add(projector_amplitude_bar(s, test.X, true));
                    double ds = initial
                                    ? reverse_mp2(test, s.amplitudes, initial_pno_response_[p].virtual_energy, oi + oj)
                                    : 0.;
                    record_error(singles_error[initial], check(test_f, test, ds));
                }
            }
            // Scale this stage's own loss once for both ordered orientations.
            g.K->scale(weight);
            g.T->scale(weight);
            g.F->scale(weight);
            g.f->scale(weight);
            occupied_sum *= weight;
            if (initial && child_bar) {
                occupied_sum += initial_pno_mp2_bar(p, child_bar, g.K, g.F);
            }
            ccsd_foo_response_[i][i] += occupied_sum + weight * g.occ;
            ccsd_foo_response_[j][j] += occupied_sum;
            if (!initial) ccsd_lmp2_projector_source_[p]->add(g.T);
            Kbar->add(lift(g.K, A));
            auto vf = lift(g.F, A), vo = linalg::doublet(A, g.f);
            for (int u = 0; u < paos.size(); ++u) {
                if (diag) ccsd_fov_response_[i][paos[u]] += (*vo)(u, 0);
                for (int v = 0; v < paos.size(); ++v) ccsd_fvv_response_[paos[u]][paos[v]] += (*vf)(u, v);
            }
            auto da = basis_bar(K0, A, g.K);
            da->add(basis_bar(F0, A, g.F));
            if (diag) da->add(linalg::doublet(f0, g.f, false, true));
            if (initial && child_bar) {
                auto rot = linalg::doublet(x, s.canonical_rotation);
                da->add(linalg::doublet(child_bar, rot, false, true));
            }
            child_bar = da;
            truncation_parent_[p] = A;
            truncation_parent_bar_[p] = da;
        }
        auto di = linalg::doublet(bj, Kbar, false, true), dj = linalg::doublet(bi, Kbar);
        auto dJ = linalg::doublet(bi, di, false, true);
        dJ->scale(-1.);
        dJ->hermitivitize();
        if (debug_) {
            // Perturb the primitive fit, reconstruct both losses, and transport
            // the converged LMP2 tensor in the first retained space. Polar
            // alignment fixes its internal frame at the reference geometry.
            auto dqi = probe(qi, 2 * p + 1), dqj = diag ? dqi : probe(qj, 2 * p + 2), dm = probe(J, p + 3, true);
            const double analytic = di->vector_dot(dqi) + dj->vector_dot(dqj) + dJ->vector_dot(dm);
            auto sample = [&](double h) {
                auto a = shifted(qi, dqi, h), b = shifted(qj, dqj, h);
                C_DGESV_wrapper(shifted(J, dm, h), a);
                auto K = linalg::doublet(a, b, true, false);
                double value = 0.;
                auto A = last.parent;
                if (first.parent) {
                    SharedMatrix X;
                    value = reference_loss(project(K, first.parent), first.amplitudes, project(F0, first.parent),
                                           linalg::doublet(first.parent, f0, true, false), oi, oj,
                                           first.canonical_rotation->nrow(), true, diag, &X);
                    if (last.parent) {
                        auto ref = linalg::doublet(retained(first.eigenvectors, X->ncol()), first.canonical_rotation);
                        A = align_pnos(X, ref, first.parent);
                    }
                }
                if (last.parent)
                    value += reference_loss(project(K, A), last.amplitudes, project(F0, A),
                                            linalg::doublet(A, f0, true, false), oi, oj,
                                            last.canonical_rotation->nrow(), false, diag);
                return weight * value;
            };
            constexpr double h = 1.e-7;
            record_error(nested_df_error, analytic - (sample(h) - sample(-h)) / (2 * h));
        }
        scatter_pair_fit(p, dJ, di, dj, metric, density);
    }
    for (bool initial : {true, false}) {
        std::string name = initial ? "DLPNO INITIAL PNO TRUNCATION" : "DLPNO FINAL PNO TRUNCATION";
        set_scalar_variable(name + " ENERGY", energy[initial]);
        if (debug_) {
            set_scalar_variable(name + " DIRECTION ERROR", error[initial]);
            set_scalar_variable(name + " SINGLES DIRECTION ERROR", singles_error[initial]);
        }
    }
    if (debug_) set_scalar_variable("DLPNO PNO TRUNCATION DF DIRECTION ERROR", nested_df_error);
    std::string metric_name = "DLPNO PNO TRUNCATION AUXILIARY METRIC GRADIENT";
    std::string three_name = "DLPNO PNO TRUNCATION THREE-CENTER GRADIENT";
    set_array_variable(metric_name, contract_metric(metric, metric_name));
    set_array_variable(three_name, contract_three_center(density, three_name));
}
}  // namespace dlpno
}  // namespace psi
