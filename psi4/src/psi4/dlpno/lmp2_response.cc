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

#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/matrix.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include <algorithm>
#include <cmath>

namespace psi {
namespace dlpno {

// The production right solve and the independent tangent solve both use this
// row. ordered[q] is the physical (k,j) or (i,k) amplitude in that orientation.
SharedMatrix DLPNOCCSD::lmp2_right_row(int p, const LMP2Graph& graph, const std::vector<SharedMatrix>& ordered,
                                       const SharedVector& eps, const SharedMatrix& seed) const {
    const auto [i, j] = ij_to_i_j_[p];
    auto r = seed->clone();
    if (!r->nrow()) return r;
    const double occ = (*F_lmo_)(i, i) + (*F_lmo_)(j, j);
    for (int a = 0; a < r->nrow(); ++a)
        for (int b = 0; b < r->ncol(); ++b) (*r)(a, b) += ((*eps)(a) + (*eps)(b)-occ) * (*ordered[p])(a, b);
    for (const auto& e : graph[p]) {
        auto t = linalg::triplet(e.overlap, ordered[e.source], e.overlap, false, false, true);
        r->axpy(-(*F_lmo_)(e.f1, e.f2), t);
    }
    return r;
}

std::vector<SharedMatrix> DLPNOCCSD::lmp2_apply(const std::vector<SharedMatrix>& x, bool transpose) const {
    const auto& state = lmp2_state_;
    std::vector<SharedMatrix> y(x.size());
    for (int p = 0; p < x.size(); ++p) y[p] = std::make_shared<Matrix>(x[p]->nrow(), x[p]->ncol());
    if (!transpose) {
        auto ordered = x;
        for (int p = 0; p < x.size(); ++p)
            if (ij_to_i_j_[p].first > ij_to_i_j_[p].second) ordered[p] = x[ij_to_ji_[p]]->transpose();
        for (int p = 0; p < x.size(); ++p)
            if (x[p]->nrow()) y[p] = lmp2_right_row(p, state.edges, ordered, state.eps[p], y[p]);
    } else {
        for (int p = 0; p < x.size(); ++p) {
            const auto [i, j] = ij_to_i_j_[p];
            if (!x[p]->nrow()) continue;
            const auto& eps = *state.eps[p];
            const double occ = (*F_lmo_)(i, i) + (*F_lmo_)(j, j);
            for (int a = 0; a < x[p]->nrow(); ++a)
                for (int b = 0; b < x[p]->ncol(); ++b) (*y[p])(a, b) += (eps(a) + eps(b) - occ) * (*x[p])(a, b);
            for (const auto& e : state.edges[p]) {
                auto z = linalg::triplet(e.overlap, x[p], e.overlap, true, false, false);
                int q = e.source;
                if (ij_to_i_j_[q].first > ij_to_i_j_[q].second) {
                    q = ij_to_ji_[q];
                    z = z->transpose();
                }
                y[q]->axpy(-(*F_lmo_)(e.f1, e.f2), z);
            }
        }
    }
    for (int p = 0; p < x.size(); ++p)
        if (ij_to_i_j_[p].first == ij_to_i_j_[p].second) y[p]->hermitivitize();
    return y;
}

// Solve source + A x = 0 (or source + A^T x = 0) on unique pairs. Null
// sources are represented by zero blocks; ji slots are empty even in DIIS.
std::vector<SharedMatrix> DLPNOCCSD::solve_lmp2_system(const std::vector<SharedMatrix>& source, bool transpose,
                                                       double& rms, int& iterations) {
    std::vector<SharedMatrix> x(source.size());
    for (int p = 0; p < x.size(); ++p) x[p] = std::make_shared<Matrix>(source[p]->nrow(), source[p]->ncol());
    bool has_source = false;
    for (const auto& b : source) {
        if (!b->nrow()) continue;
        const double norm = b->rms();
        if (!std::isfinite(norm)) throw PSIEXCEPTION("Nonfinite source in the LMP2 response equations.");
        has_source |= norm != 0.;
    }
    if (!has_source) {
        iterations = 0;
        rms = 0.;
        return x;
    }
    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), transpose ? "LMP2 adjoint" : "LMP2 tangent",
                     DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);
    const double tol = .01 * options_.get_double("R_CONVERGENCE");
    for (iterations = 0;; ++iterations) {
        auto r = lmp2_apply(x, transpose);
        rms = 0.;
        for (int p = 0; p < x.size(); ++p) {
            if (!x[p]->nrow()) continue;
            r[p]->add(source[p]);
            const double val = r[p]->rms();
            if (!std::isfinite(val)) throw PSIEXCEPTION("Nonfinite LMP2 response residual.");
            rms = std::max(rms, val);
        }
        if (rms < tol) return x;
        if (iterations >= options_.get_int("DLPNO_MAXITER"))
            throw PSIEXCEPTION("LMP2 response equations did not converge.");
        for (int p = 0; p < x.size(); ++p) {
            if (!x[p]->nrow()) continue;
            auto [i, j] = ij_to_i_j_[p];
            const auto& eps = *lmp2_state_.eps[p];
            double occ = (*F_lmo_)(i, i) + (*F_lmo_)(j, j);
            for (int a = 0; a < x[p]->nrow(); ++a)
                for (int b = 0; b < x[p]->ncol(); ++b) (*x[p])(a, b) -= (*r[p])(a, b) / (eps(a) + eps(b) - occ);
        }
        auto v = flatten_mats(x), err = flatten_mats(r);
        if (!iterations) {
            diis.set_error_vector_size(err);
            diis.set_vector_size(v);
        }
        diis.add_entry(err.get(), v.get());
        diis.extrapolate(v.get());
        copy_flat_mats(v, x);
    }
}

namespace {
double lmp2_probe_value(int a, int b, int salt) {
    return .03 * std::sin(.39 * (std::min(a, b) + 1) + .57 * (std::max(a, b) + 1) + .67 * salt);
}
SharedMatrix probe_block(const std::vector<int>& rows, const std::vector<int>& cols, int salt) {
    auto v = std::make_shared<Matrix>(rows.size(), cols.size());
    for (int a = 0; a < rows.size(); ++a)
        for (int b = 0; b < cols.size(); ++b) (*v)(a, b) = lmp2_probe_value(rows[a], cols[b], salt);
    return v;
}
SharedMatrix in_parent(const SharedMatrix& a, const SharedMatrix& x) {
    return linalg::triplet(x, a, x, false, false, true);
}
SharedMatrix operator_coefficient_bar(const SharedMatrix& a, const SharedMatrix& x, const SharedMatrix& g) {
    auto dx = linalg::triplet(a, x, g, false, false, true);
    dx->add(linalg::triplet(a, x, g, true, false, false));
    return dx;
}
double lmp2_dot(const std::vector<SharedMatrix>& x, const std::vector<SharedMatrix>& y) {
    double v = 0.;
    for (int p = 0; p < x.size(); ++p)
        if (x[p]->nrow()) v += x[p]->vector_dot(y[p]);
    return v;
}
void scatter_lmp2_block(std::vector<std::unordered_map<int, double>>& dest, const std::vector<int>& rows,
                        const std::vector<int>& cols, const SharedMatrix& g) {
    for (int a = 0; a < rows.size(); ++a)
        for (int b = 0; b < cols.size(); ++b) dest[rows[a]][cols[b]] += (*g)(a, b);
}
}  // namespace

void DLPNOCCSD_T::form_lmp2_response() {
    const auto& state = lmp2_state_;
    const int pairs = ij_to_i_j_.size();
    std::vector<SharedMatrix> source(pairs), xbar(pairs);
    size_t edges = 0, rectangular = 0, words = 0;
    for (int p = 0; p < pairs; ++p) {
        auto [i, j] = ij_to_i_j_[p];
        const int n = i <= j ? state.X[p]->ncol() : 0;
        source[p] = std::make_shared<Matrix>(n, n);
        if (ccsd_lmp2_projector_source_[p]) source[p]->copy(ccsd_lmp2_projector_source_[p]);
        if (i == j) source[p]->hermitivitize();
        if (n) xbar[p] = std::make_shared<Matrix>(state.X[p]->nrow(), n);
        words += size_t(n) * n;
        edges += state.edges[p].size();
        for (const auto& edge : state.edges[p]) rectangular += edge.overlap->nrow() != edge.overlap->ncol();
    }
    const size_t work = words * (12 + 2 * options_.get_int("DIIS_MAX_VECS"));
    if ((debug_ || lmp2_dot(source, source) > 0.) && toggle_memory_ &&
        (ccsd_response_words_ + work) * sizeof(double) > .9 * memory_)
        throw PSIEXCEPTION("Insufficient memory for the LMP2 adjoint.");
    double rms;
    int iterations;
    auto z = solve_lmp2_system(source, true, rms, iterations);
    set_scalar_variable("DLPNO LMP2 ADJOINT RESIDUAL", rms);
    set_scalar_variable("DLPNO LMP2 ADJOINT ITERATIONS", iterations);
    set_scalar_variable("DLPNO LMP2 RESPONSE EDGES", edges);
    set_scalar_variable("DLPNO LMP2 RECTANGULAR RESPONSE EDGES", rectangular);
    set_scalar_variable("DLPNO LMP2 ADJOINT NORM", std::sqrt(lmp2_dot(z, z)));
    outfile->Printf("    LMP2 adjoint: %d updates, max pair RMS %.3e, %zu retained couplings\n", iterations, rms,
                    edges);

    MetricShellBlocks metric(ribasis_->nshell());
    std::vector<ThreeCenterDensity> density(ribasis_->nbf());
    double analytic = 0.;
    // Each edge represents -F_lm S_pq T_q S_pq^T in target row p.
    for (int p = 0; p < pairs; ++p) {
        if (!z[p]->nrow() || z[p]->absmax() == 0.) continue;
        auto [i, j] = ij_to_i_j_[p];
        const auto& rows = lmopair_to_paos_[p];
        const double occ = -z[p]->vector_dot(state.T[p]);
        ccsd_foo_response_[i][i] += occ;
        ccsd_foo_response_[j][j] += occ;
        if (debug_) analytic += occ * (lmp2_probe_value(i, i, 1) + lmp2_probe_value(j, j, 1));
        for (const auto& e : state.edges[p]) {
            const int q = e.source;
            const int owner = ij_to_i_j_[q].first <= ij_to_i_j_[q].second ? q : ij_to_ji_[q];
            const auto& cols = lmopair_to_paos_[q];
            auto S = e.overlap;
            auto proj = linalg::triplet(S, state.T[q], S, false, false, true);
            double df = -z[p]->vector_dot(proj);
            ccsd_foo_response_[e.f1][e.f2] += df;
            auto ds = linalg::triplet(z[p], S, state.T[q], false, false, true);
            ds->add(linalg::triplet(z[p], S, state.T[q], true, false, false));
            ds->scale(-(*F_lmo_)(e.f1, e.f2));
            auto ds0 = linalg::triplet(state.X[p], ds, state.X[q], false, false, true);
            scatter_lmp2_block(ccsd_overlap_response_, rows, cols, ds0);
            auto raw = submatrix_rows_and_cols(*S_pao_, rows, cols);
            xbar[p]->add(linalg::triplet(raw, state.X[q], ds, false, false, true));
            xbar[owner]->add(linalg::triplet(raw, state.X[p], ds, true, false, false));
            if (debug_) analytic += df * lmp2_probe_value(e.f1, e.f2, 1) + ds0->vector_dot(probe_block(rows, cols, 3));
        }
    }
    lmp2_parent_.assign(pairs, {});
    lmp2_parent_bar_.assign(pairs, {});
    for (int p = 0; p < pairs; ++p) {
        if (!xbar[p] || (z[p]->absmax() == 0. && xbar[p]->absmax() == 0.)) continue;
        auto [i, j] = ij_to_i_j_[p];
        const auto& rows = lmopair_to_paos_[p];
        const auto& aux = lmopair_to_ribfs_[p];
        const auto& X = state.X[p];
        auto fit = pair_exchange_fit(p);
        auto F = submatrix_rows_and_cols(*F_pao_, rows, rows);
        // F T + T F, evaluated away from the semicanonical point as a full operator.
        auto df = linalg::doublet(z[p], state.T[p], false, true);
        df->add(linalg::doublet(state.T[p], z[p], true, false));
        df->hermitivitize();
        xbar[p]->add(operator_coefficient_bar(F, X, df));
        xbar[p]->add(operator_coefficient_bar(fit.K, X, z[p]));
        auto dF = in_parent(df, X), dK = in_parent(z[p], X);
        const auto& first = initial_pno_selection_[p];
        lmp2_parent_[p] = X;
        lmp2_parent_bar_[p] = xbar[p];
        if (first.parent) {
            const int n = first.parent->ncol();
            auto k = std::make_shared<Matrix>(n, n), f = k->clone();
            double ds = initial_pno_mp2_bar(p, xbar[p], k, f);
            ccsd_foo_response_[i][i] += ds;
            ccsd_foo_response_[j][j] += ds;
            if (debug_) analytic += ds * (lmp2_probe_value(i, i, 1) + lmp2_probe_value(j, j, 1));
            dK->add(in_parent(k, first.parent));
            dF->add(in_parent(f, first.parent));
            Dimension start(1), end(1);
            end.fill(first.canonical_rotation->nrow());
            auto kept = first.eigenvectors->get_block({start, first.eigenvectors->rowspi()}, {start, end});
            auto rot = linalg::doublet(kept, first.canonical_rotation);
            lmp2_parent_[p] = first.parent;
            auto da = linalg::doublet(xbar[p], rot, false, true);
            da->add(operator_coefficient_bar(fit.K, first.parent, k));
            da->add(operator_coefficient_bar(F, first.parent, f));
            lmp2_parent_bar_[p] = da;
        }
        scatter_lmp2_block(ccsd_fvv_response_, rows, rows, dF);
        auto di = linalg::doublet(fit.bj, dK, false, true), dj = linalg::doublet(fit.bi, dK);
        auto dJ = linalg::doublet(fit.bi, di, false, true);
        dJ->scale(-1.);
        dJ->hermitivitize();
        scatter_pair_fit(p, dJ, di, dj, metric, density);
        if (debug_)
            analytic += dF->vector_dot(probe_block(rows, rows, 2)) + di->vector_dot(probe_block(aux, rows, 7 + i)) +
                        dj->vector_dot(probe_block(aux, rows, 7 + j)) + dJ->vector_dot(probe_block(aux, aux, 5));
    }
    if (debug_) check_lmp2_adjoint(source, z, analytic);
    const std::string m = "DLPNO LMP2 RESPONSE AUXILIARY METRIC GRADIENT",
                      q = "DLPNO LMP2 RESPONSE THREE-CENTER GRADIENT";
    set_array_variable(m, contract_metric(metric, m));
    set_array_variable(q, contract_three_center(density, q));
    if (!options_.get_bool("DLPNO_KEEP_LMP2_RESPONSE")) {
        lmp2_state_ = {};
        lmp2_state_words_ = 0;
    }
}

DLPNOCCSD_T::PairBlocks DLPNOCCSD_T::lmp2_blocks(const std::string& block) const {
    if (lmp2_state_.X.empty()) throw PSIEXCEPTION("LMP2 response state was not retained.");
    if (block != "FRAME" && block != "AMPLITUDE" && block != "FINAL_FRAME" && block != "OCC_DOMAIN" && block != "CCSD_DOUBLES") throw PSIEXCEPTION("Invalid LMP2 response block.");
    PairBlocks result;
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        const auto ij = ij_to_i_j_[p];
        if (ij.first > ij.second) continue;
        if (block == "OCC_DOMAIN") {
            const auto& dom = lmopair_to_lmos_[p];
            auto ids = std::make_shared<Matrix>(dom.size(), 1);
            for (size_t k = 0; k < dom.size(); ++k) (*ids)(k, 0) = dom[k];
            result[ij] = ids;
        } else if (block == "CCSD_DOUBLES") {
            result[ij] = T_iajb_[p]->clone();
        } else if (block == "AMPLITUDE") {
            result[ij] = lmp2_state_.T[p]->clone();
        } else {
            auto N = submatrix_cols(*C_pao_, lmopair_to_paos_[p]);
            result[ij] = linalg::doublet(N, block == "FINAL_FRAME" ? X_pno_.at(p) : lmp2_state_.X[p]);
        }
    }
    return result;
}

DLPNOCCSD_T::PairBlocks DLPNOCCSD_T::lmp2_tangent(const PairBlocks& dy, const PairBlocks& dk,
                                                   SharedMatrix ds, SharedMatrix df, SharedMatrix dfoo) {
    const auto y = lmp2_blocks("FRAME");
    const auto S = reference_wavefunction_->S();
    const int n = S->nrow(), no = F_lmo_->nrow();
    auto valid = [](const SharedMatrix& a, int rows, int cols) {
        return a && a->nirrep() == 1 && a->nrow() == rows && a->ncol() == cols;
    };
    if (!valid(ds,n,n) || !valid(df,n,n) || !valid(dfoo,no,no) || dy.size()!=y.size() || dk.size()!=y.size())
        throw PSIEXCEPTION("Incompatible LMP2 nuclear response inputs.");
    // Cache both terms on the right leg of the differentiated overlap.
    PairBlocks sy, dsy;
    for (const auto& [ij, yp] : y) {
        if (!dy.count(ij) || !dk.count(ij) || !valid(dy.at(ij),n,yp->ncol()) ||
            !valid(dk.at(ij),yp->ncol(),yp->ncol())) throw PSIEXCEPTION("Invalid LMP2 pair tangent.");
        sy[ij] = linalg::doublet(S, yp);
        dsy[ij] = linalg::doublet(ds, yp);
        dsy[ij]->add(linalg::doublet(S, dy.at(ij)));
    }
    std::vector<SharedMatrix> source(ij_to_i_j_.size());
    for (int p = 0; p < source.size(); ++p) {
        const auto ij = ij_to_i_j_[p];
        const auto [i,j] = ij;
        if (i > j) { source[p] = std::make_shared<Matrix>(0,0); continue; }
        const auto yp = y.at(ij), dyp = dy.at(ij), t = lmp2_state_.T[p];
        auto fv = linalg::triplet(dyp,F_ao_,yp,true,false,false);
        fv->add(fv->transpose());
        fv->add(linalg::triplet(yp,df,yp,true,false,false));
        auto b = dk.at(ij)->clone();
        b->add(linalg::doublet(fv,t));
        b->add(linalg::doublet(t,fv));
        b->axpy(-((*dfoo)(i,i)+(*dfoo)(j,j)),t);
        for (const auto& edge : lmp2_state_.edges[p]) {
            auto [k,l] = ij_to_i_j_[edge.source];
            auto ijq = std::make_pair(std::min(k,l),std::max(k,l));
            auto ov = edge.overlap, tq = lmp2_state_.T[edge.source];
            auto dov = linalg::doublet(dyp,sy.at(ijq),true,false);
            dov->add(linalg::doublet(yp,dsy.at(ijq),true,false));
            b->axpy(-(*dfoo)(edge.f1,edge.f2), linalg::triplet(ov,tq,ov,false,false,true));
            const double f = -(*F_lmo_)(edge.f1,edge.f2);
            b->axpy(f,linalg::triplet(dov,tq,ov,false,false,true));
            b->axpy(f,linalg::triplet(ov,tq,dov,false,false,true));
        }
        if (i == j) b->hermitivitize();
        source[p] = b;
    }
    double rms;
    int iterations;
    auto dt = solve_lmp2_system(source,false,rms,iterations);
    set_scalar_variable("DLPNO LMP2 NUCLEAR TANGENT RESIDUAL",rms);
    set_scalar_variable("DLPNO LMP2 NUCLEAR TANGENT ITERATIONS",iterations);
    PairBlocks result;
    for (int p = 0; p < dt.size(); ++p)
        if (dt[p]->nrow()) result[ij_to_i_j_[p]] = dt[p];
    return result;
}

// Rebuild the unrelaxed residual at fixed amplitudes on a perturbed orbital
// and fitting problem. Keep the Fock screen and PNO ranks fixed.
std::vector<SharedMatrix> DLPNOCCSD_T::probe_lmp2_residual(double h) {
    const auto& state = lmp2_state_;
    std::vector<SharedMatrix> X(state.X.size()), r(X.size());
    for (int p = 0; p < X.size(); ++p) {
        auto [i, j] = ij_to_i_j_[p];
        if (i > j) {
            r[p] = std::make_shared<Matrix>(0, 0);
            continue;
        }
        const auto& paos = lmopair_to_paos_[p];
        const auto& aux = lmopair_to_ribfs_[p];
        auto fit = pair_exchange_fit(p);
        fit.J->axpy(h, probe_block(aux, aux, 5));
        fit.qi->axpy(h, probe_block(aux, paos, 7 + i));
        fit.qj->axpy(h, probe_block(aux, paos, 7 + j));
        C_DGESV_wrapper(fit.J, fit.qi);
        auto K = linalg::doublet(fit.qi, fit.qj, true, false);
        auto F = submatrix_rows_and_cols(*F_pao_, paos, paos);
        F->axpy(h, probe_block(paos, paos, 2));
        const double oi = (*F_lmo_)(i, i) + h * lmp2_probe_value(i, i, 1),
                     oj = (*F_lmo_)(j, j) + h * lmp2_probe_value(j, j, 1);
        X[p] = X[ij_to_ji_[p]] = transported_first_pno(p, K, F, oi, oj);
        auto f = linalg::triplet(X[p], F, X[p], true, false, false);
        r[p] = linalg::triplet(X[p], K, X[p], true, false, false);
        r[p]->add(linalg::doublet(f, state.T[p]));
        r[p]->add(linalg::doublet(state.T[p], f));
        r[p]->axpy(-(oi + oj), state.T[p]);
    }
    for (int p = 0; p < X.size(); ++p)
        for (const auto& e : state.edges[p]) {
            auto s = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[p], lmopair_to_paos_[e.source]);
            s->axpy(h, probe_block(lmopair_to_paos_[p], lmopair_to_paos_[e.source], 3));
            s = linalg::triplet(X[p], s, X[e.source], true, false, false);
            auto t = linalg::triplet(s, state.T[e.source], s, false, false, true);
            r[p]->axpy(-((*F_lmo_)(e.f1, e.f2) + h * lmp2_probe_value(e.f1, e.f2, 1)), t);
        }
    return r;
}

void DLPNOCCSD_T::check_lmp2_adjoint(const std::vector<SharedMatrix>& source, const std::vector<SharedMatrix>& z,
                                     double analytic) {
    double right = 0.;
    auto trial = [&](int seed) {
        std::vector<SharedMatrix> x;
        for (int p = 0; p < z.size(); ++p) {
            auto v = z[p]->clone();
            for (int a = 0; a < v->nrow(); ++a)
                for (int b = 0; b < v->ncol(); ++b)
                    (*v)(a, b) = std::sin(.37 * (p + 1) + seed * (.31 * (a + 1) + .59 * (b + 1))) / v->nrow();
            if (ij_to_i_j_[p].first == ij_to_i_j_[p].second) v->hermitivitize();
            x.push_back(v);
        }
        return x;
    };
    auto x = trial(2), y = trial(3);
    double err = std::abs(lmp2_dot(y, lmp2_apply(x, false)) - lmp2_dot(x, lmp2_apply(y, true)));
    set_scalar_variable("DLPNO LMP2 TRANSPOSE DOT ERROR", err);
    for (int p = 0; p < z.size(); ++p)
        if (z[p]->nrow()) {
            auto r = lmp2_right_row(p, lmp2_state_.edges, lmp2_state_.T, lmp2_state_.eps[p], lmp2_state_.K[p]);
            const double norm = r->rms();
            if (!std::isfinite(norm)) throw PSIEXCEPTION("Nonfinite saved LMP2 residual.");
            right = std::max(right, norm);
        }
    set_scalar_variable("DLPNO LMP2 SAVED RIGHT RESIDUAL", right);
    // A fixed-rank first PNO projector can have a narrow occupation gap.
    // Extrapolate the two centered residual differences before solving the tangent.
    auto difference = [&](double h) {
        auto plus = probe_lmp2_residual(h), minus = probe_lmp2_residual(-h);
        for (int p = 0; p < plus.size(); ++p)
            if (plus[p]->nrow()) {
                plus[p]->subtract(minus[p]);
                plus[p]->scale(.5 / h);
                if (ij_to_i_j_[p].first == ij_to_i_j_[p].second) plus[p]->hermitivitize();
            }
        return plus;
    };
    auto coarse = difference(1.e-6), plus = difference(5.e-7);
    outfile->Printf("    LMP2 fixed-geometry direction: analytic % .12e, centered % .12e and % .12e\n", analytic,
                    lmp2_dot(z, coarse), lmp2_dot(z, plus));
    for (int p = 0; p < plus.size(); ++p)
        if (plus[p]->nrow()) {
            plus[p]->scale(4. / 3.);
            plus[p]->axpy(-1. / 3., coarse[p]);
        }
    double rms;
    int iterations;
    auto dt = solve_lmp2_system(plus, false, rms, iterations);
    double tangent = lmp2_dot(source, dt);
    double response_err = std::abs(tangent - analytic);
    if (!std::isfinite(response_err) || !std::isfinite(err)) throw PSIEXCEPTION("Nonfinite LMP2 adjoint probe.");
    set_scalar_variable("DLPNO LMP2 RELAXED DIRECTION", analytic);
    set_scalar_variable("DLPNO LMP2 TANGENT DIRECTION ERROR", response_err);
}
}  // namespace dlpno
}  // namespace psi
