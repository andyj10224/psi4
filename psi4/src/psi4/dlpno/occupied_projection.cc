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
#include "psi4/libmints/matrix.h"
#include <cmath>

namespace psi {
namespace dlpno {
namespace {
using Rows = std::vector<std::unordered_map<int, double>>;
// Accumulate W X or W^T X. Only retained entries of W are visited.
SharedMatrix sparse_action(const Rows& w, int cols, const SharedMatrix& x, bool transpose = false) {
    auto y = std::make_shared<Matrix>(transpose ? cols : w.size(), x->ncol());
    for (int i = 0; i < w.size(); ++i)
        for (const auto& [j, g] : w[i]) {
            const int dest = transpose ? j : i, src = transpose ? i : j;
            for (int k = 0; k < x->ncol(); ++k) (*y)(dest, k) += g * (*x)(src, k);
        }
    return y;
}
SharedMatrix both_actions(const Rows& w, const SharedMatrix& x) {
    auto y = sparse_action(w, w.size(), x);
    y->add(sparse_action(w, w.size(), x, true));
    return y;
}
}  // namespace

// The independent input blocks are the active LMO columns, frozen occupied
// columns, and symmetric AO F and S. Keep the Jacobian in factored form;
// evaluating one panel never constructs an AO-by-AO derivative matrix.
SharedMatrix DLPNOCCSD_T::orbital_response_action(const std::string& block, const SharedMatrix& rhs) const {
    if (!gradient_model_ || !gradient_frozen_occ_ || raw_pao_response_.overlap.empty())
        throw PSIEXCEPTION("The RHF DLPNO occupied projection response has not been built.");
    const auto& L = C_lmo_;
    const auto& core = gradient_frozen_occ_;
    const auto& S = reference_wavefunction_->S();
    const auto& F = F_ao_;
    const auto& r = raw_pao_response_;
    const int n = L->nrow();
    const bool active = block == "ACTIVE", frozen = block == "FROZEN";
    if (!active && !frozen && block != "FOCK" && block != "OVERLAP")
        throw PSIEXCEPTION("Unknown DLPNO orbital response block.");
    int rows = active ? L->ncol() : (frozen ? core->ncol() : n);
    if (!rhs || rhs->nirrep() != 1 || rhs->nrow() != rows)
        throw PSIEXCEPTION("Incompatible column panel for the DLPNO orbital response block.");
    if (!rows || !rhs->ncol()) return std::make_shared<Matrix>(n, rhs->ncol());
    auto occupied = [&](const SharedMatrix& x) {
        auto v = linalg::doublet(L, linalg::doublet(L, x, true, false));
        if (core->ncol()) v->add(linalg::doublet(core, linalg::doublet(core, x, true, false)));
        return v;
    };
    auto B = [&](const SharedMatrix& x, bool trans = false) {
        auto v = x->clone();
        v->subtract(trans ? linalg::doublet(S, occupied(x)) : occupied(linalg::doublet(S, x)));
        return v;
    };
    // G = dL/dB from the DF coefficients and three raw one-body blocks.
    auto G = [&](const SharedMatrix& x, bool trans = false) {
        auto v = sparse_action(r.coefficients, n, x, trans);
        if (trans) {
            auto fx = linalg::doublet(F, x);
            v->add(sparse_action(r.fov, n, linalg::doublet(L, fx, true, false), true));
            v->add(both_actions(r.fvv, B(fx, true)));
            v->add(both_actions(r.overlap, B(linalg::doublet(S, x), true)));
        } else {
            auto f = linalg::doublet(L, sparse_action(r.fov, n, x));
            f->add(B(both_actions(r.fvv, x)));
            v->add(linalg::doublet(F, f));
            v->add(linalg::doublet(S, B(both_actions(r.overlap, x))));
        }
        return v;
    };
    if (active || frozen) {
        auto c = linalg::doublet(active ? L : core, rhs);
        auto v = G(linalg::doublet(S, c));
        v->add(linalg::doublet(S, G(c, true)));
        v->scale(-1.);
        if (active) {
            auto f = linalg::doublet(L, both_actions(ccsd_foo_response_, rhs));
            f->add(B(sparse_action(r.fov, n, rhs, true)));
            v->add(linalg::doublet(F, f));
            v->add(linalg::doublet(S, sparse_action(df_sclmo_bar_, L->ncol(), rhs)));
            if (screened_lmo_bar_) v->add(linalg::doublet(screened_lmo_bar_, rhs));
        }
        return v;
    }
    auto bx = B(rhs, true), lx = linalg::doublet(L, rhs, true, false);
    SharedMatrix v;
    if (block == "FOCK") {
        v = linalg::doublet(L, both_actions(ccsd_foo_response_, lx));
        v->add(linalg::doublet(L, sparse_action(r.fov, n, bx)));
        v->add(B(sparse_action(r.fov, n, lx, true)));
        v->add(B(both_actions(r.fvv, bx)));
    } else {
        v = B(both_actions(r.overlap, bx));
        v->add(both_actions(df_refit_overlap_bar_, rhs));
        v->add(sparse_action(df_sclmo_bar_, L->ncol(), lx));
        v->add(linalg::doublet(L, sparse_action(df_sclmo_bar_, L->ncol(), rhs, true)));
        v->subtract(occupied(G(rhs)));
        v->subtract(G(occupied(rhs), true));
    }
    v->scale(.5);
    return v;
}

void DLPNOCCSD_T::check_occupied_projection() {
    const auto S = reference_wavefunction_->S(), F = F_ao_, L = C_lmo_, core = gradient_frozen_occ_;
    const int n = L->nrow();
    auto column = [](int size, int seed) {
        auto x = std::make_shared<Matrix>(size, 1);
        for (int i = 0; i < size; ++i) (*x)(i, 0) = std::sin(.71 * (1 + i) + seed) / std::sqrt(1. + size);
        return x;
    };
    auto dot = [](const Rows& w, const SharedMatrix& a) {
        double val = 0.;
        for (int i = 0; i < w.size(); ++i)
            for (const auto& [j, g] : w[i]) val += g * (*a)(i, j);
        return val;
    };
    for (const std::string block : {"ACTIVE", "FROZEN", "FOCK", "OVERLAP"}) {
        const bool active = block == "ACTIVE", frozen = block == "FROZEN";
        const int cols = active ? L->ncol() : (frozen ? core->ncol() : n);
        double err = 0., analytic = 0.;
        if (cols) {
            auto u = column(n, 2), v = column(cols, 5);
            auto d = linalg::doublet(u, v, false, true);
            if (!active && !frozen) d->hermitivitize();
            analytic = u->vector_dot(orbital_response_action(block, v));
            auto sample = [&](double h) {
                auto l = L->clone(), c = core->clone(), s = S->clone(), f = F->clone();
                (active ? l : frozen ? c : block == "FOCK" ? f : s)->axpy(h, d);
                auto B = std::make_shared<Matrix>(n, n);
                B->identity();
                B->subtract(linalg::triplet(l, l, s, false, true, false));
                if (c->ncol()) B->subtract(linalg::triplet(c, c, s, false, true, false));
                const auto& r = raw_pao_response_;
                return dot(ccsd_foo_response_, linalg::triplet(l, f, l, true, false, false)) +
                       dot(r.fov, linalg::triplet(l, f, B, true, false, false)) +
                       dot(r.fvv, linalg::triplet(B, f, B, true, false, false)) +
                       dot(r.overlap, linalg::triplet(B, s, B, true, false, false)) + dot(r.coefficients, B) +
                       dot(df_refit_overlap_bar_, s) + dot(df_sclmo_bar_, linalg::doublet(s, l)) +
                       (screened_lmo_bar_ ? screened_lmo_bar_->vector_dot(l) : 0.);
            };
            constexpr double h = 1.e-3;
            err = std::abs(analytic - (8. * (sample(h) - sample(-h)) - (sample(2*h) - sample(-2*h))) / (12 * h));
            if (!std::isfinite(err)) throw PSIEXCEPTION("Nonfinite occupied-space projection probe.");
        }
        set_scalar_variable("DLPNO " + block + " PROJECTOR DIRECTION ERROR", err);
        set_scalar_variable("DLPNO " + block + " PROJECTOR DIRECTION", analytic);
    }
}
}  // namespace dlpno
}  // namespace psi
