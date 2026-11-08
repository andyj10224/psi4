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

#pragma once

#include "dlpno.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>

namespace psi {
namespace dlpno {
namespace ccsd_detail {

inline std::vector<SharedMatrix> zero_blocks(const std::vector<SharedMatrix>& shapes) {
    std::vector<SharedMatrix> blocks(shapes.size());
    for (size_t p = 0; p < shapes.size(); ++p)
        if (shapes[p]) blocks[p] = std::make_shared<Matrix>(shapes[p]->nrow(), shapes[p]->ncol());
    return blocks;
}

inline SharedMatrix row_as_column(const SharedMatrix& m, int row) {
    auto v = submatrix_rows(*m, {row});
    v->reshape(m->ncol(), 1);
    return v;
}

inline void add_row(const SharedMatrix& m, int row, const SharedMatrix& v, double factor = 1.0) {
    if (m->ncol()) C_DAXPY(m->ncol(), factor, v->get_pointer(), 1, &(*m)(row, 0), 1);
}

}  // namespace ccsd_detail

// Covectors of the intermediates in the implemented t1-transformed residual.
// The PNOs and overlap maps are fixed for this amplitude transpose. In
// particular, never insert another PNO projector when reversing a product.
struct DLPNOCCSD::CCAdjoints {
    const std::vector<SharedMatrix>& z1;
    const std::vector<SharedMatrix>& z2;
    std::vector<SharedMatrix> t1, t2, u, tn, qa, qk, beta, gamma, delta, fab, fkc, fkc_bar;
    std::vector<double> fkj, fij_bar;

    void zero() {
        for (auto* blocks : {&t1, &t2, &u, &tn, &qa, &qk, &beta, &gamma, &delta, &fab, &fkc, &fkc_bar})
            for (auto& block : *blocks) if (block) block->zero();
        std::fill(fkj.begin(), fkj.end(), 0.);
        std::fill(fij_bar.begin(), fij_bar.end(), 0.);
    }

    CCAdjoints(const DLPNOCCSD& w, const std::vector<SharedMatrix>& z1, const std::vector<SharedMatrix>& z2)
        : z1(z1),
          z2(z2),
          t1(ccsd_detail::zero_blocks(w.T_ia_)),
          t2(ccsd_detail::zero_blocks(w.T_iajb_)),
          u(ccsd_detail::zero_blocks(w.Tt_iajb_)),
          tn(ccsd_detail::zero_blocks(w.T_n_ij_)),
          qa(ccsd_detail::zero_blocks(w.i_Qa_t1_)),
          qk(ccsd_detail::zero_blocks(w.i_Qk_t1_)),
          beta(ccsd_detail::zero_blocks(w.beta_)),
          gamma(ccsd_detail::zero_blocks(w.gamma_)),
          delta(ccsd_detail::zero_blocks(w.delta_)),
          fab(ccsd_detail::zero_blocks(w.Fab_)),
          fkc(ccsd_detail::zero_blocks(w.Fkc_)),
          fkc_bar(ccsd_detail::zero_blocks(w.Fkc_bar_)),
          fkj(w.ij_to_i_j_.size()),
          fij_bar(w.ij_to_i_j_.size()) {}
};

}  // namespace dlpno
}  // namespace psi
