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

#pragma once

#include "psi4/libmints/matrix.h"

namespace psi {
namespace dlpno {
namespace triples_detail {
// Shared by the production energy, adjoint and Hessian operators.
inline SharedMatrix transform_triples(SharedMatrix A, SharedMatrix X, int dim_old, int dim_new) {
    /*
    Performs the operation A'[i,j,k] = A[I,J,K] * X[i,I] * X[j,J] * X[k,K] for cube 3d tensors
    */

    SharedMatrix A_new = linalg::doublet(X, A, false, false);
    A_new->reshape(dim_new * dim_old, dim_old);
    A_new = linalg::doublet(A_new, X, false, true);

    SharedMatrix A_T = std::make_shared<Matrix>(dim_new * dim_new, dim_old);
    // Explicit strides avoid an integer divide/modulo for every tensor element.
    for (int a = 0; a < dim_new; ++a)
        for (int b = 0; b < dim_new; ++b) {
            auto* dest = A_T->pointer()[a * dim_new + b];
            for (int c = 0; c < dim_old; ++c)
                dest[c] = A_new->pointer()[a * dim_old + c][b];
        }
    A_T = linalg::doublet(A_T, X, false, true);

    A_new = std::make_shared<Matrix>(dim_new, dim_new * dim_new);

    for (int a = 0; a < dim_new; ++a)
        for (int b = 0; b < dim_new; ++b) {
            auto* dest = A_new->pointer()[a] + b * dim_new;
            for (int c = 0; c < dim_new; ++c)
                dest[c] = A_T->pointer()[a * dim_new + c][b];
        }

    return A_new;
}

}  // namespace triples_detail
}  // namespace dlpno
}  // namespace psi
