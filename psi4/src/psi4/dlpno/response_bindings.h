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

#include "response_kernels.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <climits>
#include <stdexcept>

namespace psi {
namespace dlpno {

// Keep this binding independently compilable with the production BLAS kernel.
// Contiguous NumPy inputs are borrowed; strided inputs are packed once per
// operand, rather than once per product-rule term. No shared scratch or PSIO.
inline void export_response_kernels(pybind11::module_& module) {
    namespace py = pybind11;
    using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
    module.def("dlpno_mixed_gemm", [](std::vector<Array> a, std::vector<Array> b) {
        if (a.size() != 4 || b.size() != 4 || a[0].ndim() != 2 || b[0].ndim() != 2)
            throw std::invalid_argument("Mixed GEMM requires four matrices per operand.");
        const auto m = a[0].shape(0), k = a[0].shape(1), n = b[0].shape(1);
        if (m > INT_MAX || n > INT_MAX || k > INT_MAX)
            throw std::invalid_argument("Mixed GEMM dimensions exceed the BLAS integer range.");
        std::array<const double*, 4> ap, bp;
        std::array<double*, 4> cp;
        std::vector<Array> result;
        for (int i = 0; i < 4; ++i) {
            if (a[i].ndim() != 2 || b[i].ndim() != 2 || a[i].shape(0) != m || a[i].shape(1) != k ||
                b[i].shape(0) != k || b[i].shape(1) != n)
                throw std::invalid_argument("Inconsistent mixed GEMM block dimensions.");
            ap[i] = a[i].data();
            bp[i] = b[i].data();
            result.emplace_back(std::vector<py::ssize_t>{m, n});
            cp[i] = result.back().mutable_data();
        }
        {
            py::gil_scoped_release release;
            mixed_gemm(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), ap, bp, cp);
        }
        return result;
    }, py::arg("left"), py::arg("right"),
       "Shared-intermediate BLAS product rule for four value/u/v/uv matrices.");
}

}  // namespace dlpno
}  // namespace psi
