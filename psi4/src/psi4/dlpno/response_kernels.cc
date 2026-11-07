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

#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "response_kernels.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace psi {
namespace dlpno {

void mixed_gemm(int m, int n, int k, const std::array<const double*, 4>& a,
                const std::array<const double*, 4>& b, const std::array<double*, 4>& out) {
    if (m < 0 || n < 0 || k < 0) throw std::invalid_argument("Negative mixed GEMM dimension.");
    std::array<bool, 4> az{}, bz{};
    auto zero = [](const double* p, size_t count) {
        bool result = true;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(p[i])) throw std::invalid_argument("Nonfinite mixed GEMM input.");
            result = result && p[i] == 0.;
        }
        return result;
    };
    for (int i = 0; i < 4; ++i) {
        az[i] = zero(a[i], size_t(m)*k);
        bz[i] = zero(b[i], size_t(k)*n);
        std::fill_n(out[i], size_t(m)*n, 0.);
    }
    if (!m || !n || !k) return;
    // Each output owns its accumulation. BLAS owns contraction parallelism;
    // no outer OpenMP region multiplies the configured BLAS thread count.
    for (int mask = 0; mask < 4; ++mask)
        for (int i = 0; i < 4; ++i) {
            if ((i & mask) != i) continue;
            const int j = mask ^ i;
            if (az[i] || bz[j]) continue;
            C_DGEMM('N', 'N', m, n, k, 1., const_cast<double*>(a[i]), k,
                    const_cast<double*>(b[j]), n, 1., out[mask], n);
        }
}

// Q and R contain (Q|mn) and J^{-1}(Q|mn), flattened as naux by nao^2.
// Components use the bit masks 0, u=1, v=2, uv=3. Disjoint derivative masks
// enumerate the product rule without assuming symmetric Q, R or D. One block
// requests just the reference action; four blocks request all mixed terms.
std::vector<SharedMatrix> df_fock_response(const std::vector<SharedMatrix>& q,
        const std::vector<SharedMatrix>& fitted, const std::vector<SharedMatrix>& density) {
    const int count = q.size();
    if ((count != 1 && count != 4) || fitted.size() != q.size() || density.size() != q.size() ||
        !q[0] || !density[0] || q[0]->nirrep() != 1 || density[0]->nirrep() != 1)
        throw PSIEXCEPTION("DF Fock response requires one or four C1 blocks per operand.");
    const int n = density[0]->nrow(), nq = q[0]->nrow();
    if (n <= 0 || nq <= 0 || n > INT_MAX / n || nq > INT_MAX / n)
        throw PSIEXCEPTION("Invalid or oversized DF Fock response dimensions.");
    const int nn = n*n, qn = nq*n;
    for (int i = 0; i < count; ++i) {
        for (const auto& blocks : {&q, &fitted, &density}) {
            const auto& m = (*blocks)[i];
            const bool d = blocks == &density;
            if (!m || m->nirrep() != 1 || m->nrow() != (d ? n : nq) || m->ncol() != (d ? n : nn))
                throw PSIEXCEPTION("Inconsistent DF Fock response block dimensions.");
        }
    }
    [[maybe_unused]] const int threads = std::max(1, std::min(nq, Process::environment.get_n_threads()));
    auto pack = [&](const double* src, const SharedMatrix& dest) {
        // Distinct auxiliary slices write disjoint destinations. No numerical
        // reductions occur in OpenMP; BLAS calls remain outside the region.
        auto dst = dest->pointer()[0];
#pragma omp parallel for schedule(static) num_threads(threads) if(size_t(nq)*nn >= 32768)
        for (int a = 0; a < nq; ++a)
            for (int m = 0; m < n; ++m)
                std::copy_n(src + (size_t(a)*n+m)*n, n, dst + size_t(m)*qn + a*n);
    };
    std::vector<SharedMatrix> result(count), packed(count);
    for (int i = 0; i < count; ++i) {
        result[i] = std::make_shared<Matrix>(n,n);
        packed[i] = std::make_shared<Matrix>(n,qn);
        pack(fitted[i]->pointer()[0], packed[i]);
    }
    auto work = std::make_shared<Matrix>(qn,n);
    auto left = std::make_shared<Matrix>(n,qn);
    std::vector<double> rho(nq);
    // Reuse Q_i D_k for every allowed fitted component j. At most nine
    // products are needed for the sixteen terms of the mixed trilinear map.
    for (int i = 0; i < count; ++i) {
        for (int k = 0; k < count; ++k) {
            if (i & k) continue;
            C_DGEMM('N','N',qn,n,n,1.,q[i]->pointer()[0],n,density[k]->pointer()[0],n,
                    0.,work->pointer()[0],n);
            pack(work->pointer()[0], left);
            for (int j = 0; j < count; ++j) {
                if ((i & j) || (j & k)) continue;
                auto& out = result[i | j | k];
                C_DGEMV('N',nq,nn,1.,fitted[j]->pointer()[0],nn,density[k]->pointer()[0],1,0.,rho.data(),1);
                C_DGEMV('T',nq,nn,2.,q[i]->pointer()[0],nn,rho.data(),1,1.,out->pointer()[0],1);
                out->gemm(false,true,-1.,left,packed[j],1.);
            }
        }
    }
    return result;
}

}  // namespace dlpno
}  // namespace psi
