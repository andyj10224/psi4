/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2022 The Psi4 Developers.
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

#include <unordered_map>

#include "psi4/libmints/dimension.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/lib3index/dftensor.h"

namespace psi {

THCDecomposer::THCDecomposer(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary) {
    primary_ = primary;
    auxiliary_ = auxiliary;
    rank_ = auxiliary->nbf();

    nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif
}

void THCDecomposer::perform_hypercontraction() {

    // Build the integral factory for building DF Ints
    auto zero = BasisSet::zero_ao_basis_set();
    IntegralFactory factory = IntegralFactory(auxiliary_, zero, primary_, primary_);
    std::vector<TwoBodyAOInt> eris(nthread_);
    eris[0] = factory.eri();

    for (size_t thread = 0; thread < nthread_; thread++) {
        eris[thread] = eris[0].clone();
    }

    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();
    size_t naux_shell = auxiliary_->shell();
    size_t nshell = primary_->nshell();
    auto shell_pairs = eris[0].shell_pairs();
    size_t nshell_pair = shell_pairs.size();
    // Number of significant basis function pairs
    size_t nfunc_pair = 0;
    // Starting function pair index for each shell pair
    std::vector<size_t> shell_pair_start(nshell_pair);
    // Significant neighbors for each shell
    std::vector<std::vector<size_t>> shell_neighbors(nshell);
    // Backmap of the shell pair number to the shell pair index
    std::unordered_map<size_t, size_t> shell_pair_to_index;

    for (size_t idx = 0; idx < nshell_pair; idx++) {
        size_t U = shell_pairs[idx].first;
        size_t V = shell_pairs[idx].second;

        shell_pair_to_index[U * nshell + V] = idx;

        shell_neighbors[U].push_back(V);
        if (U != V) shell_neighbors[V].push_back(U);

        size_t num_u = primary_->shell(U).nfunction();
        size_t num_v = primary_->shell(V).nfunction();

        shell_pair_start[idx] = nfunc_pair;
        nfunc_pair += num_u * num_v;
    }

    // Number of basis functions significant to a shell
    std::vector<size_t> num_sig_func_per_shell(nshell);

#pragma omp parallel for
    for (size_t U = 0; U < nshell; U++) {
        auto partners = shell_partners[U];
        size_t nfunc = 0;

        for (const size_t V : partners) {
            nfunc += primary_->shell(V).nfunction();
        }
        num_sig_func_per_shell[U] = nfunc;
    }

    // Get the fitting metric
    FittingMetric J_metric_obj(auxiliary_, true);
    J_metric_obj.form_fitting_metric();
    SharedMatrix J_metric = J_metric_obj.get_metric();
    J_metric->power(-0.5, 1.0e-14);

    Qpq = std::make_shared<Matrix>(naux, nfunc_pair);
    double** Qpqp = Qpq->pointer();

    // Build the set of ERIs Qpq
#pragma omp parallel for
    for (size_t P = 0; P < naux_shell; P++) {

        int thread = 0
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif _OPENMP

        size_t Pstart = auxiliary_->shell(P).start();
        size_t num_p = auxiliary_->shell(P).nfunction();

        for (size_t UV = 0; UV < nshell_pair; UV++) {
            size_t U = shell_pairs[idx].first;
            size_t V = shell_pairs[idx].second;

            size_t UVstart = shell_pair_start[UV];

            size_t Ustart = primary_->shell(U).start();
            size_t num_u = primary_->shell(U).nfunction();

            size_t Vstart = primary_->shell(V).start();
            size_t num_v = primary_->shell(V).nfunction();

            eris[thread]->compute_shell(P, 0, U, V);
            double *buffer = const_cast<double *>(eris[thread]->buffer());

            for (size_t dp = 0; dp < num_p; dp++) {
                for (size_t duv = 0; duv < num_u * num_v; duv++) {
                    Qpqp[Pstart + dp][UVstart + duv] = buffer;
                    (buffer)++;
                }
            }
        }
    }

    // B_Qpq = (Q|P)^(-1/2) * (P|uv)
    Qpq = linalg::doublet(J_metric, Qpq, false, false);

    // Perform the SVD guess of the Q index
    auto svd_temp_buffers = Qpq->svd_temps();
    SharedMatrix U = std::get<0>(svd_temp_buffers);
    SharedMatrix S = std::get<1>(svd_temp_buffers);
    SharedMatrix V = std::get<2>(svd_temp_buffers);
    Qpq->svd(U, S, V);

    Slive row_slice(Dimension(std::vector<int>{0}), Dimension(std::vector<int>{naux}));
    Slice col_slice(Dimension(std::vector<int>{0}), Dimension(std::vector<int>{rank_}));

    Z_ = U->get_block(row_slice, col_slice);

    // Perform the SVD guess of the primary index
    for (size_t U = 0; U < nshell; U++) {
        auto partners = shell_neighbors[U];
        SharedMatrix QV = std::make_shared<Matrix>(naux, num_sig_func_per_shell[U]);

        for (size_t Q = 0; Q < naux_shell; Q++) {
            
            for (const size_t V : partners) {
                size_t UV = (U >= V) ? U * nshell + V : V * shell + U;
                size_t UVidx = shell_pair_to_index[UV];

            }
        }
    }

}

}