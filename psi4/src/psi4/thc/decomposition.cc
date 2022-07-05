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

void THCDecomposer::prep_sparsity() {

    timer_on("THCDecomposer: Prep Sparsity");

    // Build the integral factory for building DF Ints
    auto zero = BasisSet::zero_ao_basis_set();
    IntegralFactory factory(auxiliary_, zero, primary_, primary_);
    eris.resize(nthread_);
    eris[0] = factory.eri();
    for (size_t thread = 0; thread < nthread_; thread++) {
        eris[thread] = eris[0].clone();
    }

    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();
    size_t naux_shell = auxiliary_->nshell();
    size_t nshell = primary_->nshell();

    shell_pairs_ = eris[0].shell_pairs();
    size_t nshell_pair = shell_pairs.size();


    nfunc_pair_ = 0;
    shell_pair_start_.resize(nshell_pair);
    shell_neighbors_.resize(nshell);

    for (size_t idx = 0; idx < nshell_pair; idx++) {
        size_t U = shell_pairs_[idx].first;
        size_t V = shell_pairs_[idx].second;

        shell_pair_to_index_[U * nshell + V] = idx;

        shell_neighbors_[U].push_back(V);
        if (U != V) shell_neighbors_[V].push_back(U);

        size_t num_u = primary_->shell(U).nfunction();
        size_t num_v = primary_->shell(V).nfunction();

        shell_pair_start_[idx] = nfunc_pair_;
        nfunc_pair_ += num_u * num_v;
    }

    num_sig_func_per_shell_.resize(nshell);

#pragma omp parallel for
    for (size_t U = 0; U < nshell; U++) {
        auto partners = shell_partners_[U];
        size_t nfunc = 0;

        for (const size_t V : partners) {
            nfunc += primary_->shell(V).nfunction();
        }
        num_sig_func_per_shell_[U] = nfunc;
    }

    timer_on("THCDecomposer: Prep Sparsity");
}

void THCDecomposer::build_eri() {

    timer_on("THCDecomposer: Build ERI");

    size_t naux_shell = auxiliary_->nshell();
    size_t nshell = primary_->nshell();

    shell_pairs_ = eris[0].shell_pairs();
    size_t nshell_pair = shell_pairs.size();

    // Get the fitting metric
    FittingMetric J_metric_obj(auxiliary_, true);
    J_metric_obj.form_fitting_metric();
    SharedMatrix J_metric = J_metric_obj.get_metric();
    J_metric->power(-0.5, 1.0e-14);

    B_Qpq_ = std::make_shared<Matrix>(naux, nfunc_pair);
    double** B_Qpqp = B_Qpq_->pointer();

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
            size_t U = shell_pairs_[idx].first;
            size_t V = shell_pairs_[idx].second;

            size_t UVstart = shell_pair_start_[UV];

            size_t Ustart = primary_->shell(U).start();
            size_t num_u = primary_->shell(U).nfunction();

            size_t Vstart = primary_->shell(V).start();
            size_t num_v = primary_->shell(V).nfunction();

            eris_[thread]->compute_shell(P, 0, U, V);
            double *buffer = const_cast<double *>(eris_[thread]->buffer());

            for (size_t dpuv = 0; dpuv < num_p * num_u * num_v; dpuv++) {
                size_t dp = dpuv / (num_u * num_v);
                size_t duv = dpuv % (num_u * num_v);
                Qpqp[Pstart + dp][UVstart + duv] = buffer;
                (buffer)++;
            }
        }
    }

    // B_Qpq = (Q|P)^(-1/2) * (P|uv)
    B_Qpq_ = linalg::doublet(J_metric, B_Qpq_, false, false);

    timer_off("THCDecomposer: Build ERI");
}

void THCDecomposer::perform_hypercontraction() {

    size_t naux_shell = auxiliary_->nshell();
    size_t nshell = primary_->nshell();

    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();

    // Perform the SVD guess of the Q index
    auto svd_temp_buffers = B_Qpq_->svd_temps();
    SharedMatrix U = std::get<0>(svd_temp_buffers);
    SharedVector S = std::get<1>(svd_temp_buffers);
    SharedMatrix V = std::get<2>(svd_temp_buffers);
    B_Qpq_->svd(U, S, V);

    Slice row_slice(Dimension(std::vector<int>{0}), Dimension(std::vector<int>{naux}));
    Slice col_slice(Dimension(std::vector<int>{0}), Dimension(std::vector<int>{rank_}));

    Z_ = U->get_block(row_slice, col_slice);

    // Perform the SVD guess of the primary index (only needs to be done once)
    x_pI_ = std::make_shared<Matrix>(nbf, rank_);
    x_qI_ = std::make_shared<Matrix>(nbf, rank_);

    double** B_Qpqp = B_Qpq_->pointer();

    for (size_t u = 0; u < nbf; u++) {
        size_t Ushell = primary_->function_to_shell(u);
        size_t Ustart = primary_->shell(Ushell).start();
        size_t num_u = primary_->shell(Ushell).nfunction();

        SharedMatrix QV = std::make_shared<Matrix>(naux, num_sig_func_per_shell_[Ushell]);
        double** QVp = QV->pointer()[0];

        for (size_t q = 0; q < naux; q++) {
            size_t voff = 0;
            for (size_t Vshell : shell_partners_[Ushell]) {
                size_t num_v = primary_->shell(Vshell).nfunction();

                size_t first = (Ushell >= Vshell) ? Ushell : Vshell;
                size_t second = (Ushell >= Vshell) ? Vshell : Ushell;

                size_t first_start = primary_->shell(first).start();
                size_t first_nfunction = primary_->shell(first).nfunction();
                
                size_t second_start = primary_->shell(second).start();
                size_t second_nfunction = primary_->shell(second).nfunction();
                
                size_t UV = first * nshell + second;

                size_t UVoffset = shell_pair_start_[UV];

                for (size_t dv = 0; dv < num_v; dv++) {
                    if (Ushell >= Vshell) {
                        B_Qpqp[q][v_off+dv] = B_Qpqp[q][UVoffset + (u - Ustart) * num_v + dv];
                    } else {
                        B_Qpqp[q][v_off+dv] = B_Qpqp[q][UVoffset + (dv) * num_u + (u - Ustart)];
                    }
                }
                v_off += num_v;
            }
        }
        // Perform the SVD guess of the Q index
        auto QV_svd_buffers = QV->svd_temps();
        SharedMatrix U = std::get<0>(QV_svd_buffers);
        SharedVector S = std::get<1>(QV_svd_buffers);
        SharedMatrix V = std::get<2>(QV_svd_buffers);
        QV->svd(U, S, V);

        Slice row_slice(Dimension(std::vector<int>{0}), Dimension(std::vector<int>{naux}));
        Slice col_slice(Dimension(std::vector<int>{0}), Dimension(std::vector<int>{rank_}));

        Z_ = U->get_block(row_slice, col_slice);
    }

    for (size_t U = 0; U < nshell; U++) {
        auto partners = shell_neighbors[U];
        size_t Ustart = primary_->shell(U).start();
        size_t num_u = primary_->shell(U).nfunction();

        for (size_t du = 0; du < num_u; du++) {
            SharedMatrix QV = std::make_shared<Matrix>(naux, num_sig_func_per_shell[U]);
            double **QVp = QV->pointer();

            for (size_t Q = 0; Q < naux_shell; Q++) {
                size_t Qstart = auxiliary_->shell(Q).start();
                size_t num_q = auxiliary_->shell(Q).nfunction();

                for (const size_t V : partners) {
                    size_t UV = (U >= V) ? U * nshell + V : V * shell + U;
                    size_t UVidx = shell_pair_to_index[UV];
                    size_t offset = shell_pair_start_[UVidx];

                    for (size_t dq = 0; dq < num_q; dq++) {
                        for (size_t dv = 0; dv < num_v; dv++) {
                            QVp[Qstart+dq][]
                        }
                    }
                }
            }
        }
    }

}

}