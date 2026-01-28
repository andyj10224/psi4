/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
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

#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/aiohandler.h"
#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/integral.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/lib3index/dfhelper.h"

#include "jk.h"

#include <sstream>
#include "psi4/libpsi4util/PsiOutStream.h"
#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

using namespace psi;

namespace psi {

EinsumsDFJK::EinsumsDFJK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
    Options& options) : JK(primary), auxiliary_(auxiliary), options_(options) { common_init(); }

EinsumsDFJK::~EinsumsDFJK() {}

void EinsumsDFJK::common_init() {}

void EinsumsDFJK::build_eri_computers() {

    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();
    size_t nshell = primary_->nshell();
    size_t naux_shell = auxiliary_->nshell();

    // => Make TwoBodyAOInt objects to compute ERIs <= //
    auto zero_ao = BasisSet::zero_ao_basis_set(); // Single AO basis set (gaussian with zeta = 0)
    auto factory = std::make_shared<IntegralFactory>(auxiliary_, zero_ao, primary_, primary_);

    // Make integral objects for every thread to parallelize computation of ints
    eri_computers_.resize(df_ints_num_threads_);
    eri_computers_[0] = std::shared_ptr<TwoBodyAOInt>(factory->eri());

    for (size_t thread = 1; thread < df_ints_num_threads_; ++thread) {
        // something like this instead? eri_computers_["4-Center"].front()->clone()
        eri_computers_[thread] = std::shared_ptr<TwoBodyAOInt>(eri_computers_[0]->clone());
    }

}

void EinsumsDFJK::print_header() const {
    if (print_) {
        outfile->Printf("  ==> EinsumsDFJK: Arbitrary Data Type Density-Fitted J/K Matrices <==\n\n");

        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:         %11s\n", (do_wK_ ? "Yes" : "No"));
        if (do_wK_) outfile->Printf("    Omega:             %11.3E\n", omega_);
        outfile->Printf("    Integrals threads: %11d\n", df_ints_num_threads_);
        outfile->Printf("    Memory [MiB]:      %11ld\n", (memory_ *8L) / (1024L * 1024L));
        outfile->Printf("    Screening Cutoff:  %11.0E\n", cutoff_);
        outfile->Printf("\n");
    }
}

size_t EinsumsDFJK::memory_estimate() {

    timer_on("Memory Estimate");

    // Significant number of shell pairs (that survive Schwarz/CSAM screening)
    size_t nshell_pair = eri_computers_[0]->shell_pairs().size();
    size_t nshell_triplet = auxiliary_->nshell() * nshell_pair;

    size_t num_doubles = 0L;

    // shells (P | MU NU) (AUX, AO, AO)
    // The shell pairs MN are restricted indexing M <= N, so this reduces half the memory
#pragma omp parallel for reduction(+ : num_doubles)
    for (int PMN = 0; PMN < nshell_triplet; ++PMN) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        
        size_t P = PMN / nshell_pair;
        size_t MN = PMN % nshell_pair;
        auto MN_idx = eri_computers_[rank]->shell_pairs()[MN];
        size_t M = MN_idx.first, N = MN_idx.second;

        int nfunc_p = auxiliary_->shell(P).nfunction();
        int nfunc_m = primary_->shell(M).nfunction();
        int nfunc_n = primary_->shell(N).nfunction();

        num_doubles += nfunc_p * nfunc_m * nfunc_n;
    }

    const double DOUBLES_TO_GB = pow(10.0, -9) * sizeof(double); // sizeof(double) = 8 bytes
    const double WORDS_TO_GB = pow(10.0, -9); // 1 word = 8 bytes (on 64-bit)

    // TODO: Estimate how much memory is required in buffers
    outfile->Printf("  ==> AO Three-Center DF Ints Memory Requirements <== \n\n");
    outfile->Printf("   *** Required Memory: %8.3f [GB]\n", num_doubles * DOUBLES_TO_GB);
    outfile->Printf("   *** Available Memory: %8.3f [GB]\n\n", memory_ * WORDS_TO_GB);

    // Memory 
    if (num_doubles * sizeof(double) > 0.9 * memory_) {
        outfile->Printf("  Total Required Memory is more than 90%% of Available Memory!\n");
        throw PSIEXCEPTION("  Not enough memory for in-core DF! Try DiskDFJK, or increasing your memory!");
    }

    // Set the variable as an instance variable
    num_doubles_ = num_doubles;

    timer_off("Memory Estimate");

}

// For now, Tensor -> BlockTensor
void EinsumsDFJK::AO2USO() {
    // If already C1, J/K are J_ao/K_ao, pointers are already aliased
    if (AO2USO_->nirrep() == 1) {
        return;
    } else {
        throw PSIEXCEPTION("Complex DFJK does not currently support point group symmetry");
    }

}

// For now, just do unblocking
// BlockTensor -> Tensor
void EinsumsDFJK::USO2AO() {
    // If C1, C_ao and D_ao are equal to C and D
    if (AO2USO_->nirrep() == 1) {
        C_left_ao_ = { std::make_shared<einsums::Tensor<std::complex<double>, 2>>(C_left_[0]) };
        C_right_ao_ = { std::make_shared<einsums::Tensor<std::complex<double>, 2>>(C_right_[0]) };
        D_ao_ = { std::make_shared<einsums::Tensor<std::complex<double>, 2>>(D_[0]) };
        J_ao_ = { std::make_shared<einsums::Tensor<std::complex<double>, 2>>(J_[0]) };
        K_ao_ = { std::make_shared<einsums::Tensor<std::complex<double>, 2>>(K_[0]) };
        wK_ao_ = { std::make_shared<einsums::Tensor<std::complex<double>, 2>>(wK_[0]) };
    } else {
        throw PSIEXCEPTION("Complex DFJK does not currently support point group symmetry");
    }

}

void EinsumsDFJK::preiterations() {}

// compute 2-center DF integrals (coulomb metric)
void EinsumsDFJK::compute_J_metric() {
    
    timer_on("Compute (P|Q)");

    // Compute the full metric, do not invert for now
    auto metric = FittingMetric(auxiliary_, true);
    metric.form_fitting_metric();
    J_metric_ = std::make_shared<Matrix>(metric.get_metric());

    timer_off("Compute (P|Q)");
}

// compute 3-center-DF integrals in AO basis (aux | ao ao)
void EinsumsDFJK::compute_three_center_ao_eri() {

    timer_on("Setup");

    // => Sizing <= //
    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();
    size_t nshell = primary_->nshell();
    size_t naux_shell = auxiliary_->nshell();

    // Significant number of shell pairs (that survive Schwarz/CSAM screening)
    size_t nshell_pair = eri_computers_[0]->shell_pairs().size();

    // Total number of shell triplets to compute
    size_t nshell_triplet = naux_shell * nshell_pair;

    // => Make buffer for three center AO DF ints <= //
    n_function_pairs_ = num_doubles_ / naux;
    df_ao_eri_ = std::make_shared<Matrix>(naux, n_function_pairs_);

    // => Make sparsity map for function pairs (u, v) <= //
    auto function_pairs_reverse = eri_computers_[0]->function_pairs_to_dense();

    timer_off("Setup");

    timer_on("Three-center AO ints");

#pragma omp parallel for schedule(guided) collapse(2) num_threads(df_ints_num_threads_)
    for (size_t P = 0; P < naux_shell; ++P) {
        for (int MN = 0; MN < nshell_pair; ++MN) {

            int rank = 0;
        #ifdef _OPENMP
            rank = omp_get_thread_num();
        #endif

            auto MN_idx = eri_computers_[rank]->shell_pairs()[MN];
            size_t M = MN_idx.first, N = MN_idx.second;

            // Computes the integrals (P | M N) over aux shell P, function shells M and N
            eri_computers_[rank]->compute_shell(P, 0, M, N);

            // Grabs the location in memory where the computed integrals over the triplet
            // is located
            auto &buffer = eri_computers_[rank]->buffers()[0];

            // Starting function index of shell P, as well as number of functions
            int p_start = auxiliary_->shell(P).function_index();
            int nfunc_p = auxiliary_->shell(P).nfunction();

            // Starting function index of shell M, as well as number of functions
            int m_start = primary_->shell(M).function_index();
            int nfunc_m = primary_->shell(M).nfunction();

            // Starting function index of shell N, as well as number of functions
            int n_start = primary_->shell(N).function_index();
            int nfunc_n = primary_->shell(N).nfunction();

            // Move the information from the buffer to the SharedMatrix
            int index = 0; // index represents current position in buffer;

            for (int p = p_start; p < p_start + nfunc_p; ++p) {
                for (int m = m_start; m < m_start + nfunc_m; ++m) {
                    for (int n = n_start; n < n_start + nfunc_n; ++n) {
                        int mn_idx = u_v_to_uv_[m][n];
                        df_ao_eri_->set(p, mn_idx, buffer[index]);
                        index++;
                    } // end n
                } // end m
            } // end p

        } // end MN
    } // end P

    timer_off("Three-center AO ints");
}

void EinsumsDFJK::compute() {
    compute_JK();
}

void EinsumsDFJK::compute_JK() {
    //compute_JK(C_left_, C_right_, D_, J_, K_, wK_);
    compute_JK(C_left_ao_, C_right_ao_, D_ao_, J_ao_, K_ao_, wK_ao_);
}


void EinsumsDFJK::compute_JK(
    const std::vector<EinsumsComplexMatrix>& C_left,
    const std::vector<EinsumsComplexMatrix>& C_right,
    std::vector<EinsumsComplexMatrix>& D,
    std::vector<EinsumsComplexMatrix>& J,
    std::vector<EinsumsComplexMatrix>& K,
    std::vector<EinsumsComplexMatrix>& wK
) {

    // Basis set and basis function information

    // => Sizing <= //
    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();
    size_t nshell = primary_->nshell();
    size_t naux_shell = auxiliary_->nshell();

    // Significant number of shell pairs (that survive Schwarz/CSAM screening)
    size_t nshell_pair = eri_computers_[0]->shell_pairs().size();

    // Total number of shell triplets to compute
    size_t nshell_triplet = naux_shell * nshell_pair;

    // zero out J, K, and wK matrices
    int njk = D.size();

    for (int jk = 0; jk < njk; ++jk) {
        C_left[jk].zero();
        C_right[jk].zero();
        D[jk].zero();
        J[jk].zero();
        K[jk].zero();
        wK[jk].zero();
    }

    // compute J contribution
    // (mn|P)(P|Q)^{-1}(Q|rs)D_{sr}

    // Gamma_Q = \sum_{rs} (Q|rs) D_{sr}
    std::vector<einsums::Tensor<complex_t, 1>> Gamma_Q(njk);
    for (int jk = 0; jk < njk; ++jk) {
        Gamma_Q[jk] = einsums::Tensor<complex_t, 1>("Gamma_Q", naux);
        Gamma_Q[jk].zero();
    }

    auto function_pairs = eri_computers_[0]->function_pairs();

#pragma omp parallel for schedule(dynamic, 1)
    for (int Q = 0; Q < naux; ++Q) {
        for (const auto &rs : function_pairs) { // All significant basis function pairs
            int r = rs.first, s = rs.second;
            double Q_rs = df_ao_eri_->get(Q, rs);

            for (int jk = 0; jk < njk; ++jk) { // compute for all passed in matrices
                Gamma_Q[jk](Q) += Q_rs * D[jk]->get(r, s);
                if (r != s) Gamma_Q[jk](Q) += Q_rs * D[jk]->get(s, r);
            } // end jk

        } // end rs
    } // end Q

    // TODO: make more efficient
    // Gamma_{P} = (P|Q)^{-1}\Gamma_{Q}
    std::vector<einsums::Tensor<complex_t, 1>> Gamma_P;
    for (int jk = 0; jk < njk; ++jk) {
        Gamma_P[jk] = einsums::Tensor<complex_t, 1>("Gamma_P", naux);
        Gamma_P[jk].zero();
    }

    auto J_PQ_clone = J_metric_->clone();
    J_PQ_clone->power(-1.0, condition_); // invert Coulomb metric

#pragma omp parallel for schedule(dynamic, 1)
    for (int P = 0; P < naux; ++P) {
        for (int Q = 0; Q < naux; ++Q) {
            for (int jk = 0; jk < njk; ++jk) {
                Gamma_P[jk](P) += J_PQ_clone->get(P, Q) * Gamma_Q[jk](Q);
            } // end jk
        } // end Q
    } // end P

    // Flush into final J buffer
    // J_{mn} = (mn|P)\Gamma_{P}
#pragma omp parallel for schedule(dynamic, 1)
    for (int mn_idx = 0; mn_idx < function_pairs.size(); mn_idx++) {
        auto mn = function_pairs[mn_idx];
        int m = mn.first, n = mn.second;

        for (int P = 0; P < naux; ++P) {
            double P_mn = df_ao_eri_->get(P, mn_idx);

            for (int jk = 0; jk < njk; ++jk) {
                double J_cont = P_mn * Gamma_P[jk](P);

                J[jk]->set(m, n, J[jk]->get(m, n) + J_cont);
                if (m != n) J[jk]->set(n, m, J[jk]->get(n, m) + J_cont);
            } // end jk
        } // end P
    } // end mn_idx

    // compute the (evil) K contribution (it is the bottleneck)
    // K_{mn} += C^{left}_{ri} (mr|P)(P|Q)^{-1}(Q|ns) (C^{right}_{si})^{*}

    // A^{P}_{mi} = \sum_{r} C^{left}_{ri} (mr|P)
    std::vector<einsums::Tensor<complex_t, 3>> A_P_mi(njk);
    for (int jk = 0; jk < njk; ++jk) {
        A_P_mi[jk] = einsums::Tensor<complex_t, 3>("A^{P}_{mi}", naux, nbf, max_nocc_); // The evil max nocc
        A_P_mi[jk].zero();
    }

    // Assume lr_symmetric_ has already been determined/set (it might be, I'll have to check)
    // B^{P}_{ni} = \sum_{r} C^{right}_{si} (Q|ns) (non-conjugated)
    std::vector<einsums::Tensor<complex_t, 3>> B_P_ni;

    if (!lr_symmetric_) {
        B_P_ni.resize(njk);
        for (int jk = 0; jk < njk; ++jk) {
            B_P_ni[jk] = einsums::Tensor<complex_t, 3>("B^{P}_{ni}", naux, nbf, max_nocc_); // The evil max nocc
            B_P_ni[jk].zero();
        } // end jk
    } // end

    // K_buffers
    for (int P = 0; P < naux; P++) {
        for (const auto &rs : function_pairs) { // All significant basis function pairs
            int r = rs.first, s = rs.second;
            double P_rs = df_ao_eri_->get(P, rs);

            for (int jk = 0; jk < njk; ++jk) {
                
                for (int i = 0; i < max_nocc_; ++i) {
                    // C_left contributions
                    // A^{P}_{mi} = \sum_{r} C^{left}_{ri} (mr|P)
                    A_P_mi[jk](P, r, i) += P_rs * (C_left[jk])(s, i);
                    if (r != s) A_P_mi[jk](P, s, i) += P_rs * (C_left[jk])(r, i);

                    // C_right contributions
                    // B^{P}_{ni} = \sum_{r} C^{right}_{si} (Q|ns) (non-conjugated)
                    if (!lr_symmetric_) {
                        B_P_ni[jk](P, r, i) += P_rs * (C_right[jk])(s, i);
                        if (r != s) B_P_ni[jk](P, s, i) += P_rs * (C_right[jk])(r, i);
                    } // end if

                } // end i
            } // end jk
        } // end rs
    } // end P

    // LAST LOOP FOR TODAY
    auto J_PQ_half = J_metric_->clone();
    J_PQ_half->power(-0.5, condition_); // invert Coulomb metric

    // A^{Q}_{mi} = (P|Q)^{-0.5} A^{P}_{mi}
    std::vector<einsums::Tensor<complex_t, 3>> A_Q_mi(njk);
    for (int jk = 0; jk < njk; ++jk) {
        A_Q_mi[jk] = einsums::Tensor<complex_t, 3>("A^{Q}_{mi}", naux, nbf, max_nocc_); // The evil max nocc
        A_Q_mi[jk].zero();
    }

    // Assume lr_symmetric_ has already been determined/set (it might be, I'll have to check)
    // B^{Q}_{mi} = (P|Q)^{-0.5} B^{P}_{mi} (still not conjugated)
    std::vector<einsums::Tensor<complex_t, 3>> B_Q_ni;
    if (!lr_symmetric_) {
        B_Q_ni.resize(njk);
        for (int jk = 0; jk < njk; ++jk) {
            B_Q_ni[jk] = einsums::Tensor<complex_t, 3>("B^{Q}_{ni}", naux, nbf, max_nocc_); // The evil max nocc
            B_Q_ni[jk].zero();
        } // end jk
    } // end

#pragma omp parallel for schedule(dynamic, 1)
    for (int Q = 0; Q < naux; ++Q) {
        for (int P = 0; P < naux; ++P) {
            for (int m = 0; m < nbf; ++m) {
                for (int i = 0; i < max_nocc_; ++i) {
                    for (int jk = 0; jk < njk; ++jk) {
                        (A_Q_mi[jk])(Q, m, i) += J_PQ_half->get(Q, P) * (A_P_mi[jk])(P, m, i);
                        if (!lr_symmetric_) (B_Q_ni[jk])(Q, m, i) += J_PQ_half->get(Q, P) * (B_P_ni[jk])(P, m, i);
                    } // end jk
                } // end i
            } // end m
        } // end Q
    } // end P

    // NOW IS FINALLY THE LAST LOOP
#pragma omp parallel for schedule(dynamic, 1)
    for (int mn_idx = 0; mn_idx < function_pairs.size(); mn_idx++) {
        auto mn = function_pairs[mn_idx];
        int m = mn.first, n = mn.second;

        for (int Q = 0; Q < naux; ++Q) {
            for (int i = 0; i < max_nocc_; ++i) {
                for (int jk = 0; jk < njk; ++jk) {
                    double K_cont = 0.0;
                    if (lr_symmetric_) {
                        K_cont = (A_Q_mi[jk])(Q, m, i) * std::conj((A_Q_mi[jk])(Q, n, i));
                    } else {
                        K_cont = (A_Q_mi[jk])(Q, m, i) * std::conj((B_Q_ni[jk])(Q, n, i));
                    }

                    K[jk]->set(m, n, K[jk]->get(m, n) + K_cont);
                    
                    // Account for m != n case
                    if (m != n) {
                        double K_cont = 0.0;
                        if (lr_symmetric_) {
                            K_cont = (A_Q_mi[jk])(Q, n, i) * std::conj((A_Q_mi[jk])(Q, m, i));
                        } else {
                            K_cont = (A_Q_mi[jk])(Q, n, i) * std::conj((B_Q_ni[jk])(Q, m, i));
                        }

                        K[jk]->set(n, m, K[jk]->get(n, m) + K_cont);
                    } // end if
                } // end jk
            } // end i
        } // end Q
    } // end mn_idx
}

void EinsumsDFJK::postiterations() {}

bool EinsumsDFJK::C1() const {
    // Choose the correct behavior for your implementation.
    // If you don’t support symmetry (i.e., effectively C1 only), return true.
    // If you *do* support symmetry, return false or compute it properly.
    return true;
}

}  // namespace psi
