#include "composite.h"

#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libqt/qt.h"

#include <vector>
#include <algorithm>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

using namespace psi;

namespace psi {

DirectDFJ::DirectDFJ(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options)
                        : SplitJKBase(primary, options), auxiliary_(auxiliary) {
    cutoff_ = options_.get_double("INTS_TOLERANCE");
    density_screening_ = options_.get_str("SCREENING") == "DENSITY";

    build_metric();
    build_ints();
}

void DirectDFJ::print_header() {
    if (print_) {
        outfile->Printf("  ==> Direct Density-Fitted J <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Auxiliary Basis: %11s\n", auxiliary_->name().c_str());
        outfile->Printf("    ERI Screening Cutoff: %11.0E\n", cutoff_);
        outfile->Printf("\n");
    }
}

void DirectDFJ::build_metric() {
    timer_on("DirectDFJ: Build Metric");

    // build Coulomb metric
    auto metric = std::make_shared<FittingMetric>(auxiliary_, true);
    metric->form_fitting_metric();
    Jmet_ = metric->get_metric();

    // build Coulomb metric maximums vector
    Jmet_max_= std::vector<double>(auxiliary_->nshell(), 0.0);
    double **Jmetp = Jmet_->pointer();

    if (density_screening_) {
#pragma omp parallel for
        for (size_t P = 0; P < auxiliary_->nshell(); P++) {
            int p_start = auxiliary_->shell_to_basis_function(P);
            int num_p = auxiliary_->shell(P).nfunction();
            for (size_t p = p_start; p < p_start + num_p; p++) {
                Jmet_max_[P] = std::max(Jmet_max_[P], Jmetp[p][p]);
            }
        }
    }

    timer_off("DirectDFJ: Build Metric");
}

void DirectDFJ::build_ints() {
    timer_on("DirectDFJ: Build Ints");

    auto zero = BasisSet::zero_ao_basis_set();
    auto rifactory = std::make_shared<IntegralFactory>(auxiliary_, zero, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(rifactory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }

    timer_off("DirectDFJ: Build Ints");
}

void DirectDFJ::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("DirectDFJ: J");

    // => Zeroing <= //

    for (auto& Jmat : J) {
        Jmat->zero();
    }

    // => Sizing <= //

    int pri_nshell = primary_->nshell();
    int aux_nshell = auxiliary_->nshell();
    int nmat = D.size();

    int nbf = primary_->nbf();
    int naux = auxiliary_->nbf();

    // => Get significant primary shells <=
    const auto& shell_pairs = ints_[0]->shell_pairs();

    // maximum values of Density matrix for shell pair block UV
    // TODO: Integrate this more smoothly into current density screening framework
    Matrix D_max(pri_nshell, pri_nshell);
    auto D_maxp = D_max.pointer();

    if (density_screening_) {
        for (size_t UV = 0; UV < pri_nshell * pri_nshell; UV++) {
            size_t U = UV / pri_nshell;
            size_t V = UV % pri_nshell;

            int u_start = primary_->shell_to_basis_function(U);
            int num_u = primary_->shell(U).nfunction();
	        
            int v_start = primary_->shell_to_basis_function(V);
            int num_v = primary_->shell(V).nfunction();

	        for (size_t i = 0; i < D.size(); i++) {
                auto Dp = D[i]->pointer();
                for (size_t u = u_start; u < u_start + num_u; u++) {
                    for (size_t v = v_start; v < v_start + num_v; v++) {
                        D_maxp[U][V] = std::max(D_maxp[U][V], std::abs(Dp[u][v]));
                    }
                }
            }
        }
    }

    // Weigand 2002 doi: 10.1039/b204199p (Figure 1)
    // The gamma intermediates defined in Figure 1
    // gammaP = (P|uv) * Duv
    // (P|Q) * gammaQ = gammaP
    SharedMatrix gamma = std::make_shared<Matrix>(nmat, naux); 
    gamma->zero();
    double* gamp = gamma->pointer()[0];

    int max_nbf_per_shell = 0;
    for (int P = 0; P < pri_nshell; P++) {
        max_nbf_per_shell = std::max(max_nbf_per_shell, primary_->shell(P).nfunction());
    }

    // Temporary buffers for J to minimize race tonditions
    std::vector<std::vector<SharedMatrix>> JT;

    for (int thread = 0; thread < nthread_; thread++) {
        std::vector<SharedMatrix> J2;
        for (size_t ind = 0; ind < nmat; ind++) {
            J2.push_back(std::make_shared<Matrix>(max_nbf_per_shell, max_nbf_per_shell));
            J2[ind]->zero();
        }
        JT.push_back(J2);
    }

    // Solve for gammaP = (P|rs)*Drs
#pragma omp parallel for num_threads(nthread_) schedule(dynamic)
    for (int P = 0; P < aux_nshell; P++) {

        int p_start = auxiliary_->shell_to_basis_function(P);
        int num_p = auxiliary_->shell(P).nfunction();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (const auto& UV : shell_pairs) {

            int U = UV.first;
            int V = UV.second;

            if (density_screening_) {
	            double screen_val = D_maxp[U][V] * D_maxp[U][V] * Jmet_max_[P] * ints_[thread]->shell_pair_value(U, V);
	            if (screen_val < cutoff_ * cutoff_) continue;
            }

            int u_start = primary_->shell_to_basis_function(U);
            int num_u = primary_->shell(U).nfunction();

            int v_start = primary_->shell_to_basis_function(V);
            int num_v = primary_->shell(V).nfunction();

            ints_[thread]->compute_shell(P, 0, U, V);

            const double* buffer = ints_[thread]->buffer();

            double prefactor = 2.0;
            if (U == V) prefactor *= 0.5;

            for (int i = 0; i < D.size(); i++) {
                double** Dp = D[i]->pointer();
                double *Puv = const_cast<double *>(buffer);

                /*
                for (int p = p_start; p < p_start + num_p; p++) {
                    int dp = p - p_start;
                    for (int u = u_start; u < u_start + num_u; u++) {
                        int du = u - u_start;
                        for (int v = v_start; v < v_start + num_v; v++) {
                            int dv = v - v_start;
                            gamp[i * naux + p] += prefactor * (*Puv) * Dp[u][v];
                            Puv++;
                        }
                    }
                }
                */

                std::vector<double> Dbuff(num_u * num_v, 0.0);
                for (int u = u_start; u < u_start + num_u; u++) {
                    int du = u - u_start;
                    for (int v = v_start; v < v_start + num_v; v++) {
                        int dv = v - v_start;
                        Dbuff[du * num_v + dv] = Dp[u][v];
                    }
                }
                C_DGEMV('N', num_p, num_u * num_v, prefactor, (double *)Puv, num_u * num_v, Dbuff.data(), 1, 1.0, &(gamp[i * naux + p_start]), 1);
            }
        }
    }

    // Solve for gammaQ, (P|Q) * gammaQ = gammaP
    SharedMatrix Jmet_copy = Jmet_->clone();

    std::vector<int> ipiv(naux);
    C_DGESV(naux, nmat, Jmet_copy->pointer()[0], naux, ipiv.data(), gamp, naux);

    // set up gamp_max for screening purposes
    std::vector<double> gamp_max(aux_nshell, 0.0);
    if (density_screening_) {
#pragma omp parallel for
        for (int P = 0; P < aux_nshell; P++) {
            int p_start = auxiliary_->shell_to_basis_function(P);
            int num_p = auxiliary_->shell(P).nfunction();
            for (size_t i = 0; i < D.size(); i++) {
                for (int p = p_start; p < p_start + num_p; p++) {
                    gamp_max[P] = std::max(gamp_max[P], std::abs(gamp[i * naux + p]));
                }
            }
        }
    }

#pragma omp parallel for num_threads(nthread_) schedule(dynamic)
    for (int UV = 0; UV < shell_pairs.size(); UV++) {
        int U = shell_pairs[UV].first;
        int V = shell_pairs[UV].second;

        int u_start = primary_->shell_to_basis_function(U);
        int num_u = primary_->shell(U).nfunction();

        int v_start = primary_->shell_to_basis_function(V);
        int num_v = primary_->shell(V).nfunction();

        double prefactor = 2.0;
        if (U == V) prefactor *= 0.5;

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int Q = 0; Q < aux_nshell; Q++) {
            if (density_screening_) {
                double screen_val = gamp_max[Q] * gamp_max[Q] * Jmet_max_[Q] * ints_[thread]->shell_pair_value(U,V);
	            if (screen_val < cutoff_ * cutoff_) continue;
            }

		    int q_start = auxiliary_->shell_to_basis_function(Q);
            int num_q = auxiliary_->shell(Q).nfunction();

            ints_[thread]->compute_shell(Q, 0, U, V);

            const double* buffer = ints_[thread]->buffer();

            for (int i = 0; i < D.size(); i++) {
                double* JTp = JT[thread][i]->pointer()[0];
                double* Quv = const_cast<double *>(buffer);

                    /*
                    for (int q = q_start; q < q_start + num_q; q++) {
                        int dq = q - q_start;
                        for (int u = u_start; u < u_start + num_u; u++) {
                            int du = u - u_start;
                            for (int v = v_start; v < v_start + num_v; v++) {
                                int dv = v - v_start;
                                JTp[du * num_v + dv] += prefactor * (*Quv) * gamp[i * naux + q];
                                Quv++;
                            }
                        }
                    }
                    */
                C_DGEMV('T', num_q, num_u * num_v, prefactor, (double *) Quv, num_u * num_v, &(gamp[i * naux + q_start]), 1, 1.0, JTp, 1);
            }
        }

        // => Stripeout <= //

        for (int i = 0; i < D.size(); i++) {
            double* JTp = JT[thread][i]->pointer()[0];
            double** Jp = J[i]->pointer();
            for (int u = u_start; u < u_start + num_u; u++) {
                int du = u - u_start;
                for (int v = v_start; v < v_start + num_v; v++) {
                    int dv = v - v_start;
                    Jp[u][v] += JTp[du * num_v + dv];
                }
            }
            JT[thread][i]->zero();
        }
    }

    for (auto& Jmat : J) {
        Jmat->hermitivitize();
    }

    timer_off("DirectDFJ: J");
}

} // namespace psi