#include "psi4/libfock/composite.h"
#include "psi4/libfock/jk.h"

#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

using namespace psi;

namespace psi {

SplitJKBase::SplitJKBase(std::shared_ptr<BasisSet> primary, Options& options) : primary_(primary), options_(options) {

    print_ = options_.get_int("PRINT");
    debug_ = options_.get_int("DEBUG");
    bench_ = options_.get_int("BENCH");

    nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

}

CompositeJK::CompositeJK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, 
                         std::string jtype, std::string ktype, Options& options)
                        : JK(primary), auxiliary_(auxiliary), jtype_(jtype), ktype_(ktype), options_(options) { common_init(); }

void CompositeJK::common_init() {

    nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

    if (jtype_ == "DIRECT_DF") {
        jalgo_ = std::make_shared<DirectDFJ>(primary_, auxiliary_, options_);
    } else if (jtype_ == "CFMM") {
        jalgo_ = std::make_shared<CFMM>(primary_, options_);
    } else if (jtype_ == "DFCFMM") {
        jalgo_ = std::make_shared<DFCFMM>(primary_, auxiliary_, options_);
    } else if (jtype_ == "LOCAL_DF") {
        jalgo_ = std::make_shared<LocalDFJ>(primary_, auxiliary_, options_);
    } else {
        throw PSIEXCEPTION("J BUILD TYPE " + jtype_ + " IS NOT SUPPORTED IN COMPOSITE JK!");
    }

    if (ktype_ == "LINK") {
        kalgo_ = std::make_shared<LinK>(primary_, options_);
    } else if (ktype_ == "LOCAL_DF") {
        kalgo_ = std::make_shared<LocalDFK>(primary_, auxiliary_, options_);
    } else {
        throw PSIEXCEPTION("K BUILD TYPE " + ktype_ + " IS NOT SUPPORTED IN COMPOSITE JK!");
    }
}

size_t CompositeJK::memory_estimate() {
    return 0;   // only Direct-based integral algorithms are currently implemented in Composite JK
}

void CompositeJK::preiterations() {}
void CompositeJK::postiterations() {}

void CompositeJK::print_header() const {
    std::string screen_type = options_.get_str("SCREENING");
    if (print_) {
        outfile->Printf("  ==> CompositeJK: Mix/Match JK Builds <==\n\n");
        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        if (do_J_) outfile->Printf("    J Algorithm:       %11s\n", jtype_.c_str());
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        if (do_K_) outfile->Printf("    K Algorithm:       %11s\n", ktype_.c_str());
        outfile->Printf("    Integrals threads: %11d\n", nthread_);
        outfile->Printf("    Incremental Fock:  %11s\n", incfock_ ? "Yes" : "No");
        outfile->Printf("\n");
    }
    jalgo_->print_header();
    kalgo_->print_header();
}

void CompositeJK::compute_JK() {

    if (!lr_symmetric_) {
        throw PSIEXCEPTION("Non-symmetric K matrix builds are currently not supported in Composite JK.");
    }

    if (incfock_) {
        timer_on("CompositeJK: INCFOCK Preprocessing");
        incfock_setup();
        int reset = options_.get_int("INCFOCK_FULL_FOCK_EVERY");
        double dconv = options_.get_double("INCFOCK_CONVERGENCE");
        double Dnorm = Process::environment.globals["SCF D NORM"];
        // Do IFB on this iteration?
        do_incfock_iter_ = (Dnorm >= dconv) && !initial_iteration_ && (incfock_count_ % reset != reset - 1);
        
        if (!initial_iteration_ && (Dnorm >= dconv)) incfock_count_ += 1;
        timer_off("CompositeJK: INCFOCK Preprocessing");
    }

    // Matrices to use/build depending on whether or not incremental Fock build is performed in the iteration
    std::vector<SharedMatrix>& D_ref = (do_incfock_iter_ ? delta_D_ao_ : D_ao_);
    std::vector<SharedMatrix>& J_ref = (do_incfock_iter_ ? delta_J_ao_ : J_ao_);
    std::vector<SharedMatrix>& K_ref = (do_incfock_iter_ ? delta_K_ao_ : K_ao_);

    jalgo_->build_G_component(D_ref, J_ref);
    kalgo_->build_G_component(D_ref, K_ref);

    if (incfock_) {
        timer_on("CompositeJK: INCFOCK Postprocessing");
        incfock_postiter();
        timer_off("CompositeJK: INCFOCK Postprocessing");
    }

    if (initial_iteration_) initial_iteration_ = false;
}

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
    if (density_screening_) {
        for (size_t P = 0; P < auxiliary_->nshell(); P++) {
            int p_start = auxiliary_->shell_to_basis_function(P);
            int num_p = auxiliary_->shell(P).nfunction();
            for (size_t p = p_start; p < p_start + num_p; p++) {
                Jmet_max_[P] = std::max(Jmet_max_[P], Jmet_->get(p, p));
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

    for (size_t U = 0; U < pri_nshell; U++) {
        int u_start = primary_->shell_to_basis_function(U);
        int num_u = primary_->shell(U).nfunction();
	
	    for (size_t V = 0; V < pri_nshell; V++) {
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
       
	        double screen_val = D_maxp[U][V] * D_maxp[U][V] * Jmet_max_[P] * ints_[thread]->shell_pair_value(U, V);
	        if (screen_val < cutoff_ * cutoff_) continue;
 
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
    for(size_t i = 0; i < D.size(); i++) {
        for (int P = 0; P < aux_nshell; P++) {
            int p_start = auxiliary_->shell_to_basis_function(P);
            int num_p = auxiliary_->shell(P).nfunction();
            for (int p = p_start; p < p_start + num_p; p++) {
                gamp_max[P] = std::max(gamp_max[P], std::abs(gamp[i * naux + p]));
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
            double screen_val = gamp_max[Q] * gamp_max[Q] * Jmet_max_[Q] * ints_[thread]->shell_pair_value(U,V);
	        if (screen_val < cutoff_ * cutoff_) continue;
                
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

LinK::LinK(std::shared_ptr<BasisSet> primary, Options& options) : SplitJKBase(primary, options) {
    cutoff_ = options.get_double("INTS_TOLERANCE");
    if (options_["LINK_INTS_TOLERANCE"].has_changed()) {
        linK_ints_cutoff_ = options_.get_double("LINK_INTS_TOLERANCE");
    } else {
        linK_ints_cutoff_ = options_.get_double("INTS_TOLERANCE");
    }

    build_ints();
}

void LinK::print_header() {
    if (print_) {
        outfile->Printf("  ==> Linear Exchange (LinK) <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    ERI Screening Cutoff: %11.0E\n", cutoff_);
        outfile->Printf("    Density Screening Cutoff: %11.0E\n", linK_ints_cutoff_);
        outfile->Printf("\n");
    }
}

void LinK::build_ints() {
    timer_on("LinK: Build Ints");

    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }

    timer_off("LinK: Build Ints");
}

// To follow this code, compare with figure 1 of DOI: 10.1063/1.476741
void LinK::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) {

    timer_on("LinK: K");

    for (auto& integral : ints_) {
        integral->update_density(D);
    }
    // ==> Prep Auxiliary Quantities <== //

    // => Zeroing <= //
    for (auto& Kmat : K) {
        Kmat->zero();
    }

    // => Sizing <= //
    int nshell = primary_->nshell();
    int nbf = primary_->nbf();
    int nthread = nthread_;

    // => Atom Blocking <= //
    std::vector<int> shell_endpoints_for_atom;
    std::vector<int> basis_endpoints_for_shell;

    int atomic_ind = -1;
    for (int P = 0; P < nshell; P++) {
        if (primary_->shell(P).ncenter() > atomic_ind) {
            shell_endpoints_for_atom.push_back(P);
            atomic_ind++;
        }
        basis_endpoints_for_shell.push_back(primary_->shell_to_basis_function(P));
    }
    shell_endpoints_for_atom.push_back(nshell);
    basis_endpoints_for_shell.push_back(nbf);

    size_t natom = shell_endpoints_for_atom.size() - 1;

    size_t max_functions_per_atom = 0L;
    for (size_t atom = 0; atom < natom; atom++) {
        size_t size = 0L;
        for (int P = shell_endpoints_for_atom[atom]; P < shell_endpoints_for_atom[atom + 1]; P++) {
            size += primary_->shell(P).nfunction();
        }
        max_functions_per_atom = std::max(max_functions_per_atom, size);
    }

    if (debug_) {
        outfile->Printf("  ==> LinK: Atom Blocking <==\n\n");
        for (size_t atom = 0; atom < natom; atom++) {
            outfile->Printf("  Atom: %3d, Atom Start: %4d, Atom End: %4d\n", atom, shell_endpoints_for_atom[atom],
                            shell_endpoints_for_atom[atom + 1]);
            for (int P = shell_endpoints_for_atom[atom]; P < shell_endpoints_for_atom[atom + 1]; P++) {
                int size = primary_->shell(P).nfunction();
                int off = primary_->shell(P).function_index();
                int off2 = basis_endpoints_for_shell[P];
                outfile->Printf("    Shell: %4d, Size: %4d, Offset: %4d, Offset2: %4d\n", P, size, off,
                                off2);
            }
        }
        outfile->Printf("\n");
    }

    // ==> Prep Atom Pairs <== //
    // Atom-pair blocking inherited from DirectJK code
    // TODO: Test shell-pair blocking

    std::vector<std::pair<int, int>> atom_pairs;
    for (size_t Patom = 0; Patom < natom; Patom++) {
        for (size_t Qatom = 0; Qatom <= Patom; Qatom++) {
            bool found = false;
            for (int P = shell_endpoints_for_atom[Patom]; P < shell_endpoints_for_atom[Patom + 1]; P++) {
                for (int Q = shell_endpoints_for_atom[Qatom]; Q < shell_endpoints_for_atom[Qatom + 1]; Q++) {
                    if (ints_[0]->shell_pair_significant(P, Q)) {
                        found = true;
                        atom_pairs.emplace_back(Patom, Qatom);
                        break;
                    }
                }
                if (found) break;
            }
        }
    }

    // ==> Prep Bra-Bra Shell Pairs <== //

    // A comparator used for sorting integral screening values
    auto screen_compare = [](const std::pair<int, double> &a, 
                                    const std::pair<int, double> &b) { return a.second > b.second; };

    std::vector<std::vector<int>> significant_bras(nshell);
    double max_integral = ints_[0]->max_integral();

#pragma omp parallel for
    for (size_t P = 0; P < nshell; P++) {
        std::vector<std::pair<int, double>> PQ_shell_values;
        for (size_t Q = 0; Q < nshell; Q++) {
            double pq_pq = std::sqrt(ints_[0]->shell_ceiling2(P, Q, P, Q));
            double schwarz_value = std::sqrt(pq_pq * max_integral);
            if (schwarz_value >= cutoff_) {
                PQ_shell_values.emplace_back(Q, schwarz_value);
            }
        }
        std::sort(PQ_shell_values.begin(), PQ_shell_values.end(), screen_compare);

        for (const auto& value : PQ_shell_values) {
            significant_bras[P].push_back(value.first);
        }
    }

    // ==> Prep Bra-Ket Shell Pairs <== //

    // => Calculate Shell Ceilings <= //
    std::vector<double> shell_ceilings(nshell, 0.0);

    // sqrt(Umax|Umax) in Ochsenfeld Eq. 3
#pragma omp parallel for
    for (int P = 0; P < nshell; P++) {
        for (int Q = 0; Q <= P; Q++) {
            double val = std::sqrt(ints_[0]->shell_ceiling2(P, Q, P, Q));
            shell_ceilings[P] = std::max(shell_ceilings[P], val);
#pragma omp critical
            shell_ceilings[Q] = std::max(shell_ceilings[Q], val);
        }
    }

    std::vector<std::vector<int>> significant_kets(nshell);

    // => Use shell ceilings to compute significant ket-shells for each bra-shell <= //
#pragma omp parallel for
    for (size_t P = 0; P < nshell; P++) {
        std::vector<std::pair<int, double>> PR_shell_values;
        for (size_t R = 0; R < nshell; R++) {
            double screen_val = shell_ceilings[P] * shell_ceilings[R] * ints_[0]->shell_pair_max_density(P, R);
            if (screen_val >= linK_ints_cutoff_) {
                PR_shell_values.emplace_back(R, screen_val);
            }
        }
        std::sort(PR_shell_values.begin(), PR_shell_values.end(), screen_compare);

        for (const auto& value : PR_shell_values) {
            significant_kets[P].push_back(value.first);
        }
    }

    size_t natom_pair = atom_pairs.size();

    // ==> Intermediate Buffers <== //

    // Temporary buffers used during the K contraction process to
    // Take full advantage of permutational symmetry of ERIs
    std::vector<std::vector<SharedMatrix>> KT;

    // To prevent race conditions, give every thread a buffer
    for (int thread = 0; thread < nthread; thread++) {
        std::vector<SharedMatrix> K2;
        for (size_t ind = 0; ind < D.size(); ind++) {
            // (pq|rs) can be contracted into Kpr, Kps, Kqr, Kqs (hence the 4)
            K2.push_back(std::make_shared<Matrix>("KT (linK)", 4 * max_functions_per_atom, nbf));
        }
        KT.push_back(K2);
    }

    // Number of computed shell quartets is tracked for benchmarking purposes
    size_t computed_shells = 0L;

    // ==> Integral Formation Loop <== //

#pragma omp parallel for num_threads(nthread) schedule(dynamic) reduction(+ : computed_shells)
    for (size_t ipair = 0L; ipair < natom_pair; ipair++) { // O(N) shell-pairs in asymptotic limit

        int Patom = atom_pairs[ipair].first;
        int Qatom = atom_pairs[ipair].second;
        
        // Number of shells per atom
        int nPshell = shell_endpoints_for_atom[Patom + 1] - shell_endpoints_for_atom[Patom];
        int nQshell = shell_endpoints_for_atom[Qatom + 1] - shell_endpoints_for_atom[Qatom];

        // First shell per atom
        int Pstart = shell_endpoints_for_atom[Patom];
        int Qstart = shell_endpoints_for_atom[Qatom];

        // Number of basis functions per atom
        int nPbasis = basis_endpoints_for_shell[Pstart + nPshell] - basis_endpoints_for_shell[Pstart];
        int nQbasis = basis_endpoints_for_shell[Qstart + nQshell] - basis_endpoints_for_shell[Qstart];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // Keep track of contraction indices for stripeout (Towards end of this function)
        std::vector<std::unordered_set<int>> P_stripeout_list(nPshell);
        std::vector<std::unordered_set<int>> Q_stripeout_list(nQshell);

        bool touched = false;
        for (int P = Pstart; P < Pstart + nPshell; P++) {
            for (int Q = Qstart; Q < Qstart + nQshell; Q++) {

                if (Q > P) continue;
                if (!ints_[0]->shell_pair_significant(P, Q)) continue;

                int dP = P - Pstart;
                int dQ = Q - Qstart;

                // => Formation of Significant Shell Pair List ML <= //

                // Significant ket shell pairs RS for bra shell pair PQ
                // represents the merge of ML_P and ML_Q (mini-lists) as defined in Oschenfeld
                // Unordered set structure allows for automatic merging as new elements are added
                std::unordered_set<int> ML_PQ;

                // Form ML_P as part of ML_PQ
                for (const int R : significant_kets[P]) {
                    bool is_significant = false;
                    for (const int S : significant_bras[R]) {
                        double screen_val = ints_[0]->shell_pair_max_density(P, R) * std::sqrt(ints_[0]->shell_ceiling2(P, Q, R, S));

                        if (screen_val >= linK_ints_cutoff_) {
                            if (!is_significant) is_significant = true;
                            int RS = (R >= S) ? (R * nshell + S) : (S * nshell + R);
                            if (RS > P * nshell + Q) continue;
                            ML_PQ.emplace(RS);
                            Q_stripeout_list[dQ].emplace(S);
                        }
                        else break;
                    }
                    if (!is_significant) break;
                }

                // Form ML_Q as part of ML_PQ
                for (const int R : significant_kets[Q]) {
                    bool is_significant = false;
                    for (const int S : significant_bras[R]) {
                        double screen_val = ints_[0]->shell_pair_max_density(Q, R) * std::sqrt(ints_[0]->shell_ceiling2(P, Q, R, S));

                        if (screen_val >= linK_ints_cutoff_) {
                            if (!is_significant) is_significant = true;
                            int RS = (R >= S) ? (R * nshell + S) : (S * nshell + R);
                            if (RS > P * nshell + Q) continue;
                            ML_PQ.emplace(RS);
                            P_stripeout_list[dP].emplace(S);
                        }
                        else break;
                    }
                    if (!is_significant) break;
                }

                // Loop over significant RS pairs
                for (const int RS : ML_PQ) {

                    int R = RS / nshell;
                    int S = RS % nshell;

                    if (!ints_[0]->shell_pair_significant(R, S)) continue;
                    if (!ints_[0]->shell_significant(P, Q, R, S)) continue;

                    if (ints_[thread]->compute_shell(P, Q, R, S) == 0)
                        continue;
                    computed_shells++;

                    const double* buffer = ints_[thread]->buffer();

                    // Number of basis functions in shells P, Q, R, S
                    int shell_P_nfunc = primary_->shell(P).nfunction();
                    int shell_Q_nfunc = primary_->shell(Q).nfunction();
                    int shell_R_nfunc = primary_->shell(R).nfunction();
                    int shell_S_nfunc = primary_->shell(S).nfunction();

                    // Basis Function Starting index for shell
                    int shell_P_start = primary_->shell(P).function_index();
                    int shell_Q_start = primary_->shell(Q).function_index();
                    int shell_R_start = primary_->shell(R).function_index();
                    int shell_S_start = primary_->shell(S).function_index();

                    // Basis Function offset from first basis function in the atom
                    int shell_P_offset = basis_endpoints_for_shell[P] - basis_endpoints_for_shell[Pstart];
                    int shell_Q_offset = basis_endpoints_for_shell[Q] - basis_endpoints_for_shell[Qstart];

                    for (size_t ind = 0; ind < D.size(); ind++) {
                        double** Kp = K[ind]->pointer();
                        double** Dp = D[ind]->pointer();
                        double** KTp = KT[thread][ind]->pointer();
                        const double* buffer2 = buffer;

                        if (!touched) {
                            ::memset((void*)KTp[0L * max_functions_per_atom], '\0', nPbasis * nbf * sizeof(double));
                            ::memset((void*)KTp[1L * max_functions_per_atom], '\0', nPbasis * nbf * sizeof(double));
                            ::memset((void*)KTp[2L * max_functions_per_atom], '\0', nQbasis * nbf * sizeof(double));
                            ::memset((void*)KTp[3L * max_functions_per_atom], '\0', nQbasis * nbf * sizeof(double));
                        }

                        // Four pointers needed for PR, PS, QR, QS
                        double* K1p = KTp[0L * max_functions_per_atom];
                        double* K2p = KTp[1L * max_functions_per_atom];
                        double* K3p = KTp[2L * max_functions_per_atom];
                        double* K4p = KTp[3L * max_functions_per_atom];

                        double prefactor = 1.0;
                        if (P == Q) prefactor *= 0.5;
                        if (R == S) prefactor *= 0.5;
                        if (P == R && Q == S) prefactor *= 0.5;

                        // => Computing integral contractions to K buffers <= //
                        for (int p = 0; p < shell_P_nfunc; p++) {
                            for (int q = 0; q < shell_Q_nfunc; q++) {
                                for (int r = 0; r < shell_R_nfunc; r++) {
                                    for (int s = 0; s < shell_S_nfunc; s++) {

                                        K1p[(p + shell_P_offset) * nbf + r + shell_R_start] +=
                                            prefactor * (Dp[q + shell_Q_start][s + shell_S_start]) * (*buffer2);
                                        K2p[(p + shell_P_offset) * nbf + s + shell_S_start] +=
                                            prefactor * (Dp[q + shell_Q_start][r + shell_R_start]) * (*buffer2);
                                        K3p[(q + shell_Q_offset) * nbf + r + shell_R_start] +=
                                            prefactor * (Dp[p + shell_P_start][s + shell_S_start]) * (*buffer2);
                                        K4p[(q + shell_Q_offset) * nbf + s + shell_S_start] +=
                                            prefactor * (Dp[p + shell_P_start][r + shell_R_start]) * (*buffer2);

                                        buffer2++;
                                    }
                                }
                            }
                        }
                    }
                    touched = true;
                }
            }
        }

        // => Master shell quartet loops <= //

        if (!touched) continue;

        // => Stripe out (Writing to K matrix) <= //

        for (size_t ind = 0; ind < D.size(); ind++) {
            double** KTp = KT[thread][ind]->pointer();
            double** Kp = K[ind]->pointer();

            double* K1p = KTp[0L * max_functions_per_atom];
            double* K2p = KTp[1L * max_functions_per_atom];
            double* K3p = KTp[2L * max_functions_per_atom];
            double* K4p = KTp[3L * max_functions_per_atom];

            // K_PR and K_PS
            for (int P = Pstart; P < Pstart + nPshell; P++) {
                int dP = P - Pstart;
                int shell_P_start = primary_->shell(P).function_index();
                int shell_P_nfunc = primary_->shell(P).nfunction();
                int shell_P_offset = basis_endpoints_for_shell[P] - basis_endpoints_for_shell[Pstart];
                for (const int S : P_stripeout_list[dP]) {
                    int shell_S_start = primary_->shell(S).function_index();
                    int shell_S_nfunc = primary_->shell(S).nfunction();

                    for (int p = 0; p < shell_P_nfunc; p++) {
                        for (int s = 0; s < shell_S_nfunc; s++) {
#pragma omp atomic
                            Kp[shell_P_start + p][shell_S_start + s] += K1p[(p + shell_P_offset) * nbf + s + shell_S_start];
#pragma omp atomic
                            Kp[shell_P_start + p][shell_S_start + s] += K2p[(p + shell_P_offset) * nbf + s + shell_S_start];
                        }
                    }

                }
            }

            // K_QR and K_QS
            for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                int dQ = Q - Qstart;
                int shell_Q_start = primary_->shell(Q).function_index();
                int shell_Q_nfunc = primary_->shell(Q).nfunction();
                int shell_Q_offset = basis_endpoints_for_shell[Q] - basis_endpoints_for_shell[Qstart];
                for (const int S : Q_stripeout_list[dQ]) {
                    int shell_S_start = primary_->shell(S).function_index();
                    int shell_S_nfunc = primary_->shell(S).nfunction();

                    for (int q = 0; q < shell_Q_nfunc; q++) {
                        for (int s = 0; s < shell_S_nfunc; s++) {
#pragma omp atomic
                            Kp[shell_Q_start + q][shell_S_start + s] += K3p[(q + shell_Q_offset) * nbf + s + shell_S_start];
#pragma omp atomic
                            Kp[shell_Q_start + q][shell_S_start + s] += K4p[(q + shell_Q_offset) * nbf + s + shell_S_start];
                        }
                    }

                }
            }

        }  // End stripe out

    }  // End master task list

    for (auto& Kmat : K) {
        Kmat->scale(2.0);
        Kmat->hermitivitize();
    }

    if (bench_) {
        auto mode = std::ostream::app;
        auto printer = PsiOutStream("bench.dat", mode);
        size_t ntri = nshell * (nshell + 1L) / 2L;
        size_t possible_shells = ntri * (ntri + 1L) / 2L;
        printer.Printf("(LinK) Computed %20zu Shell Quartets out of %20zu, (%11.3E ratio)\n", computed_shells,
                        possible_shells, computed_shells / (double)possible_shells);
    }

    timer_off("LinK: K");
}

CFMM::CFMM(std::shared_ptr<BasisSet> primary, Options& options) : SplitJKBase(primary, options) {
    cfmmtree_ = std::make_shared<CFMMTree>(primary_, nullptr, options_);
    build_ints();
}

void CFMM::build_ints() {
    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }
}

void CFMM::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("CFMM: J");

    cfmmtree_->build_J(ints_, D, J);

    timer_off("CFMM: J");
}

void CFMM::print_header() {
    if (print_) {
        outfile->Printf("  ==> Continuous Fast Multipole Method (CFMM) <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", cfmmtree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", cfmmtree_->nlevels());
        outfile->Printf("\n");
    }
}

DFCFMM::DFCFMM(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, 
               Options& options) : DirectDFJ(primary, auxiliary, options) {

    df_cfmm_tree_ = std::make_shared<CFMMTree>(primary_, auxiliary_, options_);
}

void DFCFMM::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {
    timer_on("DFCFMM: J");

    int naux = auxiliary_->nbf();

    if (gamma.size() == 0) {
        gamma.resize(D.size());
        for (int i = 0; i < D.size(); i++) {
            gamma[i] = std::make_shared<Matrix>(naux, 1);
        }
    }

    // Build gammaP = (P|uv)Duv
    df_cfmm_tree_->df_set_contraction(ContractionType::DF_AUX_PRI);
    df_cfmm_tree_->build_J(ints_, D, gamma, Jmet_max_);

    // Solve for gammaQ => (P|Q)*gammaQ = gammaP
    for (int i = 0; i < D.size(); i++) {
        SharedMatrix Jmet_copy = Jmet_->clone();
        std::vector<int> ipiv(naux);

        C_DGESV(naux, 1, Jmet_copy->pointer()[0], naux, ipiv.data(), gamma[i]->pointer()[0], naux);
    }

    // Build Juv = (uv|Q) * gammaQ
    df_cfmm_tree_->df_set_contraction(ContractionType::DF_PRI_AUX);
    df_cfmm_tree_->build_J(ints_, gamma, J, Jmet_max_);

    timer_off("DFCFMM: J");
}

void DFCFMM::print_header() {
    if (print_) {
        outfile->Printf("  ==> CFMM-Accelerated Direct Density Fitted J <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Auxiliary Basis: %11s\n", auxiliary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", df_cfmm_tree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", df_cfmm_tree_->nlevels());
        outfile->Printf("\n");
    }
}

LocalDFJ::LocalDFJ(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, 
               Options& options) : DirectDFJ(primary, auxiliary, options) {

    df_cfmm_tree_ = std::make_shared<CFMMTree>(primary_, auxiliary_, options_);
    molecule_ = primary_->molecule();
    r0_ = options_.get_double("LOCAL_DF_R0");
    r1_ = options_.get_double("LOCAL_DF_R1");

    setup_local_regions();
    build_atomic_inverse();
    rho_a_L_.resize(molecule_->natom());
    I_KX_.resize(molecule_->natom());
}

double bump_function(double x, double r0, double r1) {
    double bump_value;
    if (x <= r0) {
        bump_value = 1.0;
    } else if (x >= r1) {
        bump_value = 0.0;
    } else {
        double temp1 = (r1 - r0) / (r1 - x);
        double temp2 = (r1 - r0) / (x - r0);
        bump_value = 1.0 / (1.0 + std::exp(temp1 - temp2));
    }
    return bump_value;
}

void LocalDFJ::setup_local_regions() {

    unsigned natom = molecule_->natom();
    unsigned pri_nshell = primary_->nshell();
    unsigned aux_nshell = auxiliary_->nshell();
    
    auto aux_shell_pairs = ints_[0]->shell_pairs_bra();
    auto pri_shell_pairs = ints_[0]->shell_pairs_ket();

    atom_to_aux_shells_.resize(natom);
    aux_shell_to_atoms_.resize(aux_nshell);
    atom_aux_shell_bump_value_.resize(natom);
    atom_aux_shell_function_offset_.resize(natom);
    for (int atom = 0; atom < natom; atom++) {
        Vector3 Ratom = molecule_->xyz(atom);
        int offset = 0;
        for (const auto aux_pair : aux_shell_pairs) {
            int Q = aux_pair.first;
            Vector3 R_Q = auxiliary_->shell(Q).center();
            Vector3 dR = Ratom - R_Q;
            double dist2 = dR.dot(dR);
            if (dist2 <= r1_ * r1_) {
                atom_to_aux_shells_[atom].push_back(Q);
                aux_shell_to_atoms_[Q].push_back(atom);
                atom_aux_shell_function_offset_[atom][Q] = offset;
                atom_aux_shell_bump_value_[atom][Q] = bump_function(std::sqrt(dist2), r0_, r1_);
                offset += auxiliary_->shell(Q).nfunction();
            }
        }
    }

    naux_per_atom_.resize(natom);
    for (int atom = 0; atom < natom; atom++) {
        int naux = 0;
        for (const auto& L_a : atom_to_aux_shells_[atom]) {
            naux += auxiliary_->shell(L_a).nfunction();
        }
        naux_per_atom_[atom] = naux;
    }

    atom_to_pri_shells_.resize(natom);
    for (const auto& pri_pair : pri_shell_pairs) {
        int U = pri_pair.first;
        int V = pri_pair.second;

        int Ucenter = primary_->shell(U).ncenter();
        int Vcenter = primary_->shell(V).ncenter();

        atom_to_pri_shells_[Ucenter].push_back(U * pri_nshell + V);
    }
}

void LocalDFJ::build_atomic_inverse() {
    unsigned natom = molecule_->natom();
    unsigned naux = auxiliary_->nbf();
    unsigned aux_nshell = auxiliary_->nshell();

    J_X_.resize(natom);
    for (int atom = 0; atom < natom; atom++) {
        int atomic_naux = naux_per_atom_[atom];
        J_X_[atom] = std::make_shared<Matrix>(atomic_naux, atomic_naux);
        J_X_[atom]->zero();
    }

    double** Jmetp = Jmet_->pointer();

#pragma omp parallel for
    for (int atom = 0; atom < natom; atom++) {
        double** JXp = J_X_[atom]->pointer();

        for (int K : atom_to_aux_shells_[atom]) {
            int k_start = auxiliary_->shell(K).start();
            int num_k = auxiliary_->shell(K).nfunction();
            int k_off = atom_aux_shell_function_offset_[atom][K];
            double bump_k = atom_aux_shell_bump_value_[atom][K];
            int Katom = auxiliary_->shell(K).ncenter();

            for (int L : atom_to_aux_shells_[atom]) {
                int l_start = auxiliary_->shell(L).start();
                int num_l = auxiliary_->shell(L).nfunction();
                int l_off = atom_aux_shell_function_offset_[atom][L];
                double bump_l = atom_aux_shell_bump_value_[atom][L];
                int Latom = auxiliary_->shell(L).ncenter();

                double prefactor = (Katom == Latom) ? 1.0 : bump_k * bump_l;

                for (int dk = 0; dk < num_k; dk++) {
                    for (int dl = 0; dl < num_l; dl++) {
#pragma omp atomic
                        JXp[k_off + dk][l_off + dl] += prefactor * Jmetp[k_start + dk][l_start + dl];
                    }
                }
            }
        }
    }
}

void LocalDFJ::build_rho_a_L(const std::vector<SharedMatrix>& D) {
    unsigned natom = molecule_->natom();
    unsigned pri_nshell = primary_->nshell();
    unsigned aux_nshell = auxiliary_->nshell();

    for (int atom = 0; atom < natom; atom++) {
        bool uninitialized = (rho_a_L_[atom].size() == 0);
        for (int i = 0; i < D.size(); i++) {
            if (uninitialized) rho_a_L_[atom].push_back(std::make_shared<Matrix>(naux_per_atom_[atom], 1));
            rho_a_L_[atom][i]->zero();
        }
    }

    // maximum values of Density matrix for shell pair block UV
    // TODO: Integrate this more smoothly into current density screening framework
    Matrix D_max(pri_nshell, pri_nshell);
    auto D_maxp = D_max.pointer();

    for (size_t U = 0; U < pri_nshell; U++) {
        int u_start = primary_->shell_to_basis_function(U);
        int num_u = primary_->shell(U).nfunction();
	
	    for (size_t V = 0; V < pri_nshell; V++) {
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
 
#pragma omp parallel for
    for (int L = 0; L < aux_nshell; L++) {
        int l_start = auxiliary_->shell(L).start();
        int num_l = auxiliary_->shell(L).nfunction();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (const int& atom : aux_shell_to_atoms_[L]) {
            int l_off = atom_aux_shell_function_offset_[atom][L];

            for (const int& UV_a : atom_to_pri_shells_[atom]) {
                int U = UV_a / pri_nshell;
                int V = UV_a % pri_nshell;

    		    double screen_val = D_maxp[U][V] * D_maxp[U][V] * Jmet_max_[L] * ints_[thread]->shell_pair_value(U,V);
                if (screen_val < cutoff_ * cutoff_) continue;

                double prefactor = (U == V) ? 1.0 : 2.0;

                int u_start = primary_->shell(U).start();
                int num_u = primary_->shell(U).nfunction();

                int v_start = primary_->shell(V).start();
                int num_v = primary_->shell(V).nfunction();

                ints_[thread]->compute_shell(L, 0, U, V);
                const double* buffer = ints_[thread]->buffer();

                for (int i = 0; i < D.size(); i++) {
                    double** Dp = D[i]->pointer();
                    double* raLp = rho_a_L_[atom][i]->pointer()[0];
                    double* Luv = const_cast<double *>(buffer);

                    for (int dl = 0; dl < num_l; dl++) {
                        for (int du = 0; du < num_u; du++) {
                            for (int dv = 0; dv < num_v; dv++) {
#pragma omp atomic
                                raLp[l_off + dl] += prefactor * (*Luv) * Dp[u_start + du][v_start + dv];
                                (Luv)++;
                            }
                        }
                    }
                }
            }
        }
    }
}

void LocalDFJ::build_rho_tilde_K() {
    unsigned natom = molecule_->natom();
    unsigned nmat = rho_a_L_[0].size();
    unsigned naux = auxiliary_->nbf();

    bool uninitialized = (rho_tilde_K_.size() == 0);
    for (int i = 0; i < nmat; i++) {
        if (uninitialized) rho_tilde_K_.push_back(std::make_shared<Matrix>(naux, 1));
        rho_tilde_K_[i]->zero();
    }

    for (int atom = 0; atom < natom; atom++) {
        int atom_naux = naux_per_atom_[atom];

        for (int i = 0; i < nmat; i++) {
            double* raLp = rho_a_L_[atom][i]->pointer()[0];
            double* rtKp = rho_tilde_K_[i]->pointer()[0];

            SharedMatrix JXcopy = J_X_[atom]->clone();
            SharedMatrix rtKbuff = rho_a_L_[atom][i]->clone();
            std::vector<int> ipiv(atom_naux);

            double* JXcp = JXcopy->pointer()[0];
            double* rtKbp = rtKbuff->pointer()[0];

            for (const int& L_a : atom_to_aux_shells_[atom]) {
                int l_off = atom_aux_shell_function_offset_[atom][L_a];
                int l_start = auxiliary_->shell(L_a).start();
                int num_l = auxiliary_->shell(L_a).nfunction();
                double bump_l = atom_aux_shell_bump_value_[atom][L_a];

                for (int dl = 0; dl < num_l; dl++) {
                    rtKbp[l_off + dl] *= bump_l;
                }
            }
            
            C_DGESV(atom_naux, 1, JXcp, atom_naux, ipiv.data(), rtKbp, atom_naux);

            for (const int& K_a : atom_to_aux_shells_[atom]) {
                int k_off = atom_aux_shell_function_offset_[atom][K_a];
                int k_start = auxiliary_->shell(K_a).start();
                int num_k = auxiliary_->shell(K_a).nfunction();
                double bump_k = atom_aux_shell_bump_value_[atom][K_a];
                
                for (int dk = 0; dk < num_k; dk++) {
                    rtKp[k_start + dk] += bump_k * rtKbp[k_off + dk];
                }
            }
        }
    }
}

void LocalDFJ::build_J_tilde_L() {

    bool uninitialized = (J_tilde_L_.size() == 0);
    if (uninitialized) J_tilde_L_.resize(rho_tilde_K_.size());

    // TODO: Technically O(N^2), could make it O(N) with a much larger prefactor with CFMMTree
    // Worth investigating in the future
    for (int i = 0; i < rho_tilde_K_.size(); i++) {
        J_tilde_L_[i] = linalg::doublet(Jmet_, rho_tilde_K_[i]);
    }
}

void LocalDFJ::build_J_L(const std::vector<SharedMatrix>& D) {
    bool uninitialized = (J_L_.size() == 0);

    if (uninitialized) {
        J_L_.resize(D.size());
        for (int i = 0; i < D.size(); i++) {
            J_L_[i] = std::make_shared<Matrix>(auxiliary_->nbf(), 1);
        }
    }

    df_cfmm_tree_->df_set_contraction(ContractionType::DF_AUX_PRI);
    df_cfmm_tree_->build_J(ints_, D, J_L_, Jmet_max_);
}

void LocalDFJ::build_I_KX() {
    unsigned natom = molecule_->natom();
    unsigned naux = auxiliary_->nbf();
    unsigned aux_nshell = auxiliary_->nshell();
    int nmat = J_L_.size();

    std::vector<SharedMatrix> dJ_L;
    dJ_L.resize(nmat);
    for (int i = 0; i < nmat; i++) {
        dJ_L[i] = std::make_shared<Matrix>(naux, 1);
        dJ_L[i]->copy(J_L_[i]);
        dJ_L[i]->subtract(J_tilde_L_[i]);
    }

    bool uninitialized = (I_KX_[0].size() == 0);
    for (int atom = 0; atom < natom; atom++) {
        for (int i = 0; i < nmat; i++) {
            if (uninitialized) I_KX_[atom].push_back(std::make_shared<Matrix>(naux_per_atom_[atom], 1));
            I_KX_[atom][i]->zero();
        }
    }

    for (int atom = 0; atom < natom; atom++) {
        int atom_naux = naux_per_atom_[atom];

        for (int i = 0; i < nmat; i++) {
            double* dJLp = dJ_L[i]->pointer()[0];
            double* IKXp = I_KX_[atom][i]->pointer()[0];
            SharedMatrix JXcopy = J_X_[atom]->clone();
            std::vector<int> ipiv(atom_naux);

            double* JXcp = JXcopy->pointer()[0];
            
            for (const int& L_a : atom_to_aux_shells_[atom]) {
                int l_off = atom_aux_shell_function_offset_[atom][L_a];
                int l_start = auxiliary_->shell(L_a).start();
                int num_l = auxiliary_->shell(L_a).nfunction();
                double bump_l = atom_aux_shell_bump_value_[atom][L_a];

                for (int dl = 0; dl < num_l; dl++) {
                    IKXp[l_off + dl] = bump_l * dJLp[l_start + dl];
                }
            }

            C_DGESV(atom_naux, 1, JXcp, atom_naux, ipiv.data(), IKXp, atom_naux);

            for (const int& K_a : atom_to_aux_shells_[atom]) {
                int k_off = atom_aux_shell_function_offset_[atom][K_a];
                int k_start = auxiliary_->shell(K_a).start();
                int num_k = auxiliary_->shell(K_a).nfunction();
                double bump_k = atom_aux_shell_bump_value_[atom][K_a];
                
                for (int dk = 0; dk < num_k; dk++) {
                    IKXp[k_off + dk] *= bump_k;
                }
            }
        }
    }

}

void LocalDFJ::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("LocalDFJ: J");

    for (auto& Jmat : J) {
        Jmat->zero();
    }

    unsigned nmat = D.size();
    unsigned nbf = primary_->nbf();
    unsigned natom = molecule_->natom();
    unsigned aux_nshell = auxiliary_->nshell();
    unsigned pri_nshell = primary_->nshell();

    build_rho_a_L(D);
    build_rho_tilde_K();
    build_J_tilde_L();
    build_J_L(D);
    build_I_KX();

    // Contraction in Equation 23
    df_cfmm_tree_->df_set_contraction(ContractionType::DF_PRI_AUX);
    df_cfmm_tree_->build_J(ints_, rho_tilde_K_, J, Jmet_max_);

    // set up I_KX_max for screening purposes
    SharedMatrix I_KX_max = std::make_shared<Matrix>(natom, aux_nshell);
    I_KX_max->zero();
    auto I_KX_maxp = I_KX_max->pointer();

    for (size_t atom = 0; atom < natom; atom++) {
        for (const auto K : atom_to_aux_shells_[atom]) { 
            for (size_t i = 0; i < D.size(); i++) {
                double* IKXp = I_KX_[atom][i]->pointer()[0];
            	
		        int k_off = atom_aux_shell_function_offset_[atom][K];
                int num_k = auxiliary_->shell(K).nfunction();
                for (int k = k_off; k < k_off + num_k; k++) {
                    I_KX_maxp[atom][K] = std::max(I_KX_maxp[atom][K], std::abs(IKXp[k]));
                }
            }
        }
    }
 
    // Contraction in Equation 25
#pragma omp parallel for
    for (int K = 0; K < aux_nshell; K++) {
        int k_start = auxiliary_->shell(K).start();
        int num_k = auxiliary_->shell(K).nfunction();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (const int& atom : aux_shell_to_atoms_[K]) {
            int k_off = atom_aux_shell_function_offset_[atom][K];

            for (const int& UV_a : atom_to_pri_shells_[atom]) {
                int U = UV_a / pri_nshell;
                int V = UV_a % pri_nshell;

    		double screen_val = I_KX_maxp[atom][K] * I_KX_maxp[atom][K] * Jmet_max_[K] * ints_[thread]->shell_pair_value(U,V);
		    if (screen_val < cutoff_ * cutoff_) continue;

                int u_start = primary_->shell(U).start();
                int num_u = primary_->shell(U).nfunction();

                int v_start = primary_->shell(V).start();
                int num_v = primary_->shell(V).nfunction();

                int prefactor = (U == V) ? 1.0 : 2.0;

                ints_[thread]->compute_shell(K, 0, U, V);
                const double *buffer = ints_[thread]->buffer();

                for (int i = 0; i < nmat; i++) {
                    double* Kuvp = const_cast<double *>(buffer);
                    double* IKXp = I_KX_[atom][i]->pointer()[0];
                    double** Jp = J[i]->pointer();

                    for (int du = 0; du < num_u; du++) {
                        for (int dv = 0; dv < num_v; dv++) {
                            for (int dk = 0; dk < num_k; dk++) {
#pragma omp atomic
                                Jp[u_start + du][v_start + dv] += prefactor * 
                                    Kuvp[dk * num_u * num_v + du * num_v + dv] * IKXp[k_off + dk];
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto& Jmat : J) {
        Jmat->hermitivitize();
    }

    timer_off("LocalDFJ: J");

}

void LocalDFJ::print_header() {
    if (print_) {
        outfile->Printf("  ==> Direct Local Density Fitted J (with CFMM) <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Auxiliary Basis: %11s\n", auxiliary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", df_cfmm_tree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", df_cfmm_tree_->nlevels());
        outfile->Printf("\n");
    }
}

LocalDFK::LocalDFK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options) : LocalDFJ(primary, auxiliary, options) {};

void LocalDFK::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) {

    timer_on("LocalDFK: K");

    for (auto& Kmat : K) {
        Kmat->zero();
    }

    int nmat = D.size();
    int nbf = primary_->nbf();
    int nshell = primary_->nshell();
    int natom = molecule_->natom();

    std::vector<std::unordered_set<int>> S_junction(nshell);
    std::vector<std::unordered_set<int>> D_junction(nshell);
    std::unordered_map<int, std::unordered_set<int>> DS_junction;

    // Set up overlap sparsity information
    const auto& shell_pairs = ints_[0]->shell_pairs_ket();
    int num_shell_pairs = shell_pairs.size();

    for (int UVidx = 0; UVidx < shell_pairs.size(); UVidx++) {
        int U = shell_pairs[UVidx].first;
        int V = shell_pairs[UVidx].second;
        S_junction[U].emplace(V);
        S_junction[V].emplace(U);
    }

    // Used for density screening
    std::vector<std::vector<double>> shell_pair_max_density(nmat, std::vector<double>(nshell * nshell));

#pragma omp parallel for
    for (int M = 0; M < nshell; M++) {
        for (int N = M; N < nshell; N++) {
            int m_start = primary_->shell(M).function_index();
            int num_m = primary_->shell(M).nfunction();

            int n_start = primary_->shell(N).function_index();
            int num_n = primary_->shell(N).nfunction();

            for (int i = 0; i < D.size(); i++) {
                double** Dp = D[i]->pointer();
                double max_dens = 0.0;
                for (int m = m_start; m < m_start + num_m; m++) {
                    for (int n = n_start; n < n_start + num_n; n++) {
                        max_dens = std::max(max_dens, std::abs(Dp[m][n]));
                    }
                }
                shell_pair_max_density[i][M * nshell + N] = max_dens;
                if (M != N) shell_pair_max_density[i][N * nshell + M] = max_dens;
            }
        }
    }

    // Set up density sparsity information
    IntegralFactory factory(primary_);
    auto eri = std::shared_ptr<TwoBodyAOInt>(factory.eri());

    // => Calculate Shell Ceilings <= //
    std::vector<double> shell_ceilings(nshell, 0.0);

    // sqrt(Umax|Umax) in Ochsenfeld Eq. 3
#pragma omp parallel for
    for (int U = 0; U < nshell; U++) {
        for (int V = 0; V <= U; V++) {
            double val = std::sqrt(eri->shell_ceiling2(U, V, U, V));
            shell_ceilings[U] = std::max(shell_ceilings[U], val);
#pragma omp critical
            shell_ceilings[V] = std::max(shell_ceilings[V], val);
        }
    }

#pragma omp parallel for
    for (int U = 0; U < nshell; U++) {
        for (int V = 0; V < nshell; V++) {

            double UVmax = 0.0;
            for (int i = 0; i < nmat; i++) {
                UVmax = std::max(UVmax, shell_pair_max_density[i][U * nshell + V]);
            }

            if (shell_ceilings[U] * shell_ceilings[V] * UVmax >= cutoff_) {
                D_junction[U].emplace(V);
            }
        }
    }

    // Form DS junction
#pragma omp parallel for
    for (int M = 0; M < nshell; M++) {
        for (const int& L : S_junction[M]) {
            if (!DS_junction.count(M * nshell + L)) DS_junction[M * nshell + L] = std::unordered_set<int>();
            for (const int& S : D_junction[L]) {
                for (const int& N : S_junction[S]) {
                    DS_junction[M * nshell + L].emplace(N);
                }
            }
        }
    }
    

    std::vector<std::unordered_set<int>> atom_overlap_shells(natom); // MU LAMBDA
    std::vector<std::unordered_set<int>> atom_density_shells(natom); // NU SIGMA
    std::vector<std::unordered_set<int>> atom_braket_shells(natom); // NU LAMBDA

    std::vector<int> overlap_pair_nfunc(natom);
    std::vector<int> density_pair_nfunc(natom);
    std::vector<int> braket_pair_nfunc(natom);

    std::vector<std::unordered_map<int, int>> atom_overlap_shell_function_offset(natom);
    std::vector<std::unordered_map<int, int>> atom_density_shell_function_offset(natom);
    std::vector<std::unordered_map<int, int>> atom_braket_shell_function_offset(natom);

    std::vector<std::unordered_map<int, std::unordered_set<int>>> atom_S_L_partners(natom); // SIGMA to LAMBDA

#pragma omp parallel for
    for (int atom = 0; atom < natom; atom++) {
        int M_start = primary_->shell_on_center(atom, 0);
        int num_M = primary_->nshell_on_center(atom);
        for (int M = M_start; M < M_start + num_M; M++) {
            for (const int& L : S_junction[M]) {
                atom_overlap_shells[atom].emplace(M * nshell + L);
                for (const int& S : D_junction[L]) {
                    if (!atom_S_L_partners[atom].count(S)) atom_S_L_partners[atom][S] = std::unordered_set<int>();
                    atom_S_L_partners[atom][S].emplace(L);
                    for (const int& N : S_junction[S]) {
                        atom_density_shells[atom].emplace(N * nshell + S);
                        atom_braket_shells[atom].emplace(N * nshell + L);
                    } // end N
                } // end S
            } // end L
        } // end U

        size_t ml_offset = 0;
        for (const int& ML : atom_overlap_shells[atom]) {
            int M = ML / nshell;
            int L = ML % nshell;
            int num_m = primary_->shell(M).nfunction();
            int num_l = primary_->shell(L).nfunction();

            atom_overlap_shell_function_offset[atom][ML] = ml_offset;
            ml_offset += num_m * num_l;
        }
        overlap_pair_nfunc[atom] = ml_offset;

        size_t ns_offset = 0;
        for (const int& NS : atom_density_shells[atom]) {
            int N = NS / nshell;
            int S = NS % nshell;
            int num_n = primary_->shell(N).nfunction();
            int num_s = primary_->shell(S).nfunction();

            atom_density_shell_function_offset[atom][NS] = ns_offset;
            ns_offset += num_n * num_s;
        }
        density_pair_nfunc[atom] = ns_offset;

        size_t nl_offset = 0;
        for (const int& NL : atom_braket_shells[atom]) {
            int N = NL / nshell;
            int L = NL % nshell;
            int num_n = primary_->shell(N).nfunction();
            int num_l = primary_->shell(L).nfunction();

            atom_braket_shell_function_offset[atom][NL] = nl_offset;
            nl_offset += num_n * num_l;
        }
        braket_pair_nfunc[atom] = nl_offset;
    } // end atom

    // Create K buffers to help with parallel efficiency
    size_t max_nbf_per_shell = 0;
    for (int M = 0; M < nshell; M++) {
        max_nbf_per_shell = std::max(max_nbf_per_shell, (size_t)primary_->shell(M).nfunction());
    }
    std::vector<SharedMatrix> Kbuff(nthread_);

#pragma omp parallel for
    for (int thread = 0; thread < nthread_; thread++) {
        Kbuff[thread] = std::make_shared<Matrix>(max_nbf_per_shell, max_nbf_per_shell);
        Kbuff[thread]->zero();
    }

    for (int atom = 0; atom < natom; atom++) {
        int atom_naux = naux_per_atom_[atom];
        int atom_num_overlap = overlap_pair_nfunc[atom];
        int atom_num_density = density_pair_nfunc[atom];
        int atom_num_braket = braket_pair_nfunc[atom];

        timer_on("LocalDFK: Form I1");

        // => Build I1 (overlap three-center intermediate) <= //
        SharedMatrix I1 = std::make_shared<Matrix>(atom_naux * atom_num_overlap, 1);
        I1->zero();

        std::vector<int> overlap_tasks;
        overlap_tasks.insert(overlap_tasks.end(), atom_overlap_shells[atom].begin(), atom_overlap_shells[atom].end());

#pragma omp parallel for schedule(dynamic)
        for (int task = 0; task < overlap_tasks.size(); task++) {
            int ML = overlap_tasks[task];
            int M = ML / nshell;
            int L = ML % nshell;

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            int num_m = primary_->shell(M).nfunction();
            int num_l = primary_->shell(L).nfunction();

            int ml_off = atom_overlap_shell_function_offset[atom][M * nshell + L];

            for (const int& P : atom_to_aux_shells_[atom]) {
                int p_off = atom_aux_shell_function_offset_[atom][P];
                int num_p = auxiliary_->shell(P).nfunction();
                double p_bump = atom_aux_shell_bump_value_[atom][P];

                ints_[thread]->compute_shell(P, 0, M, L);
                double* Pml = const_cast<double *>(ints_[thread]->buffer());

                double* I1p = I1->pointer()[0];

                for (int dp = 0; dp < num_p; dp++) {
                    for (int dm = 0; dm < num_m; dm++) {
                        for (int dl = 0; dl < num_l; dl++) {
                            I1p[(ml_off + dm * num_l + dl) * atom_naux + (p_off + dp)] = p_bump * (*Pml);
                            Pml++;
                        } // end dl
                    } // end dm
                } // end dp
            }
        }

        // => Linear solve I1 (transform by atomic auxiliary matrix) <= //
        SharedMatrix JXcopy = J_X_[atom]->clone();
        double* JXcp = JXcopy->pointer()[0];
        double* I1p = I1->pointer()[0];

        std::vector<int> ipiv(atom_naux);
        C_DGESV(atom_naux, atom_num_overlap, JXcp, atom_naux, ipiv.data(), I1p, atom_naux);

        timer_off("LocalDFK: Form I1");

        timer_on("LocalDFK: Form I2");

        // => Build I2 (density three-center intermediate) <= //
        SharedMatrix I2 = std::make_shared<Matrix>(atom_naux * atom_num_density, 1);
        I2->zero();

        std::vector<int> density_tasks;
        density_tasks.insert(density_tasks.end(), atom_density_shells[atom].begin(), atom_density_shells[atom].end());

#pragma omp parallel for schedule(dynamic)
        for (int task = 0; task < density_tasks.size(); task++) {
            int NS = density_tasks[task];
            int N = NS / nshell;
            int S = NS % nshell;

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            int num_n = primary_->shell(N).nfunction();
            int num_s = primary_->shell(S).nfunction();
            int s_start = primary_->shell(S).start();

            int ns_off = atom_density_shell_function_offset[atom][N * nshell + S];

            for (const int& Q : atom_to_aux_shells_[atom]) {
                int q_off = atom_aux_shell_function_offset_[atom][Q];
                int num_q = auxiliary_->shell(Q).nfunction();
                double q_bump = atom_aux_shell_bump_value_[atom][Q];

                ints_[thread]->compute_shell(Q, 0, N, S);
                double* Qns = const_cast<double *>(ints_[thread]->buffer());

                double* I2p = I2->pointer()[0];

                for (int dq = 0; dq < num_q; dq++) {
                    for (int dn = 0; dn < num_n; dn++) {
                        for (int ds = 0; ds < num_s; ds++) {
                            I2p[(ns_off + dn * num_s + ds) * atom_naux + (q_off + dq)] = q_bump * (*Qns);
                            (Qns)++;
                        }
                    }
                }
            }
        }
        
        timer_off("LocalDFK: Form I2");

        timer_on("LocalDFK: Form K");

#pragma omp parallel for schedule(dynamic)
        for (int task = 0; task < overlap_tasks.size(); task++) {
            int ML = overlap_tasks[task];
            int M = ML / nshell;
            int L = ML % nshell;

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            int m_start = primary_->shell(M).start();
            int l_start = primary_->shell(L).start();

            int num_m = primary_->shell(M).nfunction();
            int num_l = primary_->shell(L).nfunction();

            int ml_off = atom_overlap_shell_function_offset[atom][M * nshell + L];

            for (const int& S : D_junction[L]) {
                int s_start = primary_->shell(S).start();
                int num_s = primary_->shell(S).nfunction();
                
                for (const int& N : S_junction[S]) {
                    int n_start = primary_->shell(N).start();
                    int num_n = primary_->shell(N).nfunction();

                    int ns_off = atom_density_shell_function_offset[atom][N * nshell + S];

                    double** Dp = D[0]->pointer();
                    double** KBp = Kbuff[thread]->pointer();
                    double* I1p = I1->pointer()[0];
                    double* I2p = I2->pointer()[0];

                    for (int dm = 0; dm < num_m; dm++) {
                        for (int dn = 0; dn < num_n; dn++) {
                            for (int ds = 0; ds < num_s; ds++) {
                                for (int dl = 0; dl < num_l; dl++) {
                                    for (int aq = 0; aq < atom_naux; aq++) {
                                        KBp[dm][dn] += I2p[(ns_off + dn * num_s + ds) * atom_naux + (aq)] *
                                                        I1p[(ml_off + dm * num_l + dl) * atom_naux + (aq)] * Dp[l_start+dl][s_start+ds];
                                    }
                                }
                            }
                        }
                    }

                    double** Kp = K[0]->pointer();
                    // Flush the toilet (for parallel efficiency)
                    for (int dm = 0; dm < num_m; dm++) {
                        for (int dn = 0; dn < num_n; dn++) {
#pragma omp atomic
                            Kp[m_start + dm][n_start + dn] += KBp[dm][dn];
                        }
                    }
                    Kbuff[thread]->zero();
                }
            }
        }
        timer_off("LocalDFK: Form K");
    }

    for (auto& Kmat : K) {
        Kmat->hermitivitize();
    }

    timer_off("LocalDFK: K");
}

void LocalDFK::print_header() {
    if (print_) {
        outfile->Printf("  ==> Direct Local Density Fitted K <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Auxiliary Basis: %11s\n", auxiliary_->name().c_str());
        outfile->Printf("\n");
    }
}

}
