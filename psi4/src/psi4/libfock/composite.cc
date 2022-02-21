#include "composite.h"
#include "jk.h"

#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libfmm/fmm_tree.h"
#include "psi4/libqt/qt.h"

#include <vector>
#include <unordered_set>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

using namespace psi;

namespace psi {

void C_DGESV_wrapper(SharedMatrix A, SharedMatrix B) {
    int N = B->rowspi(0);
    int M = B->colspi(0);
    if (N == 0 || M == 0) return;

    // create a copy of B in fortran ordering
    std::vector<double> B_fortran(N * M, 0.0);
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            B_fortran[m * N + n] = B->get(n, m);
        }
    }

    // make the C_DGESV call, solving AX=B for X
    std::vector<int> ipiv(N);
    int errcode = C_DGESV(N, M, A->pointer()[0], N, ipiv.data(), B_fortran.data(), N);

    // copy the fortran-ordered X into the original matrix, reverting to C-ordering
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            B->set(n, m, B_fortran[m * N + n]);
        }
    }
}

JBase::JBase(std::shared_ptr<BasisSet> primary, Options& options) 
            : primary_(primary), options_(options) {
    nthread_ = 1;

#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif
}

KBase::KBase(std::shared_ptr<BasisSet> primary, Options& options) 
            : primary_(primary), options_(options) {
    nthread_ = 1;

#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif
}

CompositeJK::CompositeJK(std::shared_ptr<BasisSet> primary, Options& options) : DirectJK(primary, options) {
    common_init();
}

CompositeJK::CompositeJK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options)
                            : DirectJK(primary, options), auxiliary_(auxiliary) {
    common_init();
}

void CompositeJK::common_init() {

    // Composite JK code
    jtype_ = options_.get_str("J_TYPE");
    ktype_ = options_.get_str("K_TYPE");

    nthread_ = df_ints_num_threads_;

    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }

    if (jtype_ == "DIRECT_DF") {
        jalgo_ = std::make_shared<DirectDFJ>(primary_, auxiliary_, options_);
    } else if (jtype_ == "CFMM") {
        jalgo_ = std::make_shared<CFMM>(primary_, options_);
    } else if (jtype_ == "DIRECT") {
        jalgo_ = nullptr;
    } else {
        throw PSIEXCEPTION("J_TYPE IS NOT SUPPORTED AS A COMPOSITE METHOD");
    }

    if (ktype_ == "LINK") {
        kalgo_ = std::make_shared<LinK>(primary_, options_);
    } else if (ktype_ == "COSK") {
        kalgo_ = std::make_shared<COSK>(primary_, options_);
    } else if (ktype_ == "DIRECT") {
        kalgo_ = nullptr;
    } else {
        throw PSIEXCEPTION("K_TYPE IS NOT SUPPORTED AS A COMPOSITE METHOD");
    }

    initial_iteration_ = true;
}

void CompositeJK::print_header() const {
    std::string screen_type = options_.get_str("SCREENING");
    if (print_) {
        outfile->Printf("  ==> CompositeJK: Mix/Match JK Builds <==\n\n");
        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        if (do_J_) outfile->Printf("    J Algorithm:       %11s\n", jtype_.c_str());
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        if (do_K_) outfile->Printf("    K Algorithm:       %11s\n", ktype_.c_str());
        outfile->Printf("    Integrals threads: %11d\n", df_ints_num_threads_);
        outfile->Printf("    Screening Type:    %11s\n", screen_type.c_str());
        outfile->Printf("    Screening Cutoff:  %11.0E\n\n", cutoff_);
    }
}

void CompositeJK::compute_JK() {

    if (incfock_) {
        timer_on("CompositeJK: INCFOCK Preprocessing");
        incfock_setup();
        int reset = options_.get_int("INCFOCK_FULL_FOCK_EVERY");
        double dconv = options_.get_double("D_CONVERGENCE");
        double Dnorm = Process::environment.globals["SCF D NORM"];
        // Do IFB on this iteration?
        do_incfock_iter_ = (Dnorm >= dconv) && !initial_iteration_ && (incfock_count_ % reset != reset - 1);
        
        if (!initial_iteration_ && (Dnorm >= dconv)) incfock_count_ += 1;
        timer_off("CompositeJK: INCFOCK Preprocessing");
    }

    // Passed in as a dummy when J (and/or K) is not built
    std::vector<SharedMatrix> temp;

    std::vector<SharedMatrix>& D_ref = (do_incfock_iter_ ? delta_D_ao_ : D_ao_);
    std::vector<SharedMatrix>& J_ref = do_J_ ? (do_incfock_iter_ ? delta_J_ao_ : J_ao_) : temp;
    std::vector<SharedMatrix>& K_ref = do_K_ ? (do_incfock_iter_ ? delta_K_ao_ : K_ao_) : temp;

    // Update Densities for each integral object
    for (int thread = 0; thread < nthread_; thread++) {
        ints_[thread]->update_density(D_ref);
    }

    /*
    // Do NOT do any weird stuff for the SAD guess :)
    if (initial_iteration_) build_JK_matrices(ints_, D_ref, J_ref, K_ref);
    {
        if (jtype_ != "DIRECT_DF") build_JK_matrices(ints_, D_ref, J_ref, K_ref);
        else {
            jalgo_->build_J(D_ref, J_ref);
            build_JK_matrices(ints_, D_ref, temp, K_ref);
        }
    }
    */


    
    if (!jalgo_ && !kalgo_) build_JK_matrices(ints_, D_ref, J_ref, K_ref);
    else {
        if (!jalgo_) build_JK_matrices(ints_, D_ref, J_ref, temp);
        else jalgo_->build_J(D_ref, J_ref);

        if (!kalgo_) build_JK_matrices(ints_, D_ref, temp, K_ref);
        else kalgo_->build_K(D_ref, K_ref);
    }

    if (incfock_) {
        timer_on("CompositeJK: INCFOCK Postprocessing");
        incfock_postiter();
        timer_off("CompositeJK: INCFOCK Postprocessing");
    }

    if (initial_iteration_) initial_iteration_ = false;
}

DirectDFJ::DirectDFJ(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options)
                        : JBase(primary, options), auxiliary_(auxiliary) {
    form_Jinv();
    build_ints();
}

void DirectDFJ::form_Jinv() {
    timer_on("DirectDFJ: Build Metric");

    auto metric = std::make_shared<FittingMetric>(auxiliary_, true);
    metric->form_fitting_metric();
    Jmet_ = metric->get_metric();

    timer_off("DirectDFJ: Build Metric");
}

void DirectDFJ::build_ints() {
    auto zero = BasisSet::zero_ao_basis_set();
    auto rifactory = std::make_shared<IntegralFactory>(auxiliary_, zero, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(rifactory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }
}

void DirectDFJ::build_J(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {
    
    timer_on("DirectDFJ::build_J()");

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

    std::vector<std::vector<int>> shell_partners(pri_nshell);
    for (const auto& pair : shell_pairs) {
        int U = pair.first;
        int V = pair.second;
        shell_partners[U].push_back(V);
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
        std::vector<SharedVector> gp2;
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

#pragma omp parallel for num_threads(nthread_) schedule(dynamic)
    for (int U = 0; U < pri_nshell; U++) {

        int u_start = primary_->shell_to_basis_function(U);
        int num_u = primary_->shell(U).nfunction();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (const int V : shell_partners[U]) {

            int v_start = primary_->shell_to_basis_function(V);
            int num_v = primary_->shell(V).nfunction();

            double prefactor = 2.0;
            if (U == V) prefactor *= 0.5;

            for (int Q = 0; Q < aux_nshell; Q++) {

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
    }

    for (auto& Jmat : J) {
        Jmat->hermitivitize();
    }

    timer_off("DirectDFJ::build_J()");
}

CFMM::CFMM(std::shared_ptr<BasisSet> primary, Options& options) : JBase(primary, options) {
    build_ints();
}

void CFMM::build_ints() {
    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }
}

void CFMM::build_J(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("CFMM::build_J()");

    // => Update the Density (for density screening the near-field) <= //
    for (int thread = 0; thread < nthread_; thread++) {
        ints_[thread]->update_density(D);
    }

    auto tree = std::make_shared<CFMMTree>(ints_, D, J, options_);
    tree->build_J();

    timer_off("CFMM::build_J()");
}

LinK::LinK(std::shared_ptr<BasisSet> primary, Options& options)
                        : KBase(primary, options) {
    cutoff_ = options.get_double("INTS_TOLERANCE");
    linK_ints_cutoff_ = options.get_double("LINK_INTS_TOLERANCE");
    build_ints();
}

void LinK::build_ints() {
    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }
}

void LinK::build_K(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) {

    timer_on("LinK::build_K()");

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

    // ==> Prep Atom Pairs (Better for parallel performance than shell pairs) <== //

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
    for (int P = 0; P < nshell; P++) {
        for (int Q = 0; Q <= P; Q++) {
            double val = std::sqrt(ints_[0]->shell_ceiling2(P, Q, P, Q));
            shell_ceilings[P] = std::max(shell_ceilings[P], val);
            shell_ceilings[Q] = std::max(shell_ceilings[Q], val);
        }
    }

    std::vector<std::vector<int>> significant_kets(nshell);

    // => Use shell ceilings to compute significant ket-shells for each bra-shell <= //
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

                // => "Formation of Significant Shell Pair List ML" <= //

                // Significant ket shell pairs RS for bra shell pair PQ
                // represents the merge of ML_P and ML_Q as defined in Oschenfeld
                // Unordered set structure allows for automatic merging as new elements are added
                std::unordered_set<int> ML_PQ;

                // Form ML_P inside ML_PQ
                for (const int R : significant_kets[P]) {
                    int count = 0;
                    for (const int S : significant_bras[R]) {
                        double screen_val = ints_[0]->shell_pair_max_density(P, R) * std::sqrt(ints_[0]->shell_ceiling2(P, Q, R, S));

                        if (screen_val >= linK_ints_cutoff_) {
                            count += 1;
                            int RS = (R >= S) ? (R * nshell + S) : (S * nshell + R);
                            if (RS > P * nshell + Q) continue;
                            ML_PQ.emplace(RS);
                            Q_stripeout_list[dQ].emplace(S);
                        }
                        else break;
                    }
                    if (count == 0) break;
                }

                // Form ML_Q inside ML_PQ
                for (const int R : significant_kets[Q]) {
                    int count = 0;
                    for (const int S : significant_bras[R]) {
                        double screen_val = ints_[0]->shell_pair_max_density(Q, R) * std::sqrt(ints_[0]->shell_ceiling2(P, Q, R, S));

                        if (screen_val >= linK_ints_cutoff_) {
                            count += 1;
                            int RS = (R >= S) ? (R * nshell + S) : (S * nshell + R);
                            if (RS > P * nshell + Q) continue;
                            ML_PQ.emplace(RS);
                            P_stripeout_list[dP].emplace(S);
                        }
                        else break;
                    }
                    if (count == 0) break;
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

    timer_off("LinK::build_K()");
}

COSK::COSK(std::shared_ptr<BasisSet> primary, Options& options)
                        : KBase(primary, options) {
    schwarz_cutoff_ = options.get_double("COSK_S_TOLERANCE");
    density_cutoff_ = options.get_double("COSK_D_TOLERANCE");
    build_ints();
    grid_setup();
}

void COSK::build_ints() {
    auto factory = std::make_shared<IntegralFactory>(primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    for (int thread = 0; thread < nthread_; thread++) {
        grid_ints_.push_back((std::shared_ptr<PotentialInt>) (PotentialInt *)factory->ao_potential());
    }
}

void COSK::grid_setup() {

    timer_on("COSK::grid_setup()");

    // COSK Grid Options
    std::map<std::string, int> cosk_grid_options_int;
    std::map<std::string, std::string> cosk_grid_options_str;

    cosk_grid_options_int["DFT_RADIAL_POINTS"] = options_.get_int("COS_RADIAL_POINTS");
    cosk_grid_options_int["DFT_SPHERICAL_POINTS"] = options_.get_int("COS_SPHERICAL_POINTS");
    cosk_grid_options_str["DFT_PRUNING_SCHEME"] = options_.get_str("COS_PRUNING_SCHEME");

    auto mol = primary_->molecule();
    grid_ = std::make_shared<DFTGrid>(mol, primary_, cosk_grid_options_int, cosk_grid_options_str, options_);

    int nbf = primary_->nbf();
    int nshell = primary_->nshell();
    size_t npoints = grid_->npoints();

    phi_values_.resize(npoints * nbf);
    X_.resize(npoints * nbf);
    shell_to_grid_blocks_.resize(nshell);

    // Form PHI and X
    auto blocks = grid_->blocks();
    size_t nblock = blocks.size();

    size_t point = 0;
    for (size_t b = 0; b < nblock; b++) {
        auto block = blocks[b];
        int local_nbf = block->local_nbf();
        size_t local_npoints = block->npoints();

        // Add the current point to the grid point offset for the block
        block_to_grid_point_.push_back(point);

        // Set up grid blocks to shell information
        for (const auto& shell : block->shells_local_to_global()) {
            shell_to_grid_blocks_[shell].emplace(b);
        }

        double *x = block->x();
        double *y = block->y();
        double *z = block->z();
        double *w = block->w();

        for (size_t p = 0; p < local_npoints; p++) {
            primary_->compute_phi(&phi_values_[point * nbf], x[p], y[p], z[p]);
            for (int u = 0; u < nbf; u++) {
                X_[point * nbf + u] = w[p] * phi_values_[point * nbf + u];
            }
            point += 1;
        }
    }

    timer_off("COSK::grid_setup()");
}

void COSK::build_K(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) {
    timer_on("COSK::build_K()");

    for (auto& Kmat : K) {
        Kmat->zero();
    }

    ints_[0]->update_density(D);

    int nbf = primary_->nbf();
    int nshell = primary_->nshell();

    // Shells linked to each other through Schwarz Screening (Significant Overlap)
    std::vector<std::vector<int>> S_junction(nshell);
    double max_integral = ints_[0]->max_integral();

    for (size_t P = 0; P < nshell; P++) {
        for (size_t Q = 0; Q < nshell; Q++) {
            double pq_pq = std::sqrt(ints_[0]->shell_ceiling2(P, Q, P, Q));
            double schwarz_value = std::sqrt(pq_pq * max_integral);
            if (schwarz_value >= schwarz_cutoff_) {
                S_junction[P].push_back(Q);
            }
        }
    }

    // => Calculate Shell Ceilings (To find significant bra-ket pairs)
    // sqrt(Umax|Umax) in Oschenfeld Eq. 3
    std::vector<double> shell_ceilings(nshell, 0.0);
    for (int P = 0; P < nshell; P++) {
        for (int Q = 0; Q <= P; Q++) {
            double val = std::sqrt(ints_[0]->shell_ceiling2(P, Q, P, Q));
            shell_ceilings[P] = std::max(shell_ceilings[P], val);
            shell_ceilings[Q] = std::max(shell_ceilings[Q], val);
        }
    }

    // Shells linked to each other through the Density Matrix
    std::vector<std::vector<int>> D_junction(nshell);

    for (size_t P = 0; P < nshell; P++) {
        for (size_t R = 0; R < nshell; R++) {
            double density_val = shell_ceilings[P] * shell_ceilings[R] * ints_[0]->shell_pair_max_density(P, R);
            if (density_val >= density_cutoff_) {
                D_junction[P].push_back(R);
            }
        }
    }

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    int npoints = grid_->npoints();
    auto blocks = grid_->blocks();
    double* xpoints = grid_->x();
    double* ypoints = grid_->y();
    double* zpoints = grid_->z();

    // Compute A, X, F, and G
    for (size_t i = 0; i < D.size(); i++) {
        double** Dp = D[i]->pointer();
        double** Kp = K[i]->pointer();

        std::vector<double> Ftg(nbf * npoints, 0.0); // Izsak Eq. 6
        std::vector<double> Gvg(nbf * npoints, 0.0); // Izsak Eq. 7

        // Compute F (Equation 6)
#pragma omp parallel for
        for (int M = 0; M < nshell; M++) {
            const GaussianShell& m_shell = primary_->shell(M);
            size_t m_start = m_shell.start();
            size_t num_m = m_shell.nfunction();
            for (int ind = 0; ind < D_junction[M].size(); ind++) {
                int N = D_junction[M][ind];
                const GaussianShell& n_shell = primary_->shell(N);
                size_t n_start = n_shell.start();
                size_t num_n = n_shell.nfunction();

                for (size_t b = 0; b < blocks.size(); b++) {
                    if (!(shell_to_grid_blocks_[M].count(b)) || !(shell_to_grid_blocks_[N].count(b))) continue;
                    auto block = blocks[b];
                    size_t block_start = block_to_grid_point_[b];
                    size_t block_npoints = block->npoints();

                    /*
                    for (size_t g = block_start; g < block_start + block_npoints; g++) {
                        for (int m = m_start; m < m_start + num_m; m++) {
                            for (int n = n_start; n < n_start + num_n; n++) {
                                Ftg[m * npoints + g] += Dp[m][n] * X_[g * nbf + n];
                            }
                        }
                    }
                    */
                    char transa = 'N';
                    char transb = 'T';
                    int m = num_m;
                    int n = block_npoints;
                    int k = num_n;
                    double* Abuff = &(Dp[m_start][n_start]);
                    double* Bbuff = &(X_[block_start * nbf + n_start]);
                    double* Cbuff = &(Ftg[m_start * npoints + block_start]);
                    int lda = nbf;
                    int ldb = nbf;
                    int ldc = npoints;

                    C_DGEMM(transa, transb, m, n, k, 1.0, Abuff, lda, Bbuff, ldb, 1.0, Cbuff, ldc);
                }
            }
        }

        // Compute G (Equation 7)
        size_t computed_shells = 0L;

#pragma omp parallel for
        for (size_t M = 0L; M < nshell; M++) {
            const GaussianShell& m_shell = primary_->shell(M);
            size_t m_start = m_shell.start();
            size_t num_m = m_shell.nfunction();

            int thread = 0;

#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif

            for (size_t ind = 0; ind < S_junction[M].size(); ind++) {
                size_t N = S_junction[M][ind];

                if (N > M) continue;
                double prefactor = 1.0;
                if (N == M) prefactor *= 0.5;

                const GaussianShell& n_shell = primary_->shell(N);
                size_t n_start = n_shell.start();
                size_t num_n = n_shell.nfunction();

                for (size_t b = 0; b < blocks.size(); b++) {
                    if (!(shell_to_grid_blocks_[M].count(b)) || !(shell_to_grid_blocks_[N].count(b))) continue;
                    auto block = blocks[b];
                    size_t block_start = block_to_grid_point_[b];
                    size_t block_npoints = block->npoints();

                    for (size_t g = block_start; g < block_start + block_npoints; g++) {
                        grid_ints_[thread]->set_charge_field({std::make_pair(-1.0, std::array<double, 3>{xpoints[g], ypoints[g], zpoints[g]})});
                        grid_ints_[thread]->compute_shell(M, N);
                        computed_shells++;

                        double* Abuff = (double *) grid_ints_[thread]->buffers()[0];

                        for (size_t m = m_start; m < m_start + num_m; m++) {
                            int dm = m - m_start;
                            for (size_t n = n_start; n < n_start + num_n; n++) {
                                int dn = n - n_start;
                                Gvg[m * npoints + g] += prefactor * Abuff[dm * num_n + dn] * Ftg[n * npoints + g];
#pragma omp atomic
                                Gvg[n * npoints + g] += prefactor * Abuff[dm * num_n + dn] * Ftg[m * npoints + g];
                            }
                        }

                       /*
                        double* Fbuff = &(Ftg[n_start * npoints + g]);
                        double* Gbuff = &(Gvg[m_start * npoints + g]);
                        C_DGEMV('N', num_m, num_n, prefactor, Abuff, num_n, Fbuff, npoints, 1.0, Gbuff, npoints);

                        Fbuff = &(Ftg[m_start * npoints + g]);
                        Gbuff = &(Gvg[n_start * npoints + g]);
                        C_DGEMV('T', num_n, num_m, prefactor, Abuff, num_m, Fbuff, npoints, 1.0, Gbuff, npoints);
                        */

                    }
                }
            }
        }

        // Compute K Matrix (Equation 8)
#pragma omp parallel for
        for (int M = 0; M < nshell; M++) {
            const GaussianShell& m_shell = primary_->shell(M);
            size_t m_start = m_shell.start();
            size_t num_m = m_shell.nfunction();
            for (int N = 0; N < nshell; N++) {
                const GaussianShell& n_shell = primary_->shell(N);
                size_t n_start = n_shell.start();
                size_t num_n = n_shell.nfunction();

                for (size_t b = 0; b < blocks.size(); b++) {
                    if (!(shell_to_grid_blocks_[M].count(b)) || !(shell_to_grid_blocks_[N].count(b))) continue;
                    auto block = blocks[b];
                    size_t block_start = block_to_grid_point_[b];
                    size_t block_npoints = block->npoints();

                    /*
                    for (size_t g = block_start; g < block_start + block_npoints; g++) {
                        for (int m = m_start; m < m_start + num_m; m++) {
                            for (int n = n_start; n < n_start + num_n; n++) {
                                Kp[m][n] += phi_values_[g * nbf + m] * Gvg[n * npoints + g];
                            }
                        }
                    }
                    */
                    char transa = 'T';
                    char transb = 'T';
                    int m = num_m;
                    int n = num_n;
                    int k = block_npoints;
                    double* Abuff = &(phi_values_[block_start * nbf + m_start]);
                    double* Bbuff = &(Gvg[n_start * npoints + block_start]);
                    double* Cbuff = &(Kp[m_start][n_start]);
                    int lda = nbf;
                    int ldb = npoints;
                    int ldc = nbf;

                    C_DGEMM(transa, transb, m, n, k, 1.0, Abuff, lda, Bbuff, ldb, 1.0, Cbuff, ldc);
                }
            }
        }

        K[i]->hermitivitize();

    }

    timer_off("COSK::build_K()");
}

} // namespace Psi