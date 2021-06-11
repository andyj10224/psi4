
#include "psi4/libfock/linearhelper.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libqt/qt.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <set>
#include <map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace psi;

namespace psi {

LinearHelper::LinearHelper(std::shared_ptr<BasisSet> primary, bool lr_symmetric) {
    // Set Primary Basis Set
    primary_ = primary;
    // Set number of threads
    nthread_ = 1;
    // Set lr_symmetric_ (from DirectJK)
    lr_symmetric_ = lr_symmetric;

#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif
}

// CFMM J build
void LinearHelper::build_cfmmJ(std::vector<std::shared_ptr<TwoBodyAOInt> >& ints, std::vector<std::shared_ptr<Matrix> >& D,
                  std::vector<std::shared_ptr<Matrix> >& J) {
    throw FeatureNotImplemented("libmints", "build_cfmmJ", __FILE__, __LINE__);
}

bool linK_sort_helper(const std::tuple<int, double>& t1, const std::tuple<int, double>& t2) {
    return std::get<1>(t1) > std::get<1>(t2);
}

void LinearHelper::build_linK(std::vector<std::shared_ptr<TwoBodyAOInt> >& ints, std::vector<std::shared_ptr<Matrix> >& D,
                  std::vector<std::shared_ptr<Matrix> >& K) {
    
    timer_on("build_linK()");
    Options& options = Process::environment.options;
    double linK_thresh = options.get_double("LINK_THRESHOLD");

    // => Zeroing <= //
    for (size_t ind = 0; ind < K.size(); ind++) {
        K[ind]->zero();
    }
    // => Sizing <= //

    int nshell = primary_->nshell();
    int nthread = nthread_;

    // => Task Blocking <= //

    std::vector<int> task_shells;
    std::vector<int> task_starts;

    // => Atomic Blocking <= //

    int atomic_ind = -1;
    for (int P = 0; P < nshell; P++) {
        if (primary_->shell(P).ncenter() > atomic_ind) {
            task_starts.push_back(P);
            atomic_ind++;
        }
        task_shells.push_back(P);
    }
    task_starts.push_back(nshell);

    // => End Atomic Blocking <= //

    size_t ntask = task_starts.size() - 1;

    std::vector<int> task_offsets;
    task_offsets.push_back(0);
    for (int P2 = 0; P2 < primary_->nshell(); P2++) {
        task_offsets.push_back(task_offsets[P2] + primary_->shell(task_shells[P2]).nfunction());
    }

    size_t max_task = 0L;
    for (size_t task = 0; task < ntask; task++) {
        size_t size = 0L;
        for (int P2 = 0; P2 < task_starts[task + 1]; P2++) {
            size += primary_->shell(task_shells[P2]).nfunction();
        }
        max_task = (max_task >= size ? max_task : size);
    }

    /*
    if (debug_) {
        outfile->Printf("  ==> LinK: Task Blocking <==\n\n");
        for (size_t task = 0; task < ntask; task++) {
            outfile->Printf("  Task: %3d, Task Start: %4d, Task End: %4d\n", task, task_starts[task],
                            task_starts[task + 1]);
            for (int P2 = task_starts[task]; P2 < task_starts[task + 1]; P2++) {
                int P = task_shells[P2];
                int size = primary_->shell(P).nfunction();
                int off = primary_->shell(P).function_index();
                int off2 = task_offsets[P2];
                outfile->Printf("    Index %4d, Shell: %4d, Size: %4d, Offset: %4d, Offset2: %4d\n", P2, P, size, off,
                                off2);
            }
        }
        outfile->Printf("\n");
    }
    */

    // => Significant Task Pairs (PQ|-style <= //

    std::vector<std::pair<int, int>> task_pairs;
    for (size_t Ptask = 0; Ptask < ntask; Ptask++) {
        for (size_t Qtask = 0; Qtask < ntask; Qtask++) {
            if (Qtask > Ptask) continue;
            bool found = false;
            for (int P2 = task_starts[Ptask]; P2 < task_starts[Ptask + 1]; P2++) {
                for (int Q2 = task_starts[Qtask]; Q2 < task_starts[Qtask + 1]; Q2++) {
                    int P = task_shells[P2];
                    int Q = task_shells[Q2];
                    if (ints[0]->shell_pair_significant(P, Q)) {
                        found = true;
                        task_pairs.push_back(std::pair<int, int>(Ptask, Qtask));
                        break;
                    }
                }
                if (found) break;
            }
        }
    }

    size_t ntask_pair = task_pairs.size();
    size_t ntask_pair2 = ntask_pair * ntask_pair;

    /*
    // => Sorted Significant Shell Pairs (PQ|-style <= //
    std::vector<std::vector<std::pair<int, int>>> shell_pairs;

    for (int ind = 0; ind < ntask_pair; ind++) {
        int Ptask = task_pairs[ind].first;
        int Qtask = task_pairs[ind].second;
        std::vector<std::pair<int, int>> temp;

        for (int P2 = task_starts[Ptask]; P2 < task_starts[Ptask + 1]; P2++) {
            for (int Q2 = task_starts[Qtask]; Q2 < task_starts[Qtask + 1]; Q2++) {
                int P = task_shells[P2];
                int Q = task_shells[Q2];
                if (ints[0]->shell_pair_significant(P, Q)) {
                    // double pair_val = ints[0]->shell_pair_max_value(P, Q);
                    temp.push_back(std::pair<int, int>(P, Q));
                }
            }
            shell_pairs.push_back(temp);
        }
        // std::sort(&(shell_pairs[ind][0]), &(shell_pairs[ind][0]) + shell_pairs[ind].size(), shell_pair_sort_helper);
    }
    */

    // => Intermediate Buffers <= //

    std::vector<std::vector<std::shared_ptr<Matrix> > > JKT;
    for (int thread = 0; thread < nthread; thread++) {
        std::vector<std::shared_ptr<Matrix> > JK2;
        for (size_t ind = 0; ind < D.size(); ind++) {
            JK2.push_back(std::make_shared<Matrix>("JKT (K only)", (lr_symmetric_ ? 4 : 8) * max_task, max_task));
        }
        JKT.push_back(JK2);
    }

    // => Benchmarks <= //

    size_t computed_shells = 0L;

// ==> Master Task Loop <== //

#pragma omp parallel for num_threads(nthread) schedule(dynamic) reduction(+ : computed_shells)
    for (size_t task = 0L; task < ntask_pair2; task++) {
        size_t task1 = task / ntask_pair;
        size_t task2 = task % ntask_pair;

        int Ptask = task_pairs[task1].first;
        int Qtask = task_pairs[task1].second;
        int Rtask = task_pairs[task2].first;
        int Stask = task_pairs[task2].second;

        // GOTCHA! Thought this should be RStask > PQtask, but
        // H2/3-21G: Task (10|11) gives valid quartets (30|22) and (31|22)
        // This is an artifact that multiple shells on each task allow
        // for for the Ptask's index to possibly trump any RStask pair,
        // regardless of Qtask's index
        if (Rtask > Ptask) continue;

        // printf("Task: %2d %2d %2d %2d\n", Ptask, Qtask, Rtask, Stask);

        int nPtask = task_starts[Ptask + 1] - task_starts[Ptask];
        int nQtask = task_starts[Qtask + 1] - task_starts[Qtask];
        int nRtask = task_starts[Rtask + 1] - task_starts[Rtask];
        int nStask = task_starts[Stask + 1] - task_starts[Stask];

        int P2start = task_starts[Ptask];
        int Q2start = task_starts[Qtask];
        int R2start = task_starts[Rtask];
        int S2start = task_starts[Stask];
        
        int Pstart = task_shells[P2start];
        int Qstart = task_shells[Q2start];
        int Rstart = task_shells[R2start];
        int Sstart = task_shells[S2start];

        int dPsize = task_offsets[P2start + nPtask] - task_offsets[P2start];
        int dQsize = task_offsets[Q2start + nQtask] - task_offsets[Q2start];
        int dRsize = task_offsets[R2start + nRtask] - task_offsets[R2start];
        int dSsize = task_offsets[S2start + nStask] - task_offsets[S2start];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        
        std::vector<std::vector<int>> PR_sig_shells(nPtask, std::vector<int>(0));
        std::vector<std::vector<int>> PS_sig_shells(nPtask, std::vector<int>(0));
        std::vector<std::vector<int>> QR_sig_shells(nQtask, std::vector<int>(0));
        std::vector<std::vector<int>> QS_sig_shells(nQtask, std::vector<int>(0));

        std::vector<std::unordered_set<int>> PR_sig_shells2(nPtask, std::unordered_set<int>());
        std::vector<std::unordered_set<int>> PS_sig_shells2(nPtask, std::unordered_set<int>());
        std::vector<std::unordered_set<int>> QR_sig_shells2(nQtask, std::unordered_set<int>());
        std::vector<std::unordered_set<int>> QS_sig_shells2(nQtask, std::unordered_set<int>());

        std::vector<double> shell_ceiling_P(nPtask, 0.0);
        std::vector<double> shell_ceiling_Q(nQtask, 0.0);
        std::vector<double> shell_ceiling_R(nRtask, 0.0);
        std::vector<double> shell_ceiling_S(nStask, 0.0);

        // P and Q shell ceilings
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            int P = task_shells[P2];
            int dP = P - Pstart;
            for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
                int Q = task_shells[Q2];
                int dQ = Q - Qstart;
                double val = ints[0]->shell_ceiling(P, Q);
                shell_ceiling_P[dP] = std::max(shell_ceiling_P[dP], val);
                shell_ceiling_Q[dQ] = std::max(shell_ceiling_Q[dQ], val);
            }
        }

        // R and S shell ceilings
        for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
            int R = task_shells[R2];
            int dR = R - Rstart;
            for (int S2 = S2start; S2 < S2start + nStask; S2++) {
                int S = task_shells[S2];
                int dS = S - Sstart;
                double val = ints[0]->shell_ceiling(R, S);
                shell_ceiling_R[dR] = std::max(shell_ceiling_R[dR], val);
                shell_ceiling_S[dS] = std::max(shell_ceiling_S[dS], val);
            }
        }
        
        // PR significant shells
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            int P = task_shells[P2];
            int dP = P - Pstart;
            std::vector<std::tuple<int, double>> temp;
            for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
                int R = task_shells[R2];
                int dR = R - Rstart;
                double val = ints[0]->shell_max_density(P, R) * shell_ceiling_P[dP] * shell_ceiling_R[dR];
                if (val >= linK_thresh) temp.push_back(std::tuple<int, double>(R, val));
            }

            if (temp.size() > 0) {
                std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);
            }

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &el = temp[ind];
                PR_sig_shells[dP].push_back(std::get<0>(el));
            }
        }

        // PS significant shells
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            int P = task_shells[P2];
            int dP = P - Pstart;
            std::vector<std::tuple<int, double>> temp;
            for (int S2 = S2start; S2 < S2start + nStask; S2++) {
                int S = task_shells[S2];
                int dS = S - Sstart;
                double val = ints[0]->shell_max_density(P, S) * shell_ceiling_P[dP] * shell_ceiling_S[dS];
                if (val > linK_thresh) temp.push_back(std::tuple<int, double>(S, val));
            }

            if (temp.size() > 0) {
                std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);
            }

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &el = temp[ind];
                PS_sig_shells[dP].push_back(std::get<0>(el));
            }
        }

        // QR significant shells
        for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
            int Q = task_shells[Q2];
            int dQ = Q - Qstart;
            std::vector<std::tuple<int, double>> temp;
            for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
                int R = task_shells[R2];
                int dR = R - Rstart;
                double val = ints[0]->shell_max_density(Q, R) * shell_ceiling_Q[dQ] * shell_ceiling_R[dR];
                if (val > linK_thresh) temp.push_back(std::tuple<int, double>(R, val));
            }

            if (temp.size() > 0) {
                std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);
            }

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &el = temp[ind];
                QR_sig_shells[dQ].push_back(std::get<0>(el));
            }
        }

        // QS significant shells
        for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
            int Q = task_shells[Q2];
            int dQ = Q - Qstart;
            std::vector<std::tuple<int, double>> temp;
            for (int S2 = S2start; S2 < S2start + nStask; S2++) {
                int S = task_shells[S2];
                int dS = S - Sstart;
                double val = ints[0]->shell_max_density(Q, S) * shell_ceiling_Q[dQ] * shell_ceiling_S[dS];
                if (val > linK_thresh) temp.push_back(std::tuple<int, double>(S, val));
            }

            if (temp.size() > 0) {
                std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);
            }

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &el = temp[ind];
                QS_sig_shells[dQ].push_back(std::get<0>(el));
            }
        }

        bool touched = false;
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {

                if (Q2 > P2) continue;

                int P = task_shells[P2];
                int Q = task_shells[Q2];

                int dP = P - Pstart;
                int dQ = Q - Qstart;

                std::unordered_set<int> PQ_sig_RS;

                // Form ML_P (Oschenfeld Fig. 1)
                for (int R2 = 0; R2 < PR_sig_shells[dP].size(); R2++) {
                    int count = 0;
                    for (int S2 = 0; S2 < PS_sig_shells[dP].size(); S2++) {
                        int R = PR_sig_shells[dP][R2];
                        int S = PS_sig_shells[dP][S2];
                        int dR = R - Rstart;
                        int dS = S - Sstart;

                        double val = ints[0]->quart_screen_linK(P, Q, R, S);
                        if (val >= linK_thresh) {
                            count += 1;
                            if (S > R) continue;
                            if (R * nshell + S > P * nshell + Q) continue;
                            if (!PR_sig_shells2[dP].count(R)) PR_sig_shells2[dP].emplace(R);
                            if (!PS_sig_shells2[dP].count(S)) PS_sig_shells2[dP].emplace(S);
                            if (PQ_sig_RS.count(R * nshell + S)) continue;
                            PQ_sig_RS.emplace(R * nshell + S);
                        }
                        else break;
                    }
                    if (count == 0) break;
                }

                // Form ML_Q
                for (int R2 = 0; R2 < QR_sig_shells[dQ].size(); R2++) {
                    int count = 0;
                    for (int S2 = 0; S2 < QS_sig_shells[dQ].size(); S2++) {
                        int R = QR_sig_shells[dQ][R2];
                        int S = QS_sig_shells[dQ][S2];
                        int dR = R - Rstart;
                        int dS = S - Sstart;

                        double val = ints[0]->quart_screen_linK(P, Q, R, S);
                        if (val >= linK_thresh) {
                            count += 1;
                            if (S > R) continue;
                            if (R * nshell + S > P * nshell + Q) continue;
                            if (!QR_sig_shells2[dQ].count(R)) QR_sig_shells2[dQ].emplace(R);
                            if (!QS_sig_shells2[dQ].count(S)) QS_sig_shells2[dQ].emplace(S);
                            if (PQ_sig_RS.count(R * nshell + S)) continue;
                            PQ_sig_RS.emplace(R * nshell + S);
                        }
                        else break;
                    }
                    if (count == 0) break;
                }

                // Loop over significant RS pairs
                for (const int RS2 : PQ_sig_RS) {

                    int R = RS2 / nshell;
                    int S = RS2 % nshell;
                    int R2 = R;
                    int S2 = S;
                        
                    if (S > R) continue;
                    if (R * nshell + S > P * nshell + Q) continue;

                    if (!ints[0]->shell_pair_significant(R, S)) continue;
                    if (!ints[0]->shell_significant_density_K(P, Q, R, S)) continue;
                    if (!ints[0]->shell_significant(P, Q, R, S)) continue;

                    // printf("Quartet: %2d %2d %2d %2d\n", P, Q, R, S);
                    // timer_on("compute_shell(P, Q, R, S)");
                    // if (thread == 0) timer_on("JK: Ints");
                    if (ints[thread]->compute_shell(P, Q, R, S) == 0)
                        continue;  // No integrals in this shell quartet
                    computed_shells++;
                    // if (thread == 0) timer_off("JK: Ints");
                    // timer_off("compute_shell(P, Q, R, S)");

                    const double* buffer = ints[thread]->buffer();

                    int Psize = primary_->shell(P).nfunction();
                    int Qsize = primary_->shell(Q).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Ssize = primary_->shell(S).nfunction();

                    int Poff = primary_->shell(P).function_index();
                    int Qoff = primary_->shell(Q).function_index();
                    int Roff = primary_->shell(R).function_index();
                    int Soff = primary_->shell(S).function_index();

                    int Poff2 = task_offsets[P2] - task_offsets[P2start];
                    int Qoff2 = task_offsets[Q2] - task_offsets[Q2start];
                    int Roff2 = task_offsets[R2] - task_offsets[R2start];
                    int Soff2 = task_offsets[S2] - task_offsets[S2start];

                    // if (thread == 0) timer_on("JK: GEMV");
                    for (size_t ind = 0; ind < D.size(); ind++) {
                        double** Dp = D[ind]->pointer();
                        double** JKTp = JKT[thread][ind]->pointer();
                        const double* buffer2 = buffer;

                        if (!touched) {
                            ::memset((void*)JKTp[0L * max_task], '\0', dPsize * dRsize * sizeof(double));
                            ::memset((void*)JKTp[1L * max_task], '\0', dPsize * dSsize * sizeof(double));
                            ::memset((void*)JKTp[2L * max_task], '\0', dQsize * dRsize * sizeof(double));
                            ::memset((void*)JKTp[3L * max_task], '\0', dQsize * dSsize * sizeof(double));
                            if (!lr_symmetric_) {
                                ::memset((void*)JKTp[4L * max_task], '\0', dRsize * dPsize * sizeof(double));
                                ::memset((void*)JKTp[5L * max_task], '\0', dSsize * dPsize * sizeof(double));
                                ::memset((void*)JKTp[6L * max_task], '\0', dRsize * dQsize * sizeof(double));
                                ::memset((void*)JKTp[7L * max_task], '\0', dSsize * dQsize * sizeof(double));
                            }
                        }

                        double* K1p = JKTp[0L * max_task];
                        double* K2p = JKTp[1L * max_task];
                        double* K3p = JKTp[2L * max_task];
                        double* K4p = JKTp[3L * max_task];
                        double* K5p;
                        double* K6p;
                        double* K7p;
                        double* K8p;
                        if (!lr_symmetric_) {
                            K5p = JKTp[4L * max_task];
                            K6p = JKTp[5L * max_task];
                            K7p = JKTp[6L * max_task];
                            K8p = JKTp[7L * max_task];
                        }

                        double prefactor = 1.0;
                        if (P == Q) prefactor *= 0.5;
                        if (R == S) prefactor *= 0.5;
                        if (P == R && Q == S) prefactor *= 0.5;

                        for (int p = 0; p < Psize; p++) {
                            for (int q = 0; q < Qsize; q++) {
                                for (int r = 0; r < Rsize; r++) {
                                    for (int s = 0; s < Ssize; s++) {
                                            
                                        K1p[(p + Poff2) * dRsize + r + Roff2] +=
                                            prefactor * (Dp[q + Qoff][s + Soff]) * (*buffer2);
                                        K2p[(p + Poff2) * dSsize + s + Soff2] +=
                                            prefactor * (Dp[q + Qoff][r + Roff]) * (*buffer2);
                                        K3p[(q + Qoff2) * dRsize + r + Roff2] +=
                                            prefactor * (Dp[p + Poff][s + Soff]) * (*buffer2);
                                        K4p[(q + Qoff2) * dSsize + s + Soff2] +=
                                            prefactor * (Dp[p + Poff][r + Roff]) * (*buffer2);
                                        if (!lr_symmetric_) {
                                            K5p[(r + Roff2) * dPsize + p + Poff2] +=
                                                prefactor * (Dp[s + Soff][q + Qoff]) * (*buffer2);
                                            K6p[(s + Soff2) * dPsize + p + Poff2] +=
                                                prefactor * (Dp[r + Roff][q + Qoff]) * (*buffer2);
                                            K7p[(r + Roff2) * dQsize + q + Qoff2] +=
                                                prefactor * (Dp[s + Soff][p + Poff]) * (*buffer2);
                                            K8p[(s + Soff2) * dQsize + q + Qoff2] +=
                                                prefactor * (Dp[r + Roff][p + Poff]) * (*buffer2);
                                        }
                                            
                                        buffer2++;
                                    }
                                }
                            }
                        }
                    }
                    touched = true;
                // if (thread == 0) timer_off("JK: GEMV");
                }
            }
        }
        
        // => Master shell quartet loops <= //

        if (!touched) continue;

        // => Stripe out <= //

        // if (thread == 0) timer_on("JK: Atomic");
        for (size_t ind = 0; ind < D.size(); ind++) {
            double** JKTp = JKT[thread][ind]->pointer();
            double** Kp = K[ind]->pointer();

            double* K1p = JKTp[0L * max_task];
            double* K2p = JKTp[1L * max_task];
            double* K3p = JKTp[2L * max_task];
            double* K4p = JKTp[3L * max_task];
            double* K5p;
            double* K6p;
            double* K7p;
            double* K8p;
            if (!lr_symmetric_) {
                K5p = JKTp[4L * max_task];
                K6p = JKTp[5L * max_task];
                K7p = JKTp[6L * max_task];
                K8p = JKTp[7L * max_task];
            }

            // > K_PR < //

            for (int P2 = 0; P2 < nPtask; P2++) {
                for (const int R : PR_sig_shells2[P2]) {
                    int P = task_shells[P2start + P2];
                    // int R = task_shells[R2start + R2];
                    int R2 = R - Rstart;
                
                    int Psize = primary_->shell(P).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Poff = primary_->shell(P).function_index();
                    int Roff = primary_->shell(R).function_index();
                
                    int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                    int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                    for (int p = 0; p < Psize; p++) {
                        for (int r = 0; r < Rsize; r++) {
#pragma omp atomic
                            Kp[p + Poff][r + Roff] += K1p[(p + Poff2) * dRsize + r + Roff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[r + Roff][p + Poff] += K5p[(r + Roff2) * dPsize + p + Poff2];
                            }
                        }
                    }
                }
            }

            // > K_PS < //

            for (int P2 = 0; P2 < nPtask; P2++) {
                for (const int S : PS_sig_shells2[P2]) {
                    int P = task_shells[P2start + P2];
                    // int S = task_shells[S2start + S2];
                    int S2 = S - Sstart;
                
                    int Psize = primary_->shell(P).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Poff = primary_->shell(P).function_index();
                    int Soff = primary_->shell(S).function_index();

                    int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                    int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                    for (int p = 0; p < Psize; p++) {
                        for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                            Kp[p + Poff][s + Soff] += K2p[(p + Poff2) * dSsize + s + Soff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[s + Soff][p + Poff] += K6p[(s + Soff2) * dPsize + p + Poff2];
                            }
                        }
                    }
                }
            }

            // > K_QR < //

            for (int Q2 = 0; Q2 < nQtask; Q2++) {
                for (const int R : QR_sig_shells2[Q2]) {
                    int Q = task_shells[Q2start + Q2];
                    // int R = task_shells[R2start + R2];
                    int R2 = R - Rstart;
                
                    int Qsize = primary_->shell(Q).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Qoff = primary_->shell(Q).function_index();
                    int Roff = primary_->shell(R).function_index();

                    int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                    int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                    for (int q = 0; q < Qsize; q++) {
                        for (int r = 0; r < Rsize; r++) {
#pragma omp atomic
                            Kp[q + Qoff][r + Roff] += K3p[(q + Qoff2) * dRsize + r + Roff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[r + Roff][q + Qoff] += K7p[(r + Roff2) * dQsize + q + Qoff2];
                            }
                        }
                    }
                }
            }

            // > K_QS < //

            for (int Q2 = 0; Q2 < nQtask; Q2++) {
                for (const int S : QS_sig_shells2[Q2]) {
                    int Q = task_shells[Q2start + Q2];
                    // int S = task_shells[S2start + S2];
                    int S2 = S - Sstart;
                
                    int Qsize = primary_->shell(Q).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Qoff = primary_->shell(Q).function_index();
                    int Soff = primary_->shell(S).function_index();

                    int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                    int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                    for (int q = 0; q < Qsize; q++) {
                        for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                            Kp[q + Qoff][s + Soff] += K4p[(q + Qoff2) * dSsize + s + Soff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[s + Soff][q + Qoff] += K8p[(s + Soff2) * dQsize + q + Qoff2];
                            }
                        }
                    }
                }
            }

        }  // End stripe out
        // if (thread == 0) timer_off("JK: Atomic");

    }  // End master task list

    if (lr_symmetric_) {
        for (size_t ind = 0; ind < D.size(); ind++) {
            K[ind]->scale(2.0);
            K[ind]->hermitivitize();
        }
    }

    /*
    if (bench_) {
        auto mode = std::ostream::app;
        auto printer = std::make_shared<PsiOutStream>("bench.dat", mode);
        size_t ntri = nshell * (nshell + 1L) / 2L;
        size_t possible_shells = ntri * (ntri + 1L) / 2L;
        printer->Printf("Computed %20zu Shell Quartets out of %20zu, (%11.3E ratio)\n", computed_shells,
                        possible_shells, computed_shells / (double)possible_shells);
    }
    */

    timer_off("build_linK()");
}

}