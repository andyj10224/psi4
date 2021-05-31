#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_tree.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector3.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/multipoles.h"
#include "psi4/libmints/overlap.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libqt/qt.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <csignal>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

namespace psi {

CFMMBox::CFMMBox(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
        std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int lmax) {
    D_ = D;
    J_ = J;

    double min_dim = molecule->x(0);
    double max_dim = molecule->x(0);

    for (int atom = 0; atom < molecule->natom(); atom++) {
        double x = molecule->x(atom);
        double y = molecule->y(atom);
        double z = molecule->z(atom);
        min_dim = std::min(x, min_dim);
        min_dim = std::min(y, min_dim);
        min_dim = std::min(z, min_dim);
        max_dim = std::max(x, max_dim);
        max_dim = std::max(y, max_dim);
        max_dim = std::max(z, max_dim);
    }

    max_dim += 0.1; // Add a small buffer to the box

    Vector3 origin = Vector3(min_dim, min_dim, min_dim);
    double length = (max_dim - min_dim);

    common_init(nullptr, molecule, basisset, origin, length, 0, lmax);
}

CFMMBox::CFMMBox(CFMMBox* parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                    std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, Vector3 origin, double length, int level, int lmax) {
    D_ = D;
    J_ = J;
    common_init(parent, molecule, basisset, origin, length, level, lmax);
}

void CFMMBox::common_init(CFMMBox* parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                        Vector3 origin, double length, int level, int lmax) {
    parent_ = parent;
    molecule_ = molecule;
    basisset_ = basisset;
    origin_ = origin;
    length_ = length;
    level_ = level;
    lmax_ = lmax;

    nthread_ = 1;

#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

    center_ = origin_ + Vector3(length_ / 2, length_ / 2, length_ / 2);
    children_.resize(8, nullptr);

    // Make the multipole coefficients
    if (!parent_) {
        mpole_coefs_ = std::make_shared<HarmonicCoefficients>(lmax_, Regular);
    } else {
        mpole_coefs_ = parent_->mpole_coefs_;
    }

    if (!parent_) {
        for (int atom = 0; atom < molecule_->natom(); atom++) {
            atoms_.push_back(atom);
        }
    } else {
        for (int ind = 0; ind < parent_->atoms_.size(); ind++) {
            int atom = parent_->atoms_[ind];
            double x = molecule_->x(atom);
            double y = molecule_->y(atom);
            double z = molecule_->z(atom);

            bool x_good = (x >= origin_[0] && x < origin_[0] + length_);
            bool y_good = (y >= origin_[1] && y < origin_[1] + length_);
            bool z_good = (z >= origin_[2] && z < origin_[2] + length_);

            if (x_good && y_good && z_good) {
                atoms_.push_back(atom);
            }
        }
    }

    // Calculate the well separated criterion for the box
    ws_ = 2;
    if (length_ > 0.0) {
        for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
            int Patom = atoms_[Ptask];
            int Pstart = basisset_->shell_on_center(Patom, 0);
            int nPshell = basisset_->nshell_on_center(Patom);

            for (int P = Pstart; P < Pstart + nPshell; P++) {
                const GaussianShell& Pshell = basisset_->shell(P);
                int nprim = Pshell.nprimitive();
                for (int prim = 0; prim < nprim; prim++) {
                    double exp = Pshell.exp(prim);
                    double rp = ERFCI10 / std::sqrt(exp);
                    int ext = 2 * std::ceil(rp / length_);
                    ws_ = std::max(ws_, ext);
                }
            }
        }
    }

    int nbf = basisset_->nbf();

    for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshells = basisset_->nshell_on_center(Patom);

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshells = basisset_->nshell_on_center(Qatom);

            for (int P = Pstart; P < Pstart + nPshells; P++) {
                const GaussianShell& Pshell = basisset_->shell(P);
                int p_start = Pshell.start();
                int num_p = Pshell.nfunction();

                for (int Q = Qstart; Q < Qstart + nQshells; Q++) {
                    const GaussianShell& Qshell = basisset_->shell(Q);
                    int q_start = Qshell.start();
                    int num_q = Qshell.nfunction();

                    for (int p = p_start; p < p_start + num_p; p++) {
                        for (int q = q_start; q < q_start + num_q; q++) {
                            mpoles_[p * nbf + q] = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
                        } // q
                    } // p
                } // Q
            } // P
        } // Qtask
    } // Ptask

    Vff_ = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);

}

void CFMMBox::set_nf_lff() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::set_nf_lff()");

    // Parent is not a nullpointer
    if (parent_) {
        // Siblings of this box
        for (CFMMBox* sibling : parent_->children_) {
            if (sibling == this) continue;
            if (sibling->natom() == 0) continue;

            Vector3 Rab = center_ - sibling->center_;
            double dist = Rab.norm();

            if (dist <= ws_ * length_ * std::sqrt(3.0)) {
                near_field_.push_back(sibling);
            } else {
                local_far_field_.push_back(sibling);
            }
        }

        // Parent's near field (Cousins)
        for (CFMMBox* uncle : parent_->near_field_) {
            if (uncle->natom() == 0) continue;

            for (CFMMBox* cousin : uncle->children_) {
                if (cousin->natom() == 0) continue;

                Vector3 Rab = center_ - cousin->center_;
                double dist = Rab.norm();

                if (dist <= ws_ * length_ * std::sqrt(3.0)) {
                    near_field_.push_back(cousin);
                } else {
                    local_far_field_.push_back(cousin);
                }
            }
        }
    }

    timer_off("CFMMBox::set_nf_lff()");
}

void CFMMBox::make_children() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::make_children()");

    for (int c = 0; c < 8; c++) {
        double half_length = length_ / 2.0;
        int dx = (c & 4) >> 2; // 0 or 1
        int dy = (c & 2) >> 1; // 0 or 1
        int dz = (c & 1) >> 0; // 0 or 1

        Vector3 child_origin = origin_ + Vector3(half_length * dx, half_length * dy, half_length * dz);

        CFMMBox* child = new CFMMBox(this, molecule_, basisset_, D_, J_, child_origin, half_length, level_+1, lmax_);
        children_[c] = child;
    }

    timer_off("CFMMBox::make_children()");

}

void CFMMBox::compute_mpoles() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_mpoles()");

    const auto &mpole_terms = mpole_coefs_->get_terms();

    std::shared_ptr<IntegralFactory> int_factory = std::make_shared<IntegralFactory>(basisset_);

    std::vector<std::shared_ptr<OneBodyAOInt>> mpints(nthread_);
    std::vector<std::shared_ptr<OneBodyAOInt>> oints(nthread_);

    for (int thread = 0; thread < nthread_; thread++) {
        mpints[thread] = std::shared_ptr<OneBodyAOInt>(int_factory->ao_multipoles(lmax_));
        oints[thread] = std::shared_ptr<OneBodyAOInt>(int_factory->ao_overlap());
    }

    int n_mult = (lmax_ + 1) * (lmax_ + 2) * (lmax_ + 3) / 6 - 1;
    int nbf = basisset_->nbf();

    int max_am = basisset_->max_am();
    int max_nao = (max_am + 1) * (max_am + 2) / 2;


    // Compute multipole integrals for all atoms in the shell pair
#pragma omp parallel for
    for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshell = basisset_->nshell_on_center(Qatom);

            for (int P = Pstart; P < Pstart + nPshell; P++) {
                const GaussianShell& Pshell = basisset_->shell(P);
                int p_start = Pshell.start();
                int num_p = Pshell.nfunction();

                for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                    const GaussianShell& Qshell = basisset_->shell(Q);
                    int q_start = Qshell.start();
                    int num_q = Qshell.nfunction();

                    mpints[thread]->compute_shell(P, Q);
                    const double *mpole_buffer = mpints[thread]->buffer();

                    oints[thread]->compute_shell(P, Q);
                    const double *overlap_buffer = oints[thread]->buffer();

                    // Compute multipoles
                    for (int p = p_start; p < p_start + num_p; p++) {
                        int dp = p - p_start;

                        for (int q = q_start; q < q_start + num_q; q++) {
                            int dq = q - q_start;
                            std::shared_ptr<RealSolidHarmonics> pq_mpoles = mpoles_[p * nbf + q];
                            pq_mpoles->add(0, 0, overlap_buffer[dp * num_q + dq]);

                            int running_index = 0;
                            for (int l = 1; l <= lmax_; l++) {
                                for (int m = -l; m <= l; m++) {
                                    int mu = m_addr(m);

                                    // std::raise(SIGINT);
                                    for (int ind = 0; ind < mpole_terms[l][mu].size(); ind++) {
                                        const std::tuple<double, int, int, int>& term_tuple = mpole_terms[l][mu][ind];
                                        double coef = std::get<0>(term_tuple);
                                        int a = std::get<1>(term_tuple);
                                        int b = std::get<2>(term_tuple);
                                        int c = std::get<3>(term_tuple);

                                        int abcindex = running_index + icart(a, b, c);
                                        pq_mpoles->add(l, mu, coef * mpole_buffer[abcindex * num_p * num_q + dp * num_q + dq]);
                                    }

                                } // end m loop

                                running_index += (l+1)*(l+2)/2;
                            } // end l loop
                        } // end q loop
                    } // end p loop
                } // end Q
            } // end P
        } // end Qtask
    } // end Ptask

    timer_off("CFMMBox::compute_mpoles()");

}

void CFMMBox::compute_mpoles_from_children() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_mpoles_from_children()");

    int nbf = basisset_->nbf();

    for (CFMMBox* child : children_) {
        if (child->atoms_.size() == 0) continue;

#pragma omp parallel for
        for (int Ptask = 0; Ptask < child->atoms_.size(); Ptask++) {
            int Patom = child->atoms_[Ptask];
            int Pstart = basisset_->shell_on_center(Patom, 0);
            int nPshell = basisset_->nshell_on_center(Patom);

            for (int Qtask = 0; Qtask < child->atoms_.size(); Qtask++) {
                int Qatom = child->atoms_[Qtask];
                int Qstart = basisset_->shell_on_center(Qatom, 0);
                int nQshell = basisset_->nshell_on_center(Qatom);

                for (int P = Pstart; P < Pstart + nPshell; P++) {
                    const GaussianShell& p_shell = basisset_->shell(P);
                    int p_start = p_shell.start();
                    int num_p = p_shell.nfunction();

                    for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                        const GaussianShell& q_shell = basisset_->shell(Q);
                        int q_start = q_shell.start();
                        int num_q = q_shell.nfunction();

                        for (int p = p_start; p < p_start + num_p; p++) {
                            for (int q = q_start; q < q_start + num_q; q++) {
                                std::shared_ptr<RealSolidHarmonics> child_mpoles = child->mpoles_[p * nbf + q]->translate(center_);
                                mpoles_[p * nbf + q]->add(child_mpoles);
                            } // End q
                        } // End p
                    } // End Q
                } // End P
            } // End Qtask
        } // End Ptask
    } // End children

    timer_off("CFMMBox::compute_mpoles_from_children()");

}

void CFMMBox::compute_far_field_vector() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_far_field_vector()");

    int nbf = basisset_->nbf();

    for (CFMMBox* box : local_far_field_) {
#pragma omp parallel for
        for (int Ptask = 0; Ptask < box->atoms_.size(); Ptask++) {
            int Patom = box->atoms_[Ptask];
            int Pstart = basisset_->shell_on_center(Patom, 0);
            int nPshell = basisset_->nshell_on_center(Patom);

            for (int Qtask = 0; Qtask < box->atoms_.size(); Qtask++) {
                int Qatom = box->atoms_[Qtask];
                int Qstart = basisset_->shell_on_center(Qatom, 0);
                int nQshell = basisset_->nshell_on_center(Qatom);

                for (int P = Pstart; P < Pstart + nPshell; P++) {
                    const GaussianShell& p_shell = basisset_->shell(P);
                    int p_start = p_shell.start();
                    int num_p = p_shell.nfunction();

                    for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                        const GaussianShell& q_shell = basisset_->shell(Q);
                        int q_start = q_shell.start();
                        int num_q = q_shell.nfunction();

                        for (int ind = 0; ind < D_.size(); ind++) {
                            for (int p = p_start; p < p_start + num_p; p++) {
                                for (int q = q_start; q < q_start + num_q; q++) {
                                    std::shared_ptr<RealSolidHarmonics> box_mpoles = box->mpoles_[p * nbf + q];
                                    // The far field effect the boxes have on this particular box
                                    std::shared_ptr<RealSolidHarmonics> far_field = box_mpoles->far_field_vector(center_);
                                    far_field->scale(D_[ind]->get(p, q));

                                    Vff_->add(far_field);
                                }
                            } // q
                        } // p
                    } // Q
                } // P
            } // Qtask
        } // Ptask
    } // box

    // Parent is not null
    if (parent_) {
        Vff_->add(parent_->Vff_->translate(center_));
    }

    timer_off("CFMMBox::compute_far_field_vector()");

}


void CFMMBox::compute_self_J() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_self_J()");

    std::vector<std::shared_ptr<TwoBodyAOInt>> ints;

    std::shared_ptr<IntegralFactory> factory = std::make_shared<IntegralFactory>(basisset_);
    std::shared_ptr<TwoBodyAOInt> eri = std::shared_ptr<TwoBodyAOInt>(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    ints.push_back(eri);

    for (int thread = 1; thread < nthread_; thread++) {
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(eri->clone()));
    }

    int nbf = basisset_->nbf();

    // outfile->Printf("   ATOMS SIZE: %d\n", atoms_.size());

    // Self-interactions
#pragma omp parallel for
    for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshell = basisset_->nshell_on_center(Qatom);

            for (int Rtask = 0; Rtask < atoms_.size(); Rtask++) {
                int Ratom = atoms_[Rtask];
                int Rstart = basisset_->shell_on_center(Ratom, 0);
                int nRshell = basisset_->nshell_on_center(Ratom);

                for (int Stask = 0; Stask < atoms_.size(); Stask++) {
                    int Satom = atoms_[Stask];
                    int Sstart = basisset_->shell_on_center(Satom, 0);
                    int nSshell = basisset_->nshell_on_center(Satom);

                    for (int P = Pstart; P < Pstart + nPshell; P++) {
                        int p_start = basisset_->shell(P).start();
                        int num_p = basisset_->shell(P).nfunction();

                        for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                            int q_start = basisset_->shell(Q).start();
                            int num_q = basisset_->shell(Q).nfunction();

                            for (int R = Rstart; R < Rstart + nRshell; R++) {
                                int r_start = basisset_->shell(R).start();
                                int num_r = basisset_->shell(R).nfunction();

                                for (int S = Sstart; S < Sstart + nSshell; S++) {
                                    int s_start = basisset_->shell(S).start();
                                    int num_s = basisset_->shell(S).nfunction();

                                    ints[thread]->compute_shell(P, Q, R, S);
                                    const double *buffer = ints[thread]->buffer();

                                    for (int ind = 0; ind < D_.size(); ind++) {
                                        double** Jp = J_[ind]->pointer();
                                        double** Dp = D_[ind]->pointer();

                                        for (int p = p_start; p < p_start + num_p; p++) {
                                            int dp = p - p_start;
                                            for (int q = q_start; q < q_start + num_q; q++) {
                                                int dq = q - q_start;
                                                for (int r = r_start; r < r_start + num_r; r++) {
                                                    int dr = r - r_start;
                                                    for (int s = s_start; s < s_start + num_s; s++) {
                                                        int ds = s - s_start;

                                                        double int_val = buffer[dp * num_q * num_r * num_s + dq * num_r * num_s + dr * num_s + ds];
                                                        Jp[p][q] += int_val * Dp[r][s];

                                                    } // s
                                                } // r
                                            } // q
                                        } // p
                                    } // ind
                                } // S
                            } // R
                        } // Q
                    } // P
                } // Stask
            } // Rtask
        } // Qtask
    } // Ptask

    timer_off("CFMMBox::compute_self_J()");
}

void CFMMBox::compute_nf_J() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_nf_J()");

    std::vector<std::shared_ptr<TwoBodyAOInt>> ints;

    std::shared_ptr<IntegralFactory> factory = std::make_shared<IntegralFactory>(basisset_);
    std::shared_ptr<TwoBodyAOInt> eri = std::shared_ptr<TwoBodyAOInt>(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    ints.push_back(eri);

    for (int thread = 1; thread < nthread_; thread++) {
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(eri->clone()));
    }

    int nbf = basisset_->nbf();

    // Near field interactions

    // outfile->Printf("Near field Num Boxes: %d\n", near_field_.size());

#pragma omp parallel for
    for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshell = basisset_->nshell_on_center(Qatom);

            for (int P = Pstart; P < Pstart + nPshell; P++) {
                int p_start = basisset_->shell(P).start();
                int num_p = basisset_->shell(P).nfunction();

                for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                    int q_start = basisset_->shell(Q).start();
                    int num_q = basisset_->shell(Q).nfunction();

                    for (int p = p_start; p < p_start + num_p; p++) {
                        int dp = p - p_start;

                        for (int q = q_start; q < q_start + num_q; q++) {
                            int dq = q - q_start;

                            for (int b = 0; b < near_field_.size(); b++) {
                                CFMMBox* box = near_field_[b];

                                for (int Rtask = 0; Rtask < box->atoms_.size(); Rtask++) {
                                    int Ratom = box->atoms_[Rtask];
                                    int Rstart = basisset_->shell_on_center(Ratom, 0);
                                    int nRshell = basisset_->nshell_on_center(Ratom);

                                    for (int Stask = 0; Stask < box->atoms_.size(); Stask++) {
                                        int Satom = box->atoms_[Stask];
                                        int Sstart = basisset_->shell_on_center(Satom, 0);
                                        int nSshell = basisset_->nshell_on_center(Satom);
                        

                                        for (int R = Rstart; R < Rstart + nRshell; R++) {
                                            int r_start = basisset_->shell(R).start();
                                            int num_r = basisset_->shell(R).nfunction();

                                            for (int S = Sstart; S < Sstart + nSshell; S++) {
                                                int s_start = basisset_->shell(S).start();
                                                int num_s = basisset_->shell(S).nfunction();

                                                ints[thread]->compute_shell(P, Q, R, S);
                                                const double *buffer = ints[thread]->buffer();

                                                for (int ind = 0; ind < D_.size(); ind++) {
                                                    double **Jp = J_[ind]->pointer();
                                                    double **Dp = D_[ind]->pointer();

                                                    for (int r = r_start; r < r_start + num_r; r++) {
                                                        int dr = r - r_start;

                                                        for (int s = s_start; s < s_start + num_s; s++) {
                                                            int ds = s - s_start;

                                                            double int_val = buffer[dp * num_q * num_r * num_s + dq * num_r * num_s + dr * num_s + ds];
                                                            Jp[p][q] += int_val * Dp[r][s];

                                                        } // s
                                                    } // r
                                                } // ind
                                            } // S
                                        } // R
                                    } // Stask
                                } // Rtask
                            } // box
                        } // q
                    } // p
                } // Q
            } // P
        } // Qtask
    } // Ptask

    timer_off("CFMMBox::compute_nf_J()");
}

void CFMMBox::compute_ff_J() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_ff_J()");

    int nbf = basisset_->nbf();

    // Far field interactions
    for (int ind = 0; ind < D_.size(); ind++) {
        double **Jp = J_[ind]->pointer();
        double **Dp = D_[ind]->pointer();
        for (auto &self_pair : mpoles_) {
            int pq_index = self_pair.first;
            std::vector<std::vector<double>>& pq_mpole = self_pair.second->get_multipoles();
            int p = pq_index / nbf;
            int q = pq_index % nbf;

            for (int l = 0; l <= lmax_; l++) {
                for (int m = -l; m <= l; m++) {
                    int mu = m_addr(m);
                    Jp[p][q] += pq_mpole[l][mu] * Vff_->get_multipoles()[l][mu];
                }
            }
            
        }
    }

    timer_off("CFMMBox::compute_ff_J()");
}

void CFMMBox::compute_J() {

    compute_self_J();
    compute_nf_J();
    compute_ff_J();

}

CFMMBox::~CFMMBox() {

    for (int c = 0; c < 8; c++) {
        if (children_[c]) delete children_[c];
    }

}

CFMMTree::CFMMTree(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                    std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int nlevels, int lmax) {
    molecule_ = molecule;
    basisset_ = basisset;
    nlevels_ = nlevels;
    lmax_ = lmax;
    D_ = D;
    J_ = J;
    root_ = new CFMMBox(molecule_, basisset_, D_, J_, lmax_);
}

void CFMMTree::make_children(CFMMBox* box) {
    if (box->get_level() == nlevels_ - 1) return;

    box->make_children();

    std::vector<CFMMBox*> children = box->get_children();
    
    for (CFMMBox* child : children) {
        make_children(child);
    }
}

void CFMMTree::calculate_multipoles(CFMMBox* box) {
    if (!box) return;
    if (box->natom() == 0) return;

    std::vector<CFMMBox*> children = box->get_children();

    for (CFMMBox* child : children) {
        calculate_multipoles(child);
    }

    if (box->get_level() == nlevels_ - 1) {
        box->compute_mpoles();
    } else {
        box->compute_mpoles_from_children();
    }

}

void CFMMTree::calculate_J(CFMMBox* box) {
    if (!box) return;
    if (box->natom() == 0) return;

    box->set_nf_lff();
    box->compute_far_field_vector();
    if (box->get_level() == nlevels_ - 1) box->compute_J();

    std::vector<CFMMBox*> children = box->get_children();

    for (CFMMBox* child : children) {
        calculate_J(child);
    }

}

void CFMMTree::build_J() {

    // Zero the J matrix
    for (int ind = 0; ind < D_.size(); ind++) {
        J_[ind]->zero();
    }

    make_children(root_);
    calculate_multipoles(root_);
    calculate_J(root_);

    // Hermitivitize J matrix afterwards
    for (int ind = 0; ind < D_.size(); ind++) {
        // J_[ind]->scale(2.0);
        J_[ind]->hermitivitize();
    }

}

CFMMTree::~CFMMTree() {
    delete root_;
}

} // end namespace psi