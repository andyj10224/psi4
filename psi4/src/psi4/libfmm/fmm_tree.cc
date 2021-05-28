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

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
int threads = Process::environment.get_n_threads();
#endif
int thread = 0;
#ifdef _OPENMP
thread = omp_get_thread_num();
#endif

namespace psi {

int choose(int n, int r) {
    if (r < 0 || r > n) {
        return 0;
    }
    int small = std::min(n, n-r);
    int nCr = 1;
    for (int t = 0; t < small; t++) {
        nCr *= n;
        nCr /= (t+1);
        n -= 1;
    }
    return nCr;
}

int m_addr(int m) {
    /*- Return the unsigned (array) address of m -*/
    if (m <= 0) {
        // 0, 1s, 2s, 3s, ...
        return 2*(-m);
    } else {
        // 1c, 2c, 3c, ...
        return 2*m-1;
    }
}

CFMMBox::CFMMBox(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
        std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int lmax) {

    double min_dim = mol->x(0);
    double max_dim = mol->x(0);

    for (int atom = 0; atom < mol->natom(); atom++) {
        double x = mol->x(atom);
        double y = mol->y(atom);
        double z = mol->z(atom);
        min_dim = std::min(x, min_dim);
        min_dim = std::min(y, min_dim);
        min_dim = std::min(z, min_dim);
        max_dim = std::max(x, max_dim);
        max_dim = std::max(y, max_dim);
        max_dim = std::max(z, max_dim);
    }

    Vector3 origin = Vector3(min_dim, min_dim, min_dim);
    double length = (max_dim - min_dim);

    common_init(nullptr, molecule, basisset, D, J, origin, length, 0, lmax);
}

CFMMBox::CFMMBox(std::shared_ptr<CFMMBox> parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                    std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, Vector3 origin, double length, int level, int lmax) {
    common_init(parent, molecule, basisset, D, origin, length, level, lmax);
}

CFMMBox::common_init(std::shared_ptr<CFMMBox> parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                        std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, Vector3 origin, double length, int level, int lmax) {
    parent_ = parent;
    molecule_ = molecule;
    basisset_ = basisset;
    D_ = D;
    J_ = J;
    origin_ = origin;
    length_ = length;
    level_ = level;
    lmax_ = lmax;

    center_ = origin_ + Vector3(length_, length_, length_);
    children_.resize(8, nullptr);

    ws_ = 2;

    if (!parent_) {
        mpole_coefs_ = std::make_shared<HarmonicCoefficients>(lmax_, Regular);
    } else {
        mpole_coefs_ = parent_->mpole_coefs_;
    }

    if (!parent_) {
        for (int atom = 0; atom < mol->natom(); atom++) {
            atoms_.append(atom);
        }
    } else {
        for (int ind = 0; ind < parent_.atoms_.size(); ind++) {
            int atom = parent_.atoms_[ind];
            double x = mol->x(atom);
            double y = mol->y(atom);
            double z = mol->z(atom);

            bool x_good = (x >= origin_[0] && x < origin_[0] + length_);
            bool y_good = (y >= origin_[1] && y < origin_[1] + length_);
            bool z_good = (z >= origin_[2] && z < origin_[2] + length_);

            if (x_good && y_good && z_good) {
                atoms_.append(atom);
            }
        }
    }

    int nbf = basisset_->nbf();

    for (int i1 = 0; i1 < atoms_.size(); i1++) {
        int atom1 = atoms_[i1];
        int atom1_shell_start = basisset_->shell_on_center(atom1, 0);
        int atom1_nshells = basisset_->nshell_on_center(atom1);
        for (int i2 = 0; i2 < atoms_.size(); i2++) {
            int atom2 = atoms_[i2];
            int atom2_shell_start = basisset_->shell_on_center(atom2, 0);
            int atom2_nshells = basisset_->nshell_on_center(atom2);

            for (int M = atom1_shell_start; M < atom1_shell_start + atom1_nshells; M++) {
                const GaussianShell& m_shell = basisset_->shell(M);
                int m_start = m_shell.start();
                int num_m = m_shell.nfunction();
                for (int N = atom2_shell_start; N < atom2_shell_start + atom2_shells; N++) {
                    const GaussianShell& n_shell = basisset_->shell(N);
                    int n_start = n_shell.start();
                    int num_n = n_shell.nfunction();

                    for (int m = m_start; m < m_start + num_m; m++) {
                        for (int n = n_start; n < n_start + num_n; n++) {
                            auto temp_mpoles = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
                            // auto temp_vff = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);
                            int basis_ind = m * nbf + n;
                            mpoles_.emplace(std::make_pair<int, std::shared_ptr<RealSolidHarmonics>(basis_ind, temp_mpoles));
                            // Vff_.emplace(std::make_pair<int, std::shared_ptr<RealSolidHarmonics>(basis_ind, temp_vff));
                        }
                    }
                }
            }
        }
    }
}

void CFMMBox::set_nf_lff() {

    if (!atoms_.size()) {
        return;
    }

    // Parent is not a nullpointer
    if (parent_) {
        // Siblings of this box
        for (auto sibling : parent_.children_) {
            if ((CFMMBox *) sibling == this) {
                continue;
            }
            Vector3 Rab = center_ - sibling->center_;
            double dist = Rab.norm();

            if (dist <= ws_ * length_ * std::sqrt(3.0)) {
                near_field_.push_back(sibling);
            } else {
                local_far_field_.push_back(sibling);
            }
        }

        // Parent's near field (Cousins)
        for (auto box : parent_.near_field_) {
            for (auto cousin : box->children_) {
                Vector3 Rab = center - cousin->center_;
                double dist = Rab.norm();

                if (dist <= ws_ * length_ * std::sqrt(3.0)) {
                    near_field_.push_back(cousin);
                } else {
                    local_far_field_.push_back(cousin);
                }
            }
        }

    }

}

void CFMMBox::make_children() {
    for (int c = 0; c < 8; c++) {
        double half_length = length_ / 2.0;
        int dx = (c & 4) >> 2; // 0 or 1
        int dy = (c & 2) >> 1; // 0 or 1
        int dz = (c & 1) >> 0; // 0 or 1

        Vector3 child_origin = origin_ + Vector3(half_length * dx, half_length * dy, half_length * dz);

        auto child = std::make_shared<CFMMBox>(std::shared_ptr<CFMMBox>(this), molecule_, basisset_, D_, J_, child_origin, half_length, level_+1, lmax_);
        children_.add(child);
    }
}

void CFMMBox::compute_mpoles() {
    auto mpole_terms = mpole_coefs_->get_terms();

    std::shared_ptr<IntegralFactory> int_factory = std::make_shared<IntegralFactory>(basisset_);
    std::shared_ptr<OneBodyAOInt> multipole_int = int_factory->ao_multipoles(lmax_);
    std::shared_ptr<OneBodyAOInt> overlap_int = int_factory->ao_overlap();

    int n_multipoles = (lmax_ + 1) * (lmax_ + 2) * (lmax_ + 3) / 6 - 1;
    int nbf = primary_->nbf();

    // Compute multipole integrals for all atoms in the shell pair
    for (int i1 = 0; i1 < atoms_.size(); i1++) {
        int atom1 = atoms_[i1];
        int atom1_shell_start = basisset_->shell_on_center(atom1, 0);
        int atom1_nshells = basisset_->nshell_on_center(atom1);
        for (int i2 = 0; i2 < atoms_.size(); i2++) {
            int atom2 = atoms_[i2];
            int atom2_shell_start = basisset_->shell_on_center(atom2, 0);
            int atom2_nshells = basisset_->nshell_on_center(atom2);

            for (int M = atom1_shell_start; M < atom1_shell_start + atom1_nshells; M++) {
                const GaussianShell& m_shell = basisset_->shell(M);
                int m_start = m_shell.start();
                int num_m = m_shell.nfunction();
                for (int N = atom2_shell_start; N < atom2_shell_start + atom2_shells; N++) {
                    const GaussianShell& n_shell = basisset_->shell(N);
                    int n_start = n_shell.start();
                    int num_n = n_shell.nfunction();

                    multipole_int->compute(M, N);
                    const double *buffer = multipole_int->buffer();

                    overlap_int->compute(M, N);
                    const double *overlap_buffer = overlap_int->buffer();

                    // Compute multipoles
                    for (int p = m_start; p < m_start + num_m; m++) {
                        int dp = p - m_start;
                        for (int q = n_start; q < n_start + num_n; n++) {
                            int dq = q - n_start;
                            auto raw_mpoles = mpoles_[p * nbf + q]->get_multipoles();
                            raw_mpoles[0][0] += overlap_buffer[dp * num_n + dq];

                            int running_index = 0;
                            for (int l = 1; l <= lmax_; l++) {
                                for (int m = -l; m <= l; l++) {
                                    int mu = m_addr(m);
                                    for (int ind = 0; ind < mpole_terms[l][mu].size(); ind++) {
                                        auto term_tuple = mpole_terms[l][mu][ind];
                                        double coef = std::get<0>(term_tuple);
                                        int a = std::get<1>(term_tuple);
                                        int b = std::get<2>(term_tuple);
                                        int c = std::get<3>(term_tuple);
                                        int abcindex = running_index + icart(a, b, c);
                                        raw_mpoles[l][mu] += coef * buffer[abc_index * num_m * num_n + dp * num_n + dq];
                                    }
                                }
                                running_index += (l+1)*(l+1)/2;
                            }
                        }
                    }
                }
            }
        }
    }
}

void CFMMBox::compute_mpoles_from_children() {

    int nbf = primary_->nbf();

    for (auto child : children_) {

        if (child->atoms_.size() == 0) continue;

        for (int i1 = 0; i1 < child->atoms_.size(); i1++) {
            int atom1 = atoms_[i1];
            int atom1_shell_start = basisset_->shell_on_center(atom1, 0);
            int atom1_nshells = basisset_->nshell_on_center(atom1);
            for (int i2 = 0; i2 < child->atoms_.size(); i2++) {
                int atom2 = atoms_[i2];
                int atom2_shell_start = basisset_->shell_on_center(atom2, 0);
                int atom2_nshells = basisset_->nshell_on_center(atom2);

                for (int M = atom1_shell_start; M < atom1_shell_start + atom1_nshells; M++) {
                    const GaussianShell& m_shell = basisset_->shell(M);
                    int m_start = m_shell.start();
                    int num_m = m_shell.nfunction();
                    for (int N = atom2_shell_start; N < atom2_shell_start + atom2_shells; N++) {
                        const GaussianShell& n_shell = basisset_->shell(N);
                        int n_start = n_shell.start();
                        int num_n = n_shell.nfunction();

                        for (int m = m_start; m < m_start + num_m; m++) {
                            for (int n = n_start; n < n_start + num_n; n++) {
                                std::shared_ptr<RealSolidHarmonics> tmpoles = child->mpoles_[m * nbf + n]->translate(center_);
                                mpoles_[m * nbf + n]->add(tmpoles);
                            }
                        }
                    }
                }
            }
        }
    }

}

void CFMMBox::compute_far_field_vector() {

    if (atoms_.size() == 0) continue;

    int nbf = primary_->nbf();

    for (auto box : local_far_field_) {

        for (int i1 = 0; i1 < box->atoms_.size(); i1++) {
            int atom1 = atoms_[i1];
            int atom1_shell_start = basisset_->shell_on_center(atom1, 0);
            int atom1_nshells = basisset_->nshell_on_center(atom1);
            for (int i2 = 0; i2 < box->atoms_.size(); i2++) {
                int atom2 = atoms_[i2];
                int atom2_shell_start = basisset_->shell_on_center(atom2, 0);
                int atom2_nshells = basisset_->nshell_on_center(atom2);

                for (int M = atom1_shell_start; M < atom1_shell_start + atom1_nshells; M++) {
                    const GaussianShell& m_shell = basisset_->shell(M);
                    int m_start = m_shell.start();
                    int num_m = m_shell.nfunction();
                    for (int N = atom2_shell_start; N < atom2_shell_start + atom2_shells; N++) {
                        const GaussianShell& n_shell = basisset_->shell(N);
                        int n_start = n_shell.start();
                        int num_n = n_shell.nfunction();

                        for (int m = m_start; m < m_start + num_m; m++) {
                            for (int n = n_start; n < n_start + num_n; n++) {
                                auto box_mpoles = box->mpoles_[m * nbf + n];
                                auto far_field = box_mpoles->far_field_vector(center_);
                                if (!(Vff_.count(m * nbf + n))) {
                                    Vff_.emplace(std::make_pair<int, std::shared_ptr<RealSolidHarmonics>(m * nbf + n, far_field));
                                } else {
                                    Vff_[m * nbf + n]->add(far_field);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (parent_) {
            for (std::pair<int, std::shared_ptr<RealSolidHarmonics> pair : parent_->Vff_) {
                int basis_ind = pair.first;
                auto vff = pair.second;
                int m = basis_ind / nbf;
                int n = basis_ind % nbf;
                auto trans_vff = vff->irregular_translate(center_);

                if (!(Vff_.count(m * nbf + n))) {
                    Vff_.emplace(std::make_pair<int, std::shared_ptr<RealSolidHarmonics>(m * nbf + n, trans_vff));
                } else {
                    Vff_[m * nbf + n]->add(trans_vff);
                }
            }
        }
    }
}

void CFMMBox::compute_self_J() {

    if (atoms_.size() == 0) return;

    auto factory = std::make_shared<IntegralFactory>(primary_);
    auto eri = std::shared_ptr<TwoBodyAOInt>(primary_->eri());

    // Self-interactions
    for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            for (int Rtask = 0; Rtask < atoms_.size(); Rtask++) {
                for (int Stask = 0; Stask < atoms_.size(); Stask++) {
                    int Patom = atoms_[Ptask];
                    int Qatom = atoms_[Qtask];
                    int Ratom = atoms_[Rtask];
                    int Satom = atoms_[Stask];

                    int Pstart = basisset_->shell_on_center(Patom, 0);
                    int Qstart = basisset_->shell_on_center(Qatom, 0);
                    int Rstart = basisset_->shell_on_center(Ratom, 0);
                    int Sstart = basisset_->shell_on_center(Satom, 0);

                    int nPshell = basisset_->shell_on_center(Patom, 0);
                    int nQshell = basisset_->shell_on_center(Qatom, 0);
                    int nRshell = basisset_->shell_on_center(Ratom, 0);
                    int nSshell = basisset_->shell_on_center(Satom, 0);

                    for (int P = Pstart; P < Pstart + nPshell; P++) {
                        for (int Q = Pstart; Q < Pstart + nQshell; Q++) {
                            for (int R = Rstart; R < Rstart + nRshell; R++) {
                                for (int S = Sstart; S < Sstart + nSshell; S++) {

                                    eri->compute_shell(P, Q, R, S);
                                    const double *buffer = eri->buffer();

                                    int p_start = basisset_->shell(P).start();
                                    int q_start = basisset_->shell(Q).start();
                                    int r_start = basisset_->shell(R).start();
                                    int s_start = basisset_->shell(S).start();

                                    int num_p = basisset_->shell(P).nfunction();
                                    int num_q = basisset_->shell(Q).nfunction();
                                    int num_r = basisset_->shell(R).nfunction();
                                    int num_s = basisset_->shell(S).nfunction();

                                    for (int ind = 0; ind < D_.size(); ind++) {
                                        double *Jp = J_[ind]->pointer()[0];
                                        double *Dp = D_[ind]->pointer()[0];
                                        for (int p = p_start; p < p_start + num_p; p++) {
                                            for (int q = q_start; q < q_start + num_q; q++) {
                                                for (int r = r_start; r < r_start + num_r; r++) {
                                                    for (int s = s_start; s < s_start + num_s; s++) {
                                                        int dp = p - p_start;
                                                        int dq = q - q_start;
                                                        int dr = r - r_start;
                                                        int ds = s - s_start;

                                                        double int_val = buffer[dp * num_q * num_r * num_s + dq * num_r * num_s + dr * num_s + ds];
                                                        Jp[p * nbf + q] = int_val * Dp[r * nbf + s];

                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void CFMMBox::compute_nf_J() {

    if (atoms_.size() == 0) return;

    auto factory = std::make_shared<IntegralFactory>(primary_);
    auto eri = std::shared_ptr<TwoBodyAOInt>(primary_->eri());

    // Near field interactions
    for (auto box : near_field_) {
        for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
            for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
                for (int Rtask = 0; Rtask < box->atoms_.size(); Rtask++) {
                    for (int Stask = 0; Stask < box->atoms_.size(); Stask++) {
                        int Patom = atoms_[Ptask];
                        int Qatom = atoms_[Qtask];
                        int Ratom = atoms_[Rtask];
                        int Satom = atoms_[Stask];

                        int Pstart = basisset_->shell_on_center(Patom, 0);
                        int Qstart = basisset_->shell_on_center(Qatom, 0);
                        int Rstart = basisset_->shell_on_center(Ratom, 0);
                        int Sstart = basisset_->shell_on_center(Satom, 0);

                        int nPshell = basisset_->shell_on_center(Patom, 0);
                        int nQshell = basisset_->shell_on_center(Qatom, 0);
                        int nRshell = basisset_->shell_on_center(Ratom, 0);
                        int nSshell = basisset_->shell_on_center(Satom, 0);

                        for (int P = Pstart; P < Pstart + nPshell; P++) {
                            for (int Q = Pstart; Q < Pstart + nQshell; Q++) {
                                for (int R = Rstart; R < Rstart + nRshell; R++) {
                                    for (int S = Sstart; S < Sstart + nSshell; S++) {

                                        eri->compute_shell(P, Q, R, S);
                                        const double *buffer = eri->buffer();

                                        int p_start = basisset_->shell(P).start();
                                        int q_start = basisset_->shell(Q).start();
                                        int r_start = basisset_->shell(R).start();
                                        int s_start = basisset_->shell(S).start();

                                        int num_p = basisset_->shell(P).nfunction();
                                        int num_q = basisset_->shell(Q).nfunction();
                                        int num_r = basisset_->shell(R).nfunction();
                                        int num_s = basisset_->shell(S).nfunction();

                                        for (int ind = 0; ind < D_.size(); ind++) {
                                            double *Jp = J_[ind]->pointer()[0];
                                            double *Dp = D_[ind]->pointer()[0];
                                            for (int p = p_start; p < p_start + num_p; p++) {
                                                for (int q = q_start; q < q_start + num_q; q++) {
                                                    for (int r = r_start; r < r_start + num_r; r++) {
                                                        for (int s = s_start; s < s_start + num_s; s++) {
                                                            int dp = p - p_start;
                                                            int dq = q - q_start;
                                                            int dr = r - r_start;
                                                            int ds = s - s_start;

                                                            double int_val = buffer[dp * num_q * num_r * num_s + dq * num_r * num_s + dr * num_s + ds];
                                                            Jp[p * nbf + q] = int_val * Dp[r * nbf + s];

                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void CFMMBox::compute_ff_J() {

    if (atoms_.size() == 0) return;

    int nbf = primary_->nbf();

    // Far field interactions
    for (int ind = 0; ind < D_.size(); ind++) {
        double *Jp = J_[ind]->pointer()[0];
        double *Dp = D_[ind]->pointer()[0];
        for (const auto self_pair& : mpoles_) {
            int pq_index = self_pair.first;
            int pq_mpole = self_pair.second;
            int p = pq_index / nbf;
            int q = pq_index % nbf;
            for (const auto vff_pair& : Vff_) {
                int rs_index = vff_pair.first;
                int rs_mpole = vff_pair.second;
                int r = rs_index / nbf;
                int s = rs_index % nbf;
                for (int l = 0; l <= lmax_; l++) {
                    for (int m = -l; m <= l; m++) {
                        int m_addr = m_addr(m);
                        Jp[p * nbf + q] += pq_mpole[l][m_addr] * rs_mole[l][m_addr] * Dp[r * nbf + s];
                    }
                }
            }
        }
    }
}

void CFMMBox::compute_J() {
    // Zero the J matrix
    for (int ind = 0; ind < D_.size(); ind++) {
        J[ind]->zero();
    }

    compute_self_J();
    compute_nf_J();
    compute_ff_J();

}


CFMMTree::CFMMTree(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                    std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int nlevels, int lmax) {
    molecule_ = molecule;
    basisset_ = basisset;
    nlevels_ = nlevels;
    lmax_ = lmax;
    D_ = D;
    J_ = J;
    root_ = std::make_shared<CFMMBox>(molecule_, basisset_, D_, J_, lmax_);
}

void CFMMTree::make_children_helper(std::shared_ptr<CFMMBox>& box) {
    if (box->get_level() == nlevels_ - 1) return;

    box->make_children();

    auto children = box->get_children();
    
    for (auto child : children) {
        make_children_helper(child);
    }
}

void CFMMTree::calculate_multipoles_helper(std::shared_ptr<CFMMBox>& box) {
    if (!box) return;

    for (auto child : children) {
        calculate_mulipoles_helper(child);
    }

    if (box->get_level() == nlevels_ - 1) {
        box->compute_mpoles();
        return;
    } else {
        box->compute_mpoles_from_children();
    }
}

void CFMMTree::calculate_J_helper(std::shared_ptr<CFMMBox>& box) {
    if (!box) return;

    box->set_nf_lff();

    if (box->get_level() == nlevels_ - 1) box->compute_J();

    for (auto child : children) {
        calculate_J_helper(child);
    }

}

void CFMMTree::build_J() {
    calculate_J_helper(root_);
}

} // end namespace psi