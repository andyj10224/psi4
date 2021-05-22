#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_tree.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>

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

CFMMBox::CFMMBox(std::shared_ptr<CFMMBox> parent, std::shared_ptr<Molecule> molecule, 
                std::shared_ptr<BasisSet> basisset, Vector3 origin, double length, int level, int lmax) {
    parent_ = parent;
    molecule_ = molecule;
    basisset_ = basisset;
    origin_ = origin;
    length_ = length;
    level_ = level;
    lmax_ = lmax;

    center_ = origin_ + Vector3(length_, length_, length_);
    children_.resize(8, nullptr);

    ws_ = 2;

    mpoles_ = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
    Vff_ = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);

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

        auto child = std::make_shared<CFMMBox>(std::shared_ptr<CFMMBox>(this), molecule_, basisset_, child_origin, half_length, level_+1, lmax_);
        children_.add(child);
    }
}

void CFMMBox::compute_mpoles() {
    mpoles_->compute_terms();
    auto mpole_terms = mpoles_->get_terms();
    auto raw_mpoles = mpoles_->get_multipoles();

    std::shared_ptr<IntegralFactory> int_factory = std::make_shared<IntegralFactory>(basisset_);
    std::shared_ptr<OneBodyAOInt> multipole_int = int_factory->ao_multipoles(lmax_);

    int n_multipoles = (lmax_ + 1) * (lmax_ + 2) * (lmax_ + 3) / 6 - 1;

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
                    
                    int running_index = 0;
                    for (int l = 0; l <= lmax_; l++) {
                        if (l == 0) continue;
                        
                        for (int m = -l; m <= l; l++) {
                            int mu = m_addr(m);
                            for (int ind = 0; ind < mpole_terms[l][mu].size(); ind++) {
                                auto term_tuple = mpole_terms[l][mu][ind];
                                double coef = std::get<0>(term_tuple);
                                int a = std::get<1>(term_tuple);
                                int b = std::get<2>(term_tuple);
                                int c = std::get<3>(term_tuple);

                                int abcindex = running_index + icart(a, b, c);

                                for (int p = m_start; p < m_start + num_m; m++) {
                                    int dp = p - m_start;
                                    for (int q = n_start; q < n_start + num_n; n++) {
                                        int dq = q - n_start;
                                        // put Dpq
                                        for (size_t a = 0; a < D_.size(); a++) {
                                            raw_mpoles[l][mu] += coef * D_[a]->get(p, q) * buffer[abc_index * num_m * num_n + dp * num_n + dq];
                                        }
                                    }
                                }
                            }
                        }

                        running_index += (l+1)*(l+1)/2;
                    }

                }
                
            }

        }
    }

}

void CFMMBox::compute_mpoles_from_children() {
    for (auto child : children_) {
        if (child.atoms_.size() == 0) continue;

        std::shared_ptr<RealSolidHarmonics> tmpoles = child.mpoles_->translate(center_);
        mpoles_->add(tmpoles);
    }
}

} // end namespace psi