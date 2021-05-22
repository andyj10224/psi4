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

    for (int l = 0; l <= lmax_; l++) {
        for (int m = -l; m <= l; l++) {
            int mu = m_addr(m);

            for (auto term : mpole_terms[l][mu]) {
                
            }

        }
    }
}

} // end namespace psi