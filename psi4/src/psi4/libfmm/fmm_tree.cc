#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_tree.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>

namespace psi {

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

CFMMBox::make_children() {
    
}

} // end namespace psi