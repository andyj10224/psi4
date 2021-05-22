#ifndef libfmm_fmm_tree_H
#define libfmm_fmm_tree_H

#include "psi4/pragma.h"

#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libfmm/multipole_helper.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

namespace psi {

class CFMMBox {

    protected:
      // Parent of the CFMMBox
      std::shared_ptr<CFMMBox> parent_;
      // Children of the CFMMBox
      std::vector<std::shared_ptr<CFMMBox>> children_;
      // Level the box is at (0 = root)
      int level_;
      // Maximum Multipole Angular Momentum
      int lmax_;
      // Well-separatedness criteria for the particular Box
      int ws_;
      // The molecule that is referenced by this box
      std::shared_ptr<Molecule> molecule_;
      // The basis set that the molecule uses
      std::shared_ptr<BasisSet> basisset_;
      // The atoms in the molecule which are centered within the bounds of the box
      std::vector<int> atoms_;
      // Length of the box
      double length_;
      // The box's origin (lower-left-front corner)
      Vector3 origin_;
      // Center of the box
      Vector3 center_;

      // Multipoles of the box
      std::shared_ptr<RealSolidHarmonics> mpoles_;
      // Far field vector of the box
      std::shared_ptr<RealSolidHarmonics> Vff_;

      // A list of all the near-field boxes to this box
      std::vector<std::shared_ptr<CFMMBox>> near_field_;
      // A list of all of the local-far-field boxes to this box
      std::vector<std::shared_ptr<CFMMBox>> local_far_field_;

      // Sets the near field and local far field vectors
      void set_nf_lff();
      // Make children for this multipole box
      void make_children();
      
    public:
      // Constructor for a root box
      CFMMBox(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset);
      // Constructor for child boxes
      CFMMBox(std::shared_ptr<CFMMBox> parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, Vector3 origin, double length, int level, int lmax);
      // Compute multipoles directly
      void compute_mpoles();
      // Compute multipoles from children
      void compute_mpoles_from_children();

}; // End class CFMMBox

class CFMMTree {

    protected:
      // Number of Levels in the CFMM Tree
      int nlevels_;
      // The molecule that this tree structure references
      std::shared_ptr<Molecule> molecule_;
      // The basis set that the molecule uses
      std::shared_ptr<BasisSet> basisset_;
      // Maximum Multipole Angular Momentum
      int lmax_;
      // Length of root
      double length_;
      // Root of this tree structure
      std::shared_ptr<CFMMBox> root_;

      // Create children
      void make_children_helper(std::shared_ptr<CFMMBox>& box);
      // Calculate multipoles
      void calculate_multipoles_helper(std::shared_ptr<CFMMBox>& box);
    
    public:
      // Constructor
      CFMMTree(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, int nlevels);

}; // End class CFMMTree

} // namespace psi

#endif