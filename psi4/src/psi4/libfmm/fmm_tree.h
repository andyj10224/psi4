#ifndef libfmm_fmm_tree_H
#define libfmm_fmm_tree_H

#include "psi4/pragma.h"

#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libfmm/multipoles_helper.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace psi {

class CFMMBox : public std::enable_shared_from_this<CFMMBox> {

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
      // Density Matrix of Molecule
      std::vector<SharedMatrix>& D_;
      // A reference to the Coulomb Matrix of the molecule (every box can modify it)
      std::vector<SharedMatrix>& J_;
      // The atoms in the molecule which are centered within the bounds of the box
      std::vector<int> atoms_;
      // Length of the box
      double length_;
      // The box's origin (lower-left-front corner)
      Vector3 origin_;
      // Center of the box
      Vector3 center_;

      // Solid Harmonics Coefficients of Box
      std::shared_ptr<HarmonicCoefficients> mpole_coefs_;

      // Multipoles of the box, per basis pair
      std::unordered_map<int, std::shared_ptr<RealSolidHarmonics>> mpoles_;
      // Far field vector of the box, per basis pair
      std::unordered_map<int, std::shared_ptr<RealSolidHarmonics>> Vff_;

      // A list of all the near-field boxes to this box
      std::vector<std::shared_ptr<CFMMBox>> near_field_;
      // A list of all of the local-far-field boxes to this box
      std::vector<std::shared_ptr<CFMMBox>> local_far_field_;

      // Common function used by constructor
      void common_init(std::shared_ptr<CFMMBox> parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                        std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, Vector3 origin, double length, int level, int lmax);

      // Calculate far field vector from local and parent far fields
      void compute_far_field_vector();

      // Compute the J matrix contributions at each level
      void compute_self_J();
      void compute_nf_J();
      void compute_ff_J();
      
    public:
      // Constructor for a root box
      CFMMBox(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int lmax);
      // Constructor for child boxes
      CFMMBox(std::shared_ptr<CFMMBox> parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, Vector3 origin, double length, int level, int lmax);
      // Compute multipoles directly
      void compute_mpoles();
      // Compute multipoles from children
      void compute_mpoles_from_children();
      // Sets the near field and local far field vectors
      void set_nf_lff();
      // Make children for this multipole box
      void make_children();
      // Get the multipole level the box is on
      int get_level() { return level_; }
      // Get the children of the box
      std::vector<std::shared_ptr<CFMMBox>>& get_children() { return children_; }
      // Compute the box's contribution to the J matrix
      void compute_J();
      

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
      // Root of this tree structure
      std::shared_ptr<CFMMBox> root_;
      // Density Matrix of Molecule
      std::vector<SharedMatrix>& D_;
      // Coulomb Matrix of Molecule
      std::vector<SharedMatrix>& J_;

      // Create children
      void make_children_helper(std::shared_ptr<CFMMBox>& box);
      // Calculate multipoles
      void calculate_multipoles_helper(std::shared_ptr<CFMMBox>& box);
      // Helper method to build the J Matrix recursively
      void calculate_J(std::shared_ptr<CFMMBox>& box);
    
    public:
      // Constructor
      CFMMTree(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int nlevels, int lmax);

      // Build the J matrix of CFMMTree
      void build_J();

}; // End class CFMMTree

} // namespace psi

#endif