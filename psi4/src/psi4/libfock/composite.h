/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2022 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifndef COMPOSITE_H
#define COMPOSITE_H

#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/potential.h"
#include "psi4/libmints/twobody.h"
#include "psi4/lib3index/dfhelper.h"
#include "psi4/libfmm/fmm_tree.h"

#include <unordered_set>

namespace psi {

class DirectDFJ : public SplitJKBase {
  protected:
   /// The auxiliary basis set used in the DF algorithm
   std::shared_ptr<BasisSet> auxiliary_;
   /// The metrix used in the DF algorithm J_PQ = (P|Q)
   SharedMatrix Jmet_;
   // maximum values of Coulomb Metric for each auxuliary shell pair block PP
   std::vector<double> Jmet_max_;
   /// Numerical cutoff for ERI screening
   double cutoff_;
   // Perform Density matrix-based integral screening?
   bool density_screening_;

   /// Form J_PQ
   void build_metric();

   /// Builds the integrals for the DirectDFJ class
   void build_ints() override;

  public:
   /**
    * @brief Construct a new DirectDFJ object
    * 
    * @param primary The primary basis set used in DirectDFJ
    * @param auxiliary The auxiliary basis set used in DirectDFJ
    * @param options The options object
    */
   DirectDFJ(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

   /**
    * @author Andy Jiang and David Poole, Georgia Tech, April 2022
    *
    * @brief Builds the J matrix according to the DirectDFJ algorithm, described in [Weigand:2002:4285]_
    * doi: 10.1039/b204199p
    * 
    * @param D The list of AO density matrixes to contract to form the J matrix (1 for RHF, 2 for UHF/ROHF)
    * @param J The list of AO J matrices to build (Same size as D)
    */
   void build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) override;

   /**
    * @brief Prints information regarding Direct-DF-J run
    * 
    */
   void print_header() override;

};

class LinK : public SplitJKBase {
  protected:
   /// ERI Screening Cutoff
   double cutoff_;
   /// Density-based Sparsity Screening Cutoff for LinK
   double linK_ints_cutoff_;

   /// Builds the integrals for the LinK class
   void build_ints() override;

  public:
   /**
    * @brief Construct a new LinK object
    * 
    * @param primary The primary basis set used in LinK
    * @param options The options object
    */
   LinK(std::shared_ptr<BasisSet> primary, Options& options);

   /**
    * @author Andy Jiang, Georgia Tech, March 2022
    *
    * @brief Builds the K matrix according to the LinK algorithm, described in [Ochsenfeld:1998:1663]_
    * doi: 10.1063/1.476741
    * 
    * @param D The list of AO density matrixes to contract to form the K matrix (1 for RHF, 2 for UHF/ROHF)
    * @param K The list of AO K matrices to build (Same size as D)
    */
   void build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) override;

   /**
    * @brief Prints information regarding LinK run
    * 
    */
   void print_header() override;

};

class CFMM : public SplitJKBase {
  protected:
   /// The CFMMTree object used to compute the CFMM integrals
   std::shared_ptr<CFMMTree> cfmmtree_;
   /// Builds the integrals (CFMMTree) for the DirectDFJ class
   void build_ints() override;

  public:
   /**
    * @brief Construct a new CFMM object
    * 
    * @param primary The primary basis set used in DirectDFJ
    * @param options The options object
    */
   CFMM(std::shared_ptr<BasisSet> primary, Options& options);

   /**
    * @author Andy Jiang, Andy Simmonett, David Poole, Georgia Tech, April 2022
    *
    * @brief Builds the J matrix according to the CFMM Algorithm
    * 
    * @param D The list of AO density matrixes to contract to form the J matrix (1 for RHF, 2 for UHF/ROHF)
    * @param J The list of AO J matrices to build (Same size as D)
    */
   void build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) override;

   /**
    * @brief Prints information regarding CFMM run
    * 
    */
   void print_header() override;

};

class DFCFMM : public DirectDFJ {
  protected:
   /// CFMMTree used to calculate the three-center integrals
   std::shared_ptr<CFMMTree> df_cfmm_tree_;
   /// The gamma intermediate used in the DirectDFJ Algorithm
   std::vector<SharedMatrix> gamma;

  public:
   /**
    * @brief Construct a new DFCFMM object
    * 
    * @param primary The primary basis set used in DFCFMM
    * @param auxiliary The auxiliary basis set used in DFCFMM
    * @param options The options object
    */
   DFCFMM(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

   /**
    * @author Andy Jiang and David Poole, Georgia Tech, May 2022
    *
    * @brief Builds the J matrix using CFMM-Accelerated DFJ Algorithm
    * 
    * @param D The list of AO density matrixes to contract to form the J matrix (1 for RHF, 2 for UHF/ROHF)
    * @param J The list of AO J matrices to build (Same size as D)
    */
   void build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) override;

   /**
    * @brief Prints information regarding DFCFMM run
    * 
    */
   void print_header() override;

};

class LocalDFJ : public DirectDFJ {
  protected:
   /// The molecule referenced by the basis set
   std::shared_ptr<Molecule> molecule_;
   
   /// CFMMTree used to calculate the three-index integrals
   std::shared_ptr<CFMMTree> df_cfmm_tree_;
   /// Equation 19 in Sodt 2006 (Per atom, per number of density matrices)
   std::vector<std::vector<SharedMatrix>> rho_a_L_;
   /// Equation 20 in Sodt 2006
   std::vector<SharedMatrix> rho_tilde_K_;
   /// Equation 21 in Sodt 2006
   std::vector<SharedMatrix> J_tilde_L_;
   /// Equation 22 in Sodt 2006 (gammaP in regular Direct-DF-J)
   std::vector<SharedMatrix> J_L_;
   /// Equation 24 in Sodt 2006
   std::vector<std::vector<SharedMatrix>> I_KX_;

   /// The beginning of the cutoff region for bump function
   double r0_;
   /// The end of the cutoff region for the bump function, (beyond this ALL auxiliary functions are screened out)
   double r1_;
   /// Which auxiliary shells contribute to atom X
   std::vector<std::vector<int>> atom_to_aux_shells_;
   /// Which atoms does each auxiliary shell P contribute to
   std::vector<std::vector<int>> aux_shell_to_atoms_;
   /// What is the function offset of shell L in atom A's rho_A_L_ vector
   std::vector<std::unordered_map<int, int>> atom_aux_shell_function_offset_;
   ///  What is the corresponding bump function value of a given auxiliary shell per atom A
   std::vector<std::unordered_map<int, double>> atom_aux_shell_bump_value_;
   /// How many auxiliary basis functions contribute to an atom
   std::vector<int> naux_per_atom_;
   /// Which primary shells belong to atom X (in dense indexing M * nshell + N)
   std::vector<std::vector<int>> atom_to_pri_shells_;
   /// Bump matrix for atom X [Expressed as a list of block diagonal matrices for each atom] (Equations 15-17)
   std::vector<std::vector<SharedMatrix>> B_X_;
   /// Localized metric for atom X (Equation 17 in Sodt 2006)
   std::vector<SharedMatrix> J_X_;

   /// Setup atom_to_aux_shells_ and atom_to_pri_shells_
   void setup_local_regions();
   /// Build the approximate inverse for each atom (Equation 17)
   void build_atomic_inverse();

   /// Build rho_a_L_ (Equation 19)
   void build_rho_a_L(const std::vector<SharedMatrix>& D);
   /// Build rho_tilde_K_ (Equation 20)
   void build_rho_tilde_K();
   /// Build J_tilde_L_ (Equation 21)
   void build_J_tilde_L();
   /// Build J_L_ (Equation 22)
   void build_J_L(const std::vector<SharedMatrix>& D);
   /// Build I_KX_ (Equation 24)
   void build_I_KX();

   public:
     /**
      * @brief Construct a new LocalDFJ object
      * 
      * @param primary The primary basis set used in DFCFMM
      * @param auxiliary The auxiliary basis set used in DFCFMM
      * @param options The options object
      */
      LocalDFJ(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

     /**
      * @author Andy Jiang and David Poole, Georgia Tech, May 2022
      *
      * @brief Builds the J matrix using LocalDF Algorithm combined with CFMM (Sodt 2006)
      * 
      * @param D The list of AO density matrices to contract to form the J matrix (1 for RHF, 2 for UHF/ROHF)
      * @param J The list of AO J matrices to build (Same size as D)
      */
      void build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) override;

    /**
      * @brief Prints information regarding DFCFMM run
      * 
      */
      void print_header() override;
};

class LocalDFK : public LocalDFJ {
  public:
    /**
      * @brief Construct a new LocalDFJ object
      * 
      * @param primary The primary basis set used in DFCFMM
      * @param auxiliary The auxiliary basis set used in DFCFMM
      * @param options The options object
      */
      LocalDFK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

    /**
      * @author Andy Jiang and David Poole, Georgia Tech, May 2022
      *
      * @brief Builds the K matrix using LADF Algorithm (Sodt 2008)
      * 
      * @param D The list of AO density matrices to contract to form the J matrix (1 for RHF, 2 for UHF/ROHF)
      * @param J The list of AO J matrices to build (Same size as D)
      */
      void build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) override;

    /**
      * @brief Prints information regarding LocalDFK run
      * 
      */
      void print_header() override;
};

}

#endif
