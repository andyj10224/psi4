/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2021 The Psi4 Developers.
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

#include "jk.h"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/twobody.h"

namespace psi {

class JBase {
  protected:
   /// Number of threads to use in the calculation
   int nthread_;
   /// Options
   Options& options_;
   /// Integral objects (one per thread)
   std::vector<std::shared_ptr<TwoBodyAOInt>> ints_;
   /// The primary basis set
   std::shared_ptr<BasisSet> primary_;
   /// Builds the integrals for the J class
   virtual void build_ints() = 0;

  public:
   /**
    * @brief Construct a new JBase object
    * All Split J algorithms extend from this class
    * 
    * @param primary The primary basis set used in the J computation
    * @param options The options object
    */
   JBase(std::shared_ptr<BasisSet> primary, Options& options);

   /**
    * @brief Builds the J matrix according to the class algorithm
    * 
    * @param D The list of density matrices
    * @param J The list of coulomb matrices
    */
   virtual void build_J(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) = 0;

};

class KBase {
  protected:
   /// Number of threads to use in the calculation
   int nthread_;
   /// Options
   Options& options_;
   /// Integral objects (one per thread)
   std::vector<std::shared_ptr<TwoBodyAOInt>> ints_;
   /// The primary basis set
   std::shared_ptr<BasisSet> primary_;
   /// Builds the integrals for the K class
   virtual void build_ints() = 0;

  public:
   /**
    * @brief Construct a new KBase object 
    * All Split K algorithms extend from this class
    * 
    * @param primary The primary basis set used in the J computation
    * @param options The options object
    */
   KBase(std::shared_ptr<BasisSet> primary, Options& options);

   /**
    * @brief Builds the K matrix according to the algorithm specified in this class
    * 
    * @param D The list of density matrices
    * @param K The list of exchange matrices
    */
   virtual void build_K(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) = 0;
};

class CompositeJK : public DirectJK {
  protected:
   std::string jtype_;
   std::string ktype_;
   std::shared_ptr<JBase> jalgo_;
   std::shared_ptr<KBase> kalgo_;
   std::shared_ptr<BasisSet> auxiliary_;
   std::vector<std::shared_ptr<TwoBodyAOInt>> ints_;
   int nthread_;
   void common_init();

  public:
   CompositeJK(std::shared_ptr<BasisSet> primary, Options& options);
   CompositeJK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

   void print_header() const override;
   void compute_JK() override;

};

class DirectDFJ : public JBase {
  protected:
   // Perform a Direct Density Fitted J instead of Direct J
   std::shared_ptr<BasisSet> auxiliary_;
   SharedMatrix Jinv_;
   double condition_ = 1.0e-12;

   // Form J^-1
   void form_Jinv();

   void build_ints() override;

  public:
   DirectDFJ(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);
   void build_J(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) override;

};

class LinK : public KBase {
  protected:
   // Schwarz Screening Cutoff
   double cutoff_;
   // LinK Sparsity Screening Cutoff
   double linK_ints_cutoff_;

   void build_ints() override;

  public:
   LinK(std::shared_ptr<BasisSet> primary, Options& options);
   void build_K(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& K) override;

};

}

#endif