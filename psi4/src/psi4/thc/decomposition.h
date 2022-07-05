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

#ifndef PSI4_SRC_THC_DECOMPOSITION_H_
#define PSI4_SRC_THC_DECOMPOSITION_H_

#include <vector>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/wavefunction.h"

namespace psi {

/**
* A helper class that is used to break AO ERIs into a 
* "tensor-hypercontracted" form
*
* (pq|rs) = x_pI * x_qI * Z_IJ * x_rJ * x_sJ
*/
class THCDecomposer {
   protected:
    /// Primary Basis Set (to Build DF-ERI)
    std::shared_ptr<BasisSet> primary_;
    /// Auxiliary Basis Set (to Build DF-ERI)
    std::shared_ptr<BasisSet> auxiliary_;
    /// Contains a reference to elements of the ERI (Metric contracted)
    SharedMatrix Qpq_;

    /// Number of threads available for parallel computing
    int nthread_;

    /// Rank of the CP Decomposition
    size_t rank_;
    /// Matrix representing the Z_IJ connection factor in the THC decomposition
    SharedMatrix Z_;
    /// Matrix representing the polyadic vector of the first AO index
    SharedMatrix x_pI_;
    /// Matrix representing the polyadic vector of the second AO index
    SharedMatrix x_qI_;

    /// Perform the CP Decomposition on the three index ERIs, form Z, x_pI, and x_qI
    void perform_hypercontraction();

   public:
    THCDecomposer(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary);
    ~THCDecomposer() override;

    SharedMatrix Z() { return Z_; }
    SharedMatrix x_pI() { return x_pI_; }
    SharedMatrix x_qI() { return x_qI_; }

};

} // namespace psi

#endif