#ifndef LIN_HELPER_H
#define LIN_HELPER_H

#include "psi4/pragma.h"

#include "psi4/libmints/twobody.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"

#include <memory>
#include <utility>
#include <vector>

namespace psi {

class LinearHelper {
  
  protected:
    // How many threads the calculation uses
    int nthread_;
    // Primary basis set
    std::shared_ptr<BasisSet> primary_;
    // Left right symmetric member from JK object
    bool lr_symmetric_;

  public:
    // Constructor
    LinearHelper(std::shared_ptr<BasisSet> primary, bool lr_symmetric);

    // CFMM J Build
    void build_cfmmJ(std::vector<std::shared_ptr<TwoBodyAOInt> >& ints, std::vector<std::shared_ptr<Matrix> >& D,
                  std::vector<std::shared_ptr<Matrix> >& J);

    // LinK Build
    void build_linK(std::vector<std::shared_ptr<TwoBodyAOInt> >& ints, std::vector<std::shared_ptr<Matrix> >& D,
                  std::vector<std::shared_ptr<Matrix> >& K);

};

}

#endif