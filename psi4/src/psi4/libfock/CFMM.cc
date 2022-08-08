#include "composite.h"

#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

using namespace psi;

namespace psi {

CFMM::CFMM(std::shared_ptr<BasisSet> primary, Options& options) : SplitJKBase(primary, options) {
    cfmmtree_ = std::make_shared<CFMMTree>(primary_, nullptr, options_);
    build_ints();
}

void CFMM::build_ints() {
    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    ints_.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    for (int thread = 1; thread < nthread_; thread++) {
        ints_.push_back(std::shared_ptr<TwoBodyAOInt>(ints_[0]->clone()));
    }
}

void CFMM::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("CFMM: J");

    cfmmtree_->build_J(ints_, D, J);

    timer_off("CFMM: J");
}

void CFMM::print_header() {
    if (print_) {
        outfile->Printf("  ==> Continuous Fast Multipole Method (CFMM) <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", cfmmtree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", cfmmtree_->nlevels());
        outfile->Printf("\n");
    }
}

}