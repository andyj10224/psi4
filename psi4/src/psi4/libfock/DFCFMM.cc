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

DFCFMM::DFCFMM(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, 
               Options& options) : DirectDFJ(primary, auxiliary, options) {

    df_cfmm_tree_ = std::make_shared<CFMMTree>(primary_, auxiliary_, options_);
}

void DFCFMM::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {
    timer_on("DFCFMM: J");

    int naux = auxiliary_->nbf();

    if (gamma.size() == 0) {
        gamma.resize(D.size());
        for (int i = 0; i < D.size(); i++) {
            gamma[i] = std::make_shared<Matrix>(naux, 1);
        }
    }

    // Build gammaP = (P|uv)Duv
    df_cfmm_tree_->df_set_contraction(ContractionType::DF_AUX_PRI);
    df_cfmm_tree_->build_J(ints_, D, gamma, Jmet_max_);

    // Solve for gammaQ => (P|Q)*gammaQ = gammaP
    for (int i = 0; i < D.size(); i++) {
        SharedMatrix Jmet_copy = Jmet_->clone();
        std::vector<int> ipiv(naux);

        C_DGESV(naux, 1, Jmet_copy->pointer()[0], naux, ipiv.data(), gamma[i]->pointer()[0], naux);
    }

    // Build Juv = (uv|Q) * gammaQ
    df_cfmm_tree_->df_set_contraction(ContractionType::DF_PRI_AUX);
    df_cfmm_tree_->build_J(ints_, gamma, J, Jmet_max_);

    timer_off("DFCFMM: J");
}

void DFCFMM::print_header() {
    if (print_) {
        outfile->Printf("  ==> CFMM-Accelerated Direct Density Fitted J <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Auxiliary Basis: %11s\n", auxiliary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", df_cfmm_tree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", df_cfmm_tree_->nlevels());
        outfile->Printf("\n");
    }
}

}