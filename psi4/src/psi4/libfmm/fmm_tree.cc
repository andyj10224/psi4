#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_tree.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector3.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/multipoles.h"
#include "psi4/libmints/overlap.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libqt/qt.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <csignal>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

#define MAX_AM 10

namespace psi {

int ndigits(long n) {
    long absn = std::abs(n);
    int count = 1;
    while (absn >= 9) {
        absn /= 10;
        count += 1;
    }
    return count;
}

static HarmonicCoefficients harm_coef(MAX_AM, Regular);

/*
Distribution::Distribution(int P, int Q, int am1, int am2, Vector3 center1, Vector3 center2, double coef1, double coef2, double exp1, double exp2, 
                    double a1, double a2, double b1, double b2, double c1, double c2, int lmax) {
    basispair_ = std::make_pair<int, int>(P, Q);
    center_ = (coef1 * center1 + coef2 * center2) / (coef1 + coef2);
    exp_ = exp1 + exp2;

    Vector3 R_12 = center1 - center2;
    double r_12_sq = R_12.dot(R_12);

    coef_ = coef1 * coef2 * std::exp(-exp1 * exp2 / (exp1 + exp2) * r_12_sq);

    lx_ = a1 + a2;
    ly_ = b1 + b2;
    lz_ = c1 + c2;

    rext_ = ERFCI10 / std::sqrt(exp_);

    std::string basistype = "PRIMARY";
    std::shared_ptr<Molecule> dummy_mol = std::make_shared<Molecule>();

    std::map<std::string, std::map<std::string, std::vector<ShellInfo>>> ecp_shell_map;
    std::map<std::string, std::map<std::string, std::vector<ShellInfo>>> shell_map1;
    std::map<std::string, std::map<std::string, std::vector<ShellInfo>>> shell_map2;

    std::vector<double> coefvec1 {coef1};
    std::vector<double> expvec1 {exp1};

    std::shared_ptr<ShellInfo> shellinfo1 = std::make_shared<ShellInfo>(am1, coefvec1, expvec1, Cartesian, Normalized);

    std::vector<double> coefvec2 {coef2};
    std::vector<double> expvec2 {exp2};

    std::shared_ptr<ShellInfo> shellinfo2 = std::make_shared<ShellInfo>(am2, coefvec2, expvec2, Cartesian, Normalized);

}
*/

Distribution::Distribution(std::shared_ptr<BasisSet> basisset, int P, int Q, double rext, double coef, double exp, Vector3 center, int lmax) {
    basisset_ = basisset;
    shellpair_ = std::make_pair<int, int>(P, Q);
    center_ = center;
    coef_ = coef;
    exp_ = exp;
    rext_ = rext;
    lmax_ = lmax;

    lP_ = basisset_->shell(P).am();
    lQ_ = basisset_->shell(Q).am();

    mi_recur_ = new ObaraSaikaTwoCenterMIRecursion(lP_ + 2, lQ_ + 2, lmax_);

    mpoles_ = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
    Vff_ = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);
}

void Distribution::compute_mpoles() {

    int am1 = s1.am();
    int am2 = s2.am();

    // The number of bf components in each shell pair
    int stride = INT_NCART(am1) * INT_NCART(am2);

    // Number of multipole components to compute
    int nmpole = (lmax_ + 1) * (lmax_ + 2) * (lmax_ + 3) / 6 - 1;

    // memset(buffer_, 0, nmpole * INT_NCART(am1) * INT_NCART(am2) * sizeof(double));
    std::vector<std::vector<double>>& mpole_buffer = mpoles_.get_multipoles();

    // Buffers to hold the {x,y,z} moments
    double *Xpowers = new double[lmax_ + 1];
    double *Ypowers = new double[lmax_ + 1];
    double *Zpowers = new double[lmax_ + 1];

    const GaussianShell& s1 = basisset_->shell(shellpair_.first);
    const GaussianShell& s2 = basisset_->shell(shellpair_.second);

    int p_start = s1.start();
    int q_start = s2.start();

    double A[3], B[3];
    A[0] = s1.center()[0];
    A[1] = s1.center()[1];
    A[2] = s1.center()[2];
    B[0] = s2.center()[0];
    B[1] = s2.center()[1];
    B[2] = s2.center()[2];

    // compute intermediates
    double AB2 = 0.0;
    AB2 += (A[0] - B[0]) * (A[0] - B[0]);
    AB2 += (A[1] - B[1]) * (A[1] - B[1]);
    AB2 += (A[2] - B[2]) * (A[2] - B[2]);

    double ***x = mi_recur_.x();
    double ***y = mi_recur_.y();
    double ***z = mi_recur_.z();

    double PA[3], PB[3];
    double P[3];

    // P[0] = (a1 * A[0] + a2 * B[0]) * oog;
    // P[1] = (a1 * A[1] + a2 * B[1]) * oog;
    // P[2] = (a1 * A[2] + a2 * B[2]) * oog;

    P[0] = center_[0];
    P[1] = center_[1];
    P[2] = center_[2];

    PA[0] = P[0] - A[0];
    PA[1] = P[1] - A[1];
    PA[2] = P[2] - A[2];
    PB[0] = P[0] - B[0];
    PB[1] = P[1] - B[1];
    PB[2] = P[2] - B[2];

    double PCx = (P[0] - center_[0]);
    double PCy = (P[1] - center_[1]);
    double PCz = (P[2] - center_[2]);

    // double d_coef = p_coef * q_coef * std::exp(-p_exp * q_exp / (p_exp + q_exp) * r_PQ_sq);

    double over_pf = coef_ * sqrt(M_PI / (exp_)) * (M_PI / exp_);

    // Do recursion
    mi_recur_.compute(PA, PB, exp_, am1 + 2, am2 + 2);
    int bf = 0;

    int p = p_start;
    // Basis function A.M. components on center 1
    for (int ii = 0; ii <= am1; ii++) {
        int lx1 = am1 - ii;
        for (int lz1 = 0; lz1 <= ii; lz1++) {
            int ly1 = ii - lz1;

            int q = q_start;
            // Basis function A.M. components on center 2
            for (int kk = 0; kk <= am2; kk++) {
                int lx2 = am2 - kk;
                for (int lz2 = 0; lz2 <= kk; lz2++) {
                    int ly2 = kk - lz2;

                    // int chunk = 0;

                    /*
                     * Define X_C = X_P + X_PC, then expand X_C in powers,
                     * up to the requested order; these will be combined to
                     * form the total moment operators below
                     */
                    // Xpowers[0] = (l) * X_C^0 * X_PC^0
                    //              (0)
                    Xpowers[0] = x[lx1][lx2][0];
                    Ypowers[0] = y[ly1][ly2][0];
                    Zpowers[0] = z[lz1][lz2][0];

                    int running_index = 0;
                    for (int l = 1; l <= order_; ++l) {
                        int ncl = ncart(l);

                        double px = PCx;
                        double py = PCy;
                        double pz = PCz;
                        // Xpowers[l] = (l) * X_C^0 * X_PC^0  +  (l) * X_C^0 * X_PC^l
                        //              (0)                      (l)
                        Xpowers[l] = x[lx1][lx2][l] + pow(px, l) * x[lx1][lx2][0];
                        Ypowers[l] = y[ly1][ly2][l] + pow(py, l) * y[ly1][ly2][0];
                        Zpowers[l] = z[lz1][lz2][l] + pow(pz, l) * z[lz1][lz2][0];
                        for (int i = 1; i < l; ++i) {
                            double coef = (double)binomial(l, i);
                            // Xpowers[l] = (l) * X_C^(l-i) * X_PC^i
                            //              (i)
                            Xpowers[l] += coef * px * x[lx1][lx2][l - i];
                            Ypowers[l] += coef * py * y[ly1][ly2][l - i];
                            Zpowers[l] += coef * pz * z[lz1][lz2][l - i];
                            px *= PCx;
                            py *= PCy;
                            pz *= PCz;
                        }

                        /*
                         * Loop over the multipole components, and combine the moments computed above
                         */

                        for (int m = -l; m <= l; m++) {
                            int mu = m_addr(m);
                            std::unordered_map<int, double>& mpole_terms = harm_coef.get_terms(l, mu);

                            int powdex = 0;
                            for (int ii = 0; ii <= l; ii++) {
                                int lx = l - ii;
                                for (int lz = 0; lz <= ii; lz++) {
                                    int ly = ii - lz;
                                    int ind = lx * ncl * ncl + ly * ncl + lz;

                                    double val = Xpowers[lx] * Ypowers[ly] * Zpowers[lz] * over_pf;

                                    if (mpole_terms.count(ind)) {
                                        double hcoef = mpole_terms[ind];
                                        int abcindex = powdex + running_index;

                                        for (int N = 0; N < D_.size(); N++) {
                                            mpole_buffer[l][mu] += pow(-1.0, (double) l + 1) * hcoef * 2.0 * D_[N]->get(p, q) * val;
                                        }

                                    }
                                    // mpole_buffer_[chunk * stride + bf] -= val;
                                    // ++chunk;
                                    powdex += 1;
                                }
                            }
                        }

                        running_index += ncl;
                    }  // End loop over l
                    // bf++;
                    q += 1;
                }
            }  // End loop over shell 2 A.M. components
        p += 1;
        }
    }  // End loop over shell 1 A.M. components

    delete[] Xpowers;
    delete[] Ypowers;
    delete[] Zpowers;
}

CFMMBox::CFMMBox(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
        std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int lmax) {
    D_ = D;
    J_ = J;

    double min_dim = molecule->x(0);
    double max_dim = molecule->x(0);

    for (int atom = 0; atom < molecule->natom(); atom++) {
        double x = molecule->x(atom);
        double y = molecule->y(atom);
        double z = molecule->z(atom);
        min_dim = std::min(x, min_dim);
        min_dim = std::min(y, min_dim);
        min_dim = std::min(z, min_dim);
        max_dim = std::max(x, max_dim);
        max_dim = std::max(y, max_dim);
        max_dim = std::max(z, max_dim);
    }

    max_dim += 0.1; // Add a small buffer to the box

    Vector3 origin = Vector3(min_dim, min_dim, min_dim);
    double length = (max_dim - min_dim);

    common_init(nullptr, molecule, basisset, origin, length, 0, lmax);
}

CFMMBox::CFMMBox(CFMMBox* parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                    std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, Vector3 origin, double length, int level, int lmax) {
    D_ = D;
    J_ = J;
    common_init(parent, molecule, basisset, origin, length, level, lmax);
}

void CFMMBox::common_init(CFMMBox* parent, std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                        Vector3 origin, double length, int level, int lmax) {
    parent_ = parent;
    molecule_ = molecule;
    basisset_ = basisset;
    origin_ = origin;
    length_ = length;
    level_ = level;
    lmax_ = lmax;

    nthread_ = 1;

#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

    center_ = origin_ + Vector3(length_ / 2, length_ / 2, length_ / 2);
    children_.resize(8, nullptr);

    // Make the multipole coefficients
    if (!parent_) {
        mpole_coefs_ = std::make_shared<HarmonicCoefficients>(lmax_, Regular);

    } else {
        mpole_coefs_ = parent_->mpole_coefs_;
    }

    if (!parent_) {
        for (int atom = 0; atom < molecule_->natom(); atom++) {
            atoms_.push_back(atom);
        }
    } else {
        for (int ind = 0; ind < parent_->atoms_.size(); ind++) {
            int atom = parent_->atoms_[ind];
            double x = molecule_->x(atom);
            double y = molecule_->y(atom);
            double z = molecule_->z(atom);

            bool x_good = (x >= origin_[0] && x < origin_[0] + length_);
            bool y_good = (y >= origin_[1] && y < origin_[1] + length_);
            bool z_good = (z >= origin_[2] && z < origin_[2] + length_);

            if (x_good && y_good && z_good) {
                atoms_.push_back(atom);
            }
        }
    }

    ws_max_ = 0;
    distributions_.resize(1);

    mpoles_ = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
    Vff_ = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);
    ff_energy_ = 0.0;

}

void CFMMBox::add_distribution(int ws, int P, int Q, double rext, double coef, double exp, Vector3 center, int lmax) {
    if (ws > ws_max_) {
        ws_max_ = ws;
        distributions_.resize(ws_max_ + 1);
    }
    distributions_[ws].push_back(new Distribution(basisset_, P, Q, rext, coef, exp, center, lmax));
}

void CFMMBox::add_distribution(int ws, Distribution* dist) {
    if (ws > ws_max_) {
        ws_max_ = ws;
        distributions_.resize(ws_max_ + 1);
    }
    distributions_[ws].push_back(dist);
}

void CFMMBox::radix_sort_distributions(int ws) {
    // Number of digits of each phase of the sort
    // Resolution for floats is 0.001 bohr
    int zdig = 1, ydig = 1, xdig = 1, rdig = 1;

    for (Distribution* dist : distributions_[ws]) {
        Vector3 center = dist->get_center();
        double x = center[0];
        double y = center[1];
        double z = center[2];
        double r = dist->get_rext();

        zdig = std::max(zdig, ndigits(z * 1000));
        ydig = std::max(ydig, ndigits(y * 1000));
        xdig = std::max(xdig, ndigits(x * 1000));
        rdig = std::max(rdig, ndigits(r * 1000));
    }

    std::vector<std::vector<Distribution *>> buckets(19);

    // Sort by z coordinates
    long curr10 = 10;

    for (int iter = 0; iter < zdig; iter++) {
        for (int ind = 0; ind < distributions_[ws].size(); ind++) {
            int dig = (long)(distributions_[ws][ind]->get_center()[2] * 1000) / curr10;
            buckets[dig + 9].push_back(distributions_[ws][ind]);
        }
        curr10 *= 10;
    }

    int idx = 0;
    for (int b = 0; b < 19; b++) {
        while (buckets[b].size() > 0) {
            dist[idx] = buckets[b].back();
            buckets[b].pop_back();
            idx += 1;
        }
    }

    // Sort by y coordinates
    curr10 = 10;

    for (int iter = 0; iter < ydig; iter++) {
        for (int ind = 0; ind < distributions_[ws].size(); ind++) {
            int dig = (long)(distributions_[ws][ind]->get_center()[1] * 1000) / curr10;
            buckets[dig + 9].push_back(distributions_[ws][ind]);
        }
        curr10 *= 10;
    }

    idx = 0;
    for (int b = 0; b < 19; b++) {
        while (buckets[b].size() > 0) {
            dist[idx] = buckets[b].back();
            buckets[b].pop_back();
            idx += 1;
        }
    }

    // Sort by x coordinates
    curr10 = 10;

    for (int iter = 0; iter < xdig; iter++) {
        for (int ind = 0; ind < distributions_[ws].size(); ind++) {
            int dig = (long)(distributions_[ws][ind]->get_center()[0] * 1000) / curr10;
            buckets[dig + 9].push_back(distributions_[ws][ind]);
        }
        curr10 *= 10;
    }

    idx = 0;
    for (int b = 0; b < 19; b++) {
        while (buckets[b].size() > 0) {
            dist[idx] = buckets[b].back();
            buckets[b].pop_back();
            idx += 1;
        }
    }

    // Sort by radial extents
    curr10 = 10;

    for (int iter = 0; iter < rdig; iter++) {
        for (int ind = 0; ind < distributions_[ws].size(); ind++) {
            int dig = (long)(distributions_[ws][ind]->get_rext() * 1000) / curr10;
            buckets[dig + 9].push_back(distributions_[ws][ind]);
        }
        curr10 *= 10;
    }

    idx = 0;
    for (int b = 0; b < 19; b++) {
        while (buckets[b].size() > 0) {
            dist[idx] = buckets[b].back();
            buckets[b].pop_back();
            idx += 1;
        }
    }
}

void CFMMBox::form_distributions() {
    
    // Create distributions and add them to the list
    for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);
        Vector3 Pcenter = molecule_->xyz(Patom);

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshell = basisset_->nshell_on_center(Qatom);
            Vector3 Qcenter = molecule_->xyz(Qatom);
            Vector3 R_PQ = Pcenter - Qcenter;
            double r_PQ_sq = R_PQ.dot(R_PQ);

            for (int P = Pstart; P < Pstart + nPshell; P++) {
                const GaussianShell& Pshell = basisset_->shell(P);
                int nprim_p = Pshell.nprimitive();
                int p_start = Pshell.start();
                int pl = Pshell.am();

                for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                    const GaussianShell& Qshell = basisset_->shell(Q);
                    int nprim_q = Qshell.nprimitive();
                    int q_start = Qshell.start();
                    int ql = Qshell.am();

                    for (int pp = 0; pp < nprim_p < pp++) {
                        double p_coef = Pshell.coef(pp);
                        double p_exp = Pshell.exp(pp);

                        for (int qp = 0; qp < nprim_q; qp++) {
                            double q_coef = Qshell.coef(qp);
                            double q_exp = Qshell.exp(qp);

                            // Distribution information from product of primative p and primative q
                            double d_coef = p_coef * q_coef * std::exp(-p_exp * q_exp / (p_exp + q_exp) * r_PQ_sq);
                            double d_exp = p_exp + q_exp;
                            Vector3 Dcenter = (p_coef * Pcenter + q_coef * Qcenter) / (p_coef + q_coef);

                            double r_ext = ERFCI10 / std::sqrt(d_exp);
                            int ext = 2 * std::ceil(r_ext / length_);
                            int ws_d = std::max(2, ext);

                            add_distribution(ws_d, P, Q, r_ext, d_coef, d_exp, Dcenter, lmax_);

                            /*
                            int poff = 0;
                            for (int pi = 0; pi <= pl; pi++) {
                                int pa = pl - pi;
                                for (int pj = 0; pj <= pi; pj++) {
                                    int pb = pi - pj;
                                    int pc = pj;

                                    int qoff = 0;
                                    for (int qi = 0; qi <= ql; qi++) {
                                        int qa = ql - qi;
                                        for (int qj = qj <= qi; qj++) {
                                            int qb = qi - qj;
                                            int qc = qj;

                                            add_distribution(ws_d, p_start + poff, q_start + qoff, r_ext, d_coef, d_exp, 
                                                            Dcenter, pa + qa, pb + qb, pc + qc, lmax_);

                                            qoff += 1;
                                        }
                                    }

                                    poff += 1;
                                }
                            }
                            */

                        }
                    }
                }
            }
        }
    }

    // Radix Sort Distributions after they are added
    for (int ws = 2; ws <= ws_max_; ws += 2) {
        radix_sort_distributions(ws);
    }
}

void CFMMBox::form_distributions_from_children() {

    for (CFMMBox* child : children_) {
        for (int ws = 0; ws <= child->ws_max_; ws++) {
            for (Distribution* dist : child->distributions_[ws]) {
                add_distribution(std::ceil((double)ws/2), dist);
            }
        }
    }

    for (int ws = 0; ws <= ws_max_; ws++) {
        radix_sort_distributions(ws);
    }

}

void CFMMBox::set_nf_lff() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::set_nf_lff()");

    // Parent is not a nullpointer
    if (parent_) {
        // Siblings of this box
        for (CFMMBox* sibling : parent_->children_) {
            if (sibling == this) continue;
            if (sibling->natom() == 0) continue;

            Vector3 Rab = center_ - sibling->center_;
            double dist = Rab.norm();

            if (dist <= ws_ * length_ * std::sqrt(3.0)) {
                near_field_.push_back(sibling);
            } else {
                local_far_field_.push_back(sibling);
            }
        }

        // Parent's near field (Cousins)
        for (CFMMBox* uncle : parent_->near_field_) {
            if (uncle->natom() == 0) continue;

            for (CFMMBox* cousin : uncle->children_) {
                if (cousin->natom() == 0) continue;

                Vector3 Rab = center_ - cousin->center_;
                double dist = Rab.norm();

                if (dist <= ws_ * length_ * std::sqrt(3.0)) {
                    near_field_.push_back(cousin);
                } else {
                    local_far_field_.push_back(cousin);
                }
            }
        }
    }

    timer_off("CFMMBox::set_nf_lff()");
}

void CFMMBox::make_children() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::make_children()");

    for (int c = 0; c < 8; c++) {
        double half_length = length_ / 2.0;
        int dx = (c & 4) >> 2; // 0 or 1
        int dy = (c & 2) >> 1; // 0 or 1
        int dz = (c & 1) >> 0; // 0 or 1

        Vector3 child_origin = origin_ + Vector3(half_length * dx, half_length * dy, half_length * dz);

        CFMMBox* child = new CFMMBox(this, molecule_, basisset_, D_, J_, child_origin, half_length, level_+1, lmax_);
        children_[c] = child;
    }

    timer_off("CFMMBox::make_children()");

}

void CFMMBox::compute_mpoles() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_mpoles()");

    std::shared_ptr<IntegralFactory> int_factory = std::make_shared<IntegralFactory>(basisset_);

    std::vector<std::shared_ptr<OneBodyAOInt>> mpints(nthread_);
    std::vector<std::shared_ptr<OneBodyAOInt>> oints(nthread_);

    for (int thread = 0; thread < nthread_; thread++) {
        mpints[thread] = std::shared_ptr<OneBodyAOInt>(int_factory->ao_multipoles(lmax_));
        oints[thread] = std::shared_ptr<OneBodyAOInt>(int_factory->ao_overlap());
        mpints[thread]->set_origin(center_); // + Vector3(10.0, 0.0, 0.0));
    }

    int n_mult = (lmax_ + 1) * (lmax_ + 2) * (lmax_ + 3) / 6 - 1;
    int nbf = basisset_->nbf();

    // Compute multipole integrals for all atoms in the shell pair
#pragma omp parallel for
    for (int Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshell = basisset_->nshell_on_center(Qatom);

            for (int P = Pstart; P < Pstart + nPshell; P++) {
                const GaussianShell& Pshell = basisset_->shell(P);
                int p_start = Pshell.start();
                int num_p = Pshell.nfunction();

                for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                    const GaussianShell& Qshell = basisset_->shell(Q);
                    int q_start = Qshell.start();
                    int num_q = Qshell.nfunction();

                    mpints[thread]->compute_shell(P, Q);
                    const double *mpole_buffer = mpints[thread]->buffer();

                    oints[thread]->compute_shell(P, Q);
                    const double *overlap_buffer = oints[thread]->buffer();

                    // Compute multipoles
                    for (int p = p_start; p < p_start + num_p; p++) {
                        int dp = p - p_start;

                        for (int q = q_start; q < q_start + num_q; q++) {
                            int dq = q - q_start;

                            std::shared_ptr<RealSolidHarmonics> pq_mpole_buff = std::make_shared<RealSolidHarmonics>(lmax_, mpints[thread]->origin(), Regular);

                            for (int N = 0; N < D_.size(); N++) {

                                pq_mpole_buff->add(0, 0, -2.0 * D_[N]->get(p, q) * overlap_buffer[dp * num_q + dq]);

                                int running_index = 0;
                                for (int l = 1; l <= lmax_; l++) {
                                    int ncl = ncart(l);

                                    /*
                                    int powind = 0;
                                    for (int ii = 0; ii <= l; ii++) {
                                        int a = l - ii;
                                        for (int jj = 0; jj <= ii; jj++) {
                                            int b = ii - jj;
                                            int c = jj;
                                            int ind = a * ncl * ncl + b * ncl + c;

                                            outfile->Printf("   POWDEX: %d, A: %d, B: %d, C: %d\n", powind, a, b, c);
                                            outfile->Printf("   ICART:  %d, A: %d, B: %d, C: %d\n", icart(a, b, c), a, b, c);
                                                
                                            powind += 1;
                                        }
                                    }
                                    */
                                    

                                    for (int m = -l; m <= l; m++) {
                                        int mu = m_addr(m);
                                        std::unordered_map<int, double>& mpole_terms = mpole_coefs_->get_terms(l, mu);

                                        int powdex = 0;
                                        for (int ii = 0; ii <= l; ii++) {
                                            int a = l - ii;
                                            for (int jj = 0; jj <= ii; jj++) {
                                                int b = ii - jj;
                                                int c = jj;
                                                int ind = a * ncl * ncl + b * ncl + c;

                                                if (mpole_terms.count(ind)) {
                                                    double coef = mpole_terms[ind];
                                                    int abcindex = powdex + running_index;
                                                    // outfile->Printf("   L: %d, M: %d, A: %d, B: %d, C: %d, COEF: %8.5f\n", l, m, a, b, c, coef);
                                                    // coef *= factorial(a) * factorial(b) * factorial(c);
                                                    pq_mpole_buff->add(l, mu, pow(-1.0, (double) l) * 2.0 * D_[N]->get(p, q) * coef * mpole_buffer[abcindex * num_p * num_q + dp * num_q + dq]);
                                                }
                                                powdex += 1;
                                            }
                                        }

                                    } // end m loop

                                    running_index += ncl;
                                } // end l loop
                            } // end N loop
                            // ->translate(center_ + Vector3(10.0, 0.0, 0.0))
                            mpoles_->add(pq_mpole_buff); // ->translate(center_ + Vector3(10.0, 0.0, 0.0)));
                        } // end q loop
                    } // end p loop
                } // end Q
            } // end P
        } // end Qtask
    } // end Ptask

    timer_off("CFMMBox::compute_mpoles()");

}

void CFMMBox::compute_mpoles_from_children() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_mpoles_from_children()");

    int nbf = basisset_->nbf();

    for (CFMMBox* child : children_) {
        if (child->atoms_.size() == 0) continue;

        std::shared_ptr<RealSolidHarmonics> child_mpoles = child->mpoles_->translate(center_);
        mpoles_->add(child_mpoles);
    }

    /*

#pragma omp parallel for
        for (int Ptask = 0; Ptask < child->atoms_.size(); Ptask++) {
            int Patom = child->atoms_[Ptask];
            int Pstart = basisset_->shell_on_center(Patom, 0);
            int nPshell = basisset_->nshell_on_center(Patom);

            for (int Qtask = 0; Qtask < child->atoms_.size(); Qtask++) {
                int Qatom = child->atoms_[Qtask];
                int Qstart = basisset_->shell_on_center(Qatom, 0);
                int nQshell = basisset_->nshell_on_center(Qatom);

                for (int P = Pstart; P < Pstart + nPshell; P++) {
                    const GaussianShell& p_shell = basisset_->shell(P);
                    int p_start = p_shell.start();
                    int num_p = p_shell.nfunction();

                    for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                        const GaussianShell& q_shell = basisset_->shell(Q);
                        int q_start = q_shell.start();
                        int num_q = q_shell.nfunction();

                        for (int p = p_start; p < p_start + num_p; p++) {
                            for (int q = q_start; q < q_start + num_q; q++) {
                                std::shared_ptr<RealSolidHarmonics> child_mpoles = child->mpoles_[p * nbf + q]->translate(center_);
                                mpoles_[p * nbf + q]->add(child_mpoles);
                            } // End q
                        } // End p
                    } // End Q
                } // End P
            } // End Qtask
        } // End Ptask
    } // End children
    */

    timer_off("CFMMBox::compute_mpoles_from_children()");

}

void CFMMBox::compute_far_field_vector() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_far_field_vector()");

    int nbf = basisset_->nbf();

    for (CFMMBox* box : local_far_field_) {
        std::shared_ptr<RealSolidHarmonics> box_mpoles = box->mpoles_;
        // The far field effect the boxes have on this particular box
        std::shared_ptr<RealSolidHarmonics> far_field = box_mpoles->far_field_vector(center_);
        Vff_->add(far_field);
    }

        /*
#pragma omp parallel for
        for (int Ptask = 0; Ptask < box->atoms_.size(); Ptask++) {
            int Patom = box->atoms_[Ptask];
            int Pstart = basisset_->shell_on_center(Patom, 0);
            int nPshell = basisset_->nshell_on_center(Patom);

            for (int Qtask = 0; Qtask < box->atoms_.size(); Qtask++) {
                int Qatom = box->atoms_[Qtask];
                int Qstart = basisset_->shell_on_center(Qatom, 0);
                int nQshell = basisset_->nshell_on_center(Qatom);

                for (int P = Pstart; P < Pstart + nPshell; P++) {
                    const GaussianShell& p_shell = basisset_->shell(P);
                    int p_start = p_shell.start();
                    int num_p = p_shell.nfunction();

                    for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                        const GaussianShell& q_shell = basisset_->shell(Q);
                        int q_start = q_shell.start();
                        int num_q = q_shell.nfunction();

                        for (int ind = 0; ind < D_.size(); ind++) {
                            for (int p = p_start; p < p_start + num_p; p++) {
                                for (int q = q_start; q < q_start + num_q; q++) {
                                    std::shared_ptr<RealSolidHarmonics> box_mpoles = box->mpoles_[p * nbf + q];
                                    // The far field effect the boxes have on this particular box
                                    std::shared_ptr<RealSolidHarmonics> far_field = box_mpoles->far_field_vector(center_);
                                    far_field->scale(D_[ind]->get(p, q));

                                    Vff_->add(far_field);
                                }
                            } // q
                        } // p
                    } // Q
                } // P
            } // Qtask
        } // Ptask
    } // box

    */

    // Parent is not null
    if (parent_) {
        Vff_->add(parent_->Vff_->translate(center_));
    }

    timer_off("CFMMBox::compute_far_field_vector()");

}


void CFMMBox::compute_self_J() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_self_J()");

    std::vector<std::shared_ptr<TwoBodyAOInt>> ints;

    std::shared_ptr<IntegralFactory> factory = std::make_shared<IntegralFactory>(basisset_);
    std::shared_ptr<TwoBodyAOInt> eri = std::shared_ptr<TwoBodyAOInt>(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    ints.push_back(eri);

    for (int thread = 1; thread < nthread_; thread++) {
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(eri->clone()));
    }

    size_t atom_max_nshell = 0L;
    for (size_t Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        size_t size = 0L;
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        for (int P = Pstart; P < Pstart + nPshell; P++) {
            size += basisset_->shell(P).nfunction();
        }

        atom_max_nshell = std::max(atom_max_nshell, size);
    }

    // Get significant atom pairs (PQ|-style
    std::vector<std::pair<int, int>> atom_pairs;
    for (size_t Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            if (Qtask > Ptask) continue;
            bool found = false;

            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshell = basisset_->nshell_on_center(Qatom);

            for (int P = Pstart; P < Pstart + nPshell; P++) {
                for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                    if (ints[0]->shell_pair_significant(P, Q)) {
                        found = true;
                        atom_pairs.push_back(std::pair<int, int>(Patom, Qatom));
                        break;
                    }
                }
                if (found) break;
            }
        }
    }

    size_t natom_pair = atom_pairs.size();
    size_t natom_pair2 = natom_pair * natom_pair;

    int nbf = basisset_->nbf();
    int nshell = basisset_->nshell();

    // => Intermediate Buffers <= //
    std::vector<std::vector<std::shared_ptr<Matrix>>> JT;
    for (int thread = 0; thread < nthread_; thread++) {
        std::vector<std::shared_ptr<Matrix>> J2;
        for (size_t ind = 0; ind < D_.size(); ind++) {
            J2.push_back(std::make_shared<Matrix>("JT", 2 * atom_max_nshell, atom_max_nshell));
        }
        JT.push_back(J2);
    }

    // outfile->Printf("   ATOMS SIZE: %d\n", atoms_.size());

    // Self-interactions
#pragma omp parallel for
    for (size_t task = 0L; task < natom_pair2; task++) {
        size_t task1 = task / natom_pair;
        size_t task2 = task % natom_pair;

        int Patom = atom_pairs[task1].first;
        int Qatom = atom_pairs[task1].second;
        int Ratom = atom_pairs[task2].first;
        int Satom = atom_pairs[task2].second;

        if (Ratom > Patom) continue;

        int Pstart = basisset_->shell_on_center(Patom, 0);
        int Qstart = basisset_->shell_on_center(Qatom, 0);
        int Rstart = basisset_->shell_on_center(Ratom, 0);
        int Sstart = basisset_->shell_on_center(Satom, 0);

        int nPshell = basisset_->nshell_on_center(Patom);
        int nQshell = basisset_->nshell_on_center(Qatom);
        int nRshell = basisset_->nshell_on_center(Ratom);
        int nSshell = basisset_->nshell_on_center(Satom);

        int dPsize = (Pstart + nPshell >= nshell ? nbf : basisset_->shell(Pstart + nPshell).start())
                    - basisset_->shell(Pstart).start();
        int dQsize = (Qstart + nQshell >= nshell ? nbf : basisset_->shell(Qstart + nQshell).start())
                    - basisset_->shell(Qstart).start();
        int dRsize = (Rstart + nRshell >= nshell ? nbf : basisset_->shell(Rstart + nRshell).start())
                    - basisset_->shell(Rstart).start();
        int dSsize = (Sstart + nSshell >= nshell ? nbf : basisset_->shell(Sstart + nSshell).start())
                    - basisset_->shell(Sstart).start();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        bool touched = false;
        for (int P = Pstart; P < Pstart + nPshell; P++) {

            for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                if (Q > P) continue;
                if (!ints[0]->shell_pair_significant(P, Q)) continue;

                for (int R = Rstart; R < Rstart + nRshell; R++) {

                    for (int S = Sstart; S < Sstart + nSshell; S++) {
                        if (S > R) continue;
                        if (R * nshell + S > P * nshell + Q) continue;
                        if (!ints[0]->shell_pair_significant(R, S)) continue;
                        if (!ints[0]->shell_significant(P, Q, R, S)) continue;

                        if (ints[thread]->compute_shell(P, Q, R, S) == 0) continue;
                            
                        const double *buffer = ints[thread]->buffer();

                        int Psize = basisset_->shell(P).nfunction();
                        int Qsize = basisset_->shell(Q).nfunction();
                        int Rsize = basisset_->shell(R).nfunction();
                        int Ssize = basisset_->shell(S).nfunction();

                        int Poff = basisset_->shell(P).start();
                        int Qoff = basisset_->shell(Q).start();
                        int Roff = basisset_->shell(R).start();
                        int Soff = basisset_->shell(S).start();

                        int Poff2 = basisset_->shell(P).start() - basisset_->shell(Pstart).start();
                        int Qoff2 = basisset_->shell(Q).start() - basisset_->shell(Qstart).start();
                        int Roff2 = basisset_->shell(R).start() - basisset_->shell(Rstart).start();
                        int Soff2 = basisset_->shell(S).start() - basisset_->shell(Sstart).start();

                        for (size_t ind = 0; ind < D_.size(); ind++) {
                            double** Dp = D_[ind]->pointer();
                            double** JTp = JT[thread][ind]->pointer();
                            const double* buffer2 = buffer;

                            if (!touched) {
                                ::memset((void*)JTp[0L * atom_max_nshell], '\0', dPsize * dQsize * sizeof(double));
                                ::memset((void*)JTp[1L * atom_max_nshell], '\0', dRsize * dSsize * sizeof(double));
                            }

                            double* J1p = JTp[0L * atom_max_nshell];
                            double* J2p = JTp[1L * atom_max_nshell];

                            double prefactor = 1.0;
                            if (P == Q) prefactor *= 0.5;
                            if (R == S) prefactor *= 0.5;
                            if (P == R && Q == S) prefactor *= 0.5;

                            for (int p = 0; p < Psize; p++) {
                                for (int q = 0; q < Qsize; q++) {
                                    for (int r = 0; r < Rsize; r++) {
                                        for (int s = 0; s < Ssize; s++) {
                                            J1p[(p + Poff2) * dQsize + q + Qoff2] +=
                                                prefactor * (Dp[r + Roff][s + Soff] + Dp[s + Soff][r + Roff]) *
                                                (*buffer2);
                                            J2p[(r + Roff2) * dSsize + s + Soff2] +=
                                                prefactor * (Dp[p + Poff][q + Qoff] + Dp[q + Qoff][p + Poff]) *
                                                (*buffer2);
                                            buffer2++;
                                        } // s
                                    } // r
                                } // q
                            } // p
                        } // ind
                        touched = true;

                    } // S
                } // R
            } // Q 
        } // P

        if (!touched) continue;

        // => Stripe out <= //
        for (size_t ind = 0; ind < D_.size(); ind++) {
            double** JTp = JT[thread][ind]->pointer();
            double** Jp = J_[ind]->pointer();

            double* J1p = JTp[0L * atom_max_nshell];
            double* J2p = JTp[1L * atom_max_nshell];
            

            // > J_PQ < //

            for (int P2 = 0; P2 < nPshell; P2++) {
                for (int Q2 = 0; Q2 < nQshell; Q2++) {
                    int P = Pstart + P2;
                    int Q = Qstart + Q2;
                    int Psize = basisset_->shell(P).nfunction();
                    int Qsize = basisset_->shell(Q).nfunction();
                    int Poff = basisset_->shell(P).function_index();
                    int Qoff = basisset_->shell(Q).function_index();

                    int Poff2 = basisset_->shell(P).start() - basisset_->shell(Pstart).start();
                    int Qoff2 = basisset_->shell(Q).start() - basisset_->shell(Qstart).start();

                    for (int p = 0; p < Psize; p++) {
                        for (int q = 0; q < Qsize; q++) {
#pragma omp atomic
                            Jp[p + Poff][q + Qoff] += 2.0 * J1p[(p + Poff2) * dQsize + q + Qoff2];
                        }
                    }
                }
            }

            // > J_RS < //

            for (int R2 = 0; R2 < nRshell; R2++) {
                for (int S2 = 0; S2 < nSshell; S2++) {
                    int R = Rstart + R2;
                    int S = Sstart + S2;
                    int Rsize = basisset_->shell(R).nfunction();
                    int Ssize = basisset_->shell(S).nfunction();
                    int Roff = basisset_->shell(R).function_index();
                    int Soff = basisset_->shell(S).function_index();

                    int Roff2 = basisset_->shell(R).start() - basisset_->shell(Rstart).start();
                    int Soff2 = basisset_->shell(S).start() - basisset_->shell(Sstart).start();

                    for (int r = 0; r < Rsize; r++) {
                        for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                            Jp[r + Roff][s + Soff] += 2.0 * J2p[(r + Roff2) * dSsize + s + Soff2];
                        }
                    }
                }
            }

        } // end stripe out

    } // end task

    for (size_t ind = 0; ind < D_.size(); ind++) {
        J_[ind]->hermitivitize();
    }

    timer_off("CFMMBox::compute_self_J()");
}

void CFMMBox::compute_nf_J() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_nf_J()");

    std::vector<std::shared_ptr<TwoBodyAOInt>> ints;

    std::shared_ptr<IntegralFactory> factory = std::make_shared<IntegralFactory>(basisset_);
    std::shared_ptr<TwoBodyAOInt> eri = std::shared_ptr<TwoBodyAOInt>(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    ints.push_back(eri);

    for (int thread = 1; thread < nthread_; thread++) {
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(eri->clone()));
    }

    size_t atom_max_nshell = 0L;
    for (size_t Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        size_t size = 0L;
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        for (int P = Pstart; P < Pstart + nPshell; P++) {
            size += basisset_->shell(P).nfunction();
        }

        atom_max_nshell = std::max(atom_max_nshell, size);
    }

    // Get significant atom pairs (PQ|-style
    std::vector<std::pair<int, int>> atom_pairs;
    for (size_t Ptask = 0; Ptask < atoms_.size(); Ptask++) {
        int Patom = atoms_[Ptask];
        int Pstart = basisset_->shell_on_center(Patom, 0);
        int nPshell = basisset_->nshell_on_center(Patom);

        for (int Qtask = 0; Qtask < atoms_.size(); Qtask++) {
            if (Qtask > Ptask) continue;
            bool found = false;

            int Qatom = atoms_[Qtask];
            int Qstart = basisset_->shell_on_center(Qatom, 0);
            int nQshell = basisset_->nshell_on_center(Qatom);

            for (int P = Pstart; P < Pstart + nPshell; P++) {
                for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                    if (ints[0]->shell_pair_significant(P, Q)) {
                        found = true;
                        atom_pairs.push_back(std::pair<int, int>(Patom, Qatom));
                        break;
                    }
                }
                if (found) break;
            }
        }
    }

    size_t natom_pair = atom_pairs.size();

    int nbf = basisset_->nbf();
    int nshell = basisset_->nshell();

    // => Intermediate Buffers <= //
    std::vector<std::vector<std::shared_ptr<Matrix>>> JT;
    for (int thread = 0; thread < nthread_; thread++) {
        std::vector<std::shared_ptr<Matrix>> J2;
        for (size_t ind = 0; ind < D_.size(); ind++) {
            J2.push_back(std::make_shared<Matrix>("JT", atom_max_nshell, atom_max_nshell));
        }
        JT.push_back(J2);
    }

    // Near field interactions
#pragma omp parallel for
    for (size_t task1 = 0L; task1 < natom_pair; task1++) {

        int Patom = atom_pairs[task1].first;
        int Qatom = atom_pairs[task1].second;

        int Pstart = basisset_->shell_on_center(Patom, 0);
        int Qstart = basisset_->shell_on_center(Qatom, 0);

        int nPshell = basisset_->nshell_on_center(Patom);
        int nQshell = basisset_->nshell_on_center(Qatom);

        int dPsize = (Pstart + nPshell >= nshell ? nbf : basisset_->shell(Pstart + nPshell).start())
                    - basisset_->shell(Pstart).start();
        int dQsize = (Qstart + nQshell >= nshell ? nbf : basisset_->shell(Qstart + nQshell).start())
                    - basisset_->shell(Qstart).start();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        bool touched = false;
        for (int P = Pstart; P < Pstart + nPshell; P++) {
            for (int Q = Qstart; Q < Qstart + nQshell; Q++) {
                if (Q > P) continue;
                if (!ints[0]->shell_pair_significant(P, Q)) continue;

                for (int b = 0; b < near_field_.size(); b++) {
                    CFMMBox* box = near_field_[b];

                    size_t nf_atom_max_nshell = 0L;
                    for (size_t Rtask = 0; Rtask < box->atoms_.size(); Rtask++) {
                        size_t size = 0L;
                        int Ratom = box->atoms_[Rtask];
                        int Rstart = basisset_->shell_on_center(Ratom, 0);
                        int nRshell = basisset_->nshell_on_center(Ratom);

                        for (int R = Rstart; R < Rstart + nRshell; R++) {
                            size += basisset_->shell(R).nfunction();
                        }

                        nf_atom_max_nshell = std::max(atom_max_nshell, size);
                    }

                    // Get significant atom pairs (RS|-style
                    std::vector<std::pair<int, int>> nf_atom_pairs;
                    for (size_t Rtask = 0; Rtask < box->atoms_.size(); Rtask++) {
                        int Ratom = box->atoms_[Rtask];
                        int Rstart = basisset_->shell_on_center(Ratom, 0);
                        int nRshell = basisset_->nshell_on_center(Ratom);

                        for (int Stask = 0; Stask < box->atoms_.size(); Stask++) {
                            if (Stask > Rtask) continue;
                            bool found = false;

                            int Satom = box->atoms_[Stask];
                            int Sstart = basisset_->shell_on_center(Satom, 0);
                            int nSshell = basisset_->nshell_on_center(Satom);

                            for (int R = Rstart; R < Rstart + nRshell; R++) {
                                for (int S = Sstart; S < Sstart + nSshell; S++) {
                                    if (ints[0]->shell_pair_significant(R, S)) {
                                        found = true;
                                        nf_atom_pairs.push_back(std::pair<int, int>(Ratom, Satom));
                                        break;
                                    }
                                }
                                if (found) break;
                            }
                        }
                    }

                    size_t nf_natom_pair = nf_atom_pairs.size();

                    for (size_t task2 = 0L; task2 < nf_natom_pair; task2++) {

                        int Ratom = nf_atom_pairs[task2].first;
                        int Satom = nf_atom_pairs[task2].second;

                        int Rstart = basisset_->shell_on_center(Ratom, 0);
                        int Sstart = basisset_->shell_on_center(Satom, 0);

                        int nRshell = basisset_->nshell_on_center(Ratom);
                        int nSshell = basisset_->nshell_on_center(Satom);

                        int dRsize = (Rstart + nRshell >= nshell ? nbf : basisset_->shell(Rstart + nRshell).start())
                                    - basisset_->shell(Rstart).start();
                        int dSsize = (Sstart + nSshell >= nshell ? nbf : basisset_->shell(Sstart + nSshell).start())
                                    - basisset_->shell(Sstart).start();

                        for (int R = Rstart; R < Rstart + nRshell; R++) {

                            for (int S = Sstart; S < Sstart + nSshell; S++) {
                                if (S > R) continue;
                                if (!ints[0]->shell_pair_significant(R, S)) continue;
                                if (!ints[0]->shell_significant(P, Q, R, S)) continue;

                                if (ints[thread]->compute_shell(P, Q, R, S) == 0) continue;

                                const double *buffer = ints[thread]->buffer();

                                int Psize = basisset_->shell(P).nfunction();
                                int Qsize = basisset_->shell(Q).nfunction();
                                int Rsize = basisset_->shell(R).nfunction();
                                int Ssize = basisset_->shell(S).nfunction();

                                int Poff = basisset_->shell(P).start();
                                int Qoff = basisset_->shell(Q).start();
                                int Roff = basisset_->shell(R).start();
                                int Soff = basisset_->shell(S).start();

                                int Poff2 = basisset_->shell(P).start() - basisset_->shell(Pstart).start();
                                int Qoff2 = basisset_->shell(Q).start() - basisset_->shell(Qstart).start();
                                int Roff2 = basisset_->shell(R).start() - basisset_->shell(Rstart).start();
                                int Soff2 = basisset_->shell(S).start() - basisset_->shell(Sstart).start();

                                for (int ind = 0; ind < D_.size(); ind++) {
                                    double **Dp = D_[ind]->pointer();
                                    double **JTp = JT[thread][ind]->pointer();
                                    const double* buffer2 = buffer;

                                    if (!touched) {
                                        ::memset((void*)JTp[0L * atom_max_nshell], '\0', dPsize * dQsize * sizeof(double));
                                    }

                                    double* J1p = JTp[0L * atom_max_nshell];

                                    double prefactor = 1.0;
                                    if (P == Q) prefactor *= 0.5;
                                    if (R == S) prefactor *= 0.5;
                                    // if (P == R && Q == S) prefactor *= 0.5;

                                    for (int p = 0; p < Psize; p++) {
                                        for (int q = 0; q < Qsize; q++) {
                                            for (int r = 0; r < Rsize; r++) {
                                                for (int s = 0; s < Ssize; s++) {
                                                    J1p[(p + Poff2) * dQsize + q + Qoff2] +=
                                                        prefactor * (Dp[r + Roff][s + Soff] + Dp[s + Soff][r + Roff]) *
                                                        (*buffer2);
                                                    buffer2++;
                                                } // s
                                            } // r
                                        } // q
                                    } // p

                                } // ind
                                touched = true;
                            } // S
                        } // R
                    } // task2
                } // box
            } // Q
        } // P

        if (!touched) continue;

        // => Stripe out <= //
        for (size_t ind = 0; ind < D_.size(); ind++) {
            double** JTp = JT[thread][ind]->pointer();

            double** Jp = J_[ind]->pointer();
            double* J1p = JTp[0L * atom_max_nshell];
            

            // > J_PQ < //
            for (int P2 = 0; P2 < nPshell; P2++) {
                for (int Q2 = 0; Q2 < nQshell; Q2++) {
                    int P = Pstart + P2;
                    int Q = Qstart + Q2;
                    int Psize = basisset_->shell(P).nfunction();
                    int Qsize = basisset_->shell(Q).nfunction();
                    int Poff = basisset_->shell(P).function_index();
                    int Qoff = basisset_->shell(Q).function_index();

                    int Poff2 = basisset_->shell(P).start() - basisset_->shell(Pstart).start();
                    int Qoff2 = basisset_->shell(Q).start() - basisset_->shell(Qstart).start();

                    for (int p = 0; p < Psize; p++) {
                        for (int q = 0; q < Qsize; q++) {
#pragma omp atomic
                            Jp[p + Poff][q + Qoff] += 2.0 * J1p[(p + Poff2) * dQsize + q + Qoff2];
                        }
                    }
                }
            }

        } // end stripe out

    } // task1

    for (size_t ind = 0; ind < D_.size(); ind++) {
        J_[ind]->hermitivitize();
    }

    timer_off("CFMMBox::compute_nf_J()");
}

void CFMMBox::compute_ff_J() {

    // std::raise(SIGINT);

    timer_on("CFMMBox::compute_ff_J()");

    for (int l = 0; l <= lmax_; l++) {
        for (int m = -l; m <= l; m++) {
            int mu = m_addr(m);
            ff_energy_ += 0.5 * mpoles_->get_multipoles()[l][mu] * Vff_->get_multipoles()[l][mu];          
        }
    }

    /*
    int nbf = basisset_->nbf();

    // Far field interactions
    for (int ind = 0; ind < D_.size(); ind++) {
        double **Jp = J_[ind]->pointer();
        double **Dp = D_[ind]->pointer();
        for (const auto &self_pair : mpoles_) {
            int pq_index = self_pair.first;
            std::vector<std::vector<double>>& pq_mpole = self_pair.second->get_multipoles();
            int p = pq_index / nbf;
            int q = pq_index % nbf;

            double cont = 0.0;

            for (int l = 0; l <= lmax_; l++) {
                for (int m = -l; m <= l; m++) {
                    int mu = m_addr(m);
                    Jp[p][q] += pq_mpole[l][mu] * Vff_->get_multipoles()[l][mu];
                    cont += pq_mpole[l][mu] * Vff_->get_multipoles()[l][mu];
                }
            }

            outfile->Printf("  BASIS PAIR FAR-FIELD CONTRIBUTION: p=%d, q=%d %.16f\n", p, q, cont);
            
        }
    }
    */

    timer_off("CFMMBox::compute_ff_J()");
}

void CFMMBox::compute_J() {

    compute_self_J();
    compute_nf_J();
    compute_ff_J();

}

CFMMBox::~CFMMBox() {

    for (int c = 0; c < 8; c++) {
        if (children_[c]) delete children_[c];
    }

}

CFMMTree::CFMMTree(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basisset, 
                    std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, int nlevels, int lmax) {
    molecule_ = molecule;
    basisset_ = basisset;
    nlevels_ = nlevels;
    lmax_ = lmax;
    D_ = D;
    J_ = J;
    ff_energy_ = 0.0;
    root_ = new CFMMBox(molecule_, basisset_, D_, J_, lmax_);
}

void CFMMTree::make_children(CFMMBox* box) {
    if (box->get_level() == nlevels_ - 1) return;

    box->make_children();

    std::vector<CFMMBox*> children = box->get_children();
    
    for (CFMMBox* child : children) {
        make_children(child);
    }
}

void CFMMTree::calculate_multipoles(CFMMBox* box) {
    if (!box) return;
    if (box->natom() == 0) return;

    std::vector<CFMMBox*> children = box->get_children();

    for (CFMMBox* child : children) {
        calculate_multipoles(child);
    }

    if (box->get_level() == nlevels_ - 1) {
        box->compute_mpoles();
    } else {
        box->compute_mpoles_from_children();
    }

}

void CFMMTree::set_nf_lff(CFMMBox* box) {
    if (!box) return;
    if (box->natom() == 0) return;

    box->set_nf_lff();

    std::vector<CFMMBox*> children = box->get_children();

    for (CFMMBox* child : children) {
        set_nf_lff(child);
    }
}

void CFMMTree::compute_far_field(CFMMBox* box) {
    if (!box) return;
    if (box->natom() == 0) return;

    box->compute_far_field_vector();

    std::vector<CFMMBox*> children = box->get_children();

    for (CFMMBox* child : children) {
        compute_far_field(child);
    }
}

void CFMMTree::calculate_J(CFMMBox* box) {
    if (!box) return;
    if (box->natom() == 0) return;

    if (box->get_level() == nlevels_ - 1) {
        box->compute_J();
        ff_energy_ += box->ff_energy();
        return;
    }

    std::vector<CFMMBox*> children = box->get_children();

    for (CFMMBox* child : children) {
        calculate_J(child);
    }

}

void CFMMTree::build_J() {

    // Zero the J matrix
    for (int ind = 0; ind < D_.size(); ind++) {
        J_[ind]->zero();
    }

    make_children(root_);
    calculate_multipoles(root_);
    set_nf_lff(root_);
    compute_far_field(root_);
    calculate_J(root_);

    // Hermitivitize J matrix afterwards
    for (int ind = 0; ind < D_.size(); ind++) {
        // J_[ind]->scale(2.0);
        J_[ind]->hermitivitize();
    }

    outfile->Printf("  FAR FIELD ENERGY: %.16f\n", ff_energy_);

    for (int l = 0; l <= lmax_; l++) {
        for (int m = -l; m <= l; m++) {
            int mu = m_addr(m);
            double pole = root_->get_mpole_val(l, mu);
            // outfile->Printf("MPOLE  L: %d, M: %d, Ylm: %8.8f\n", l, m, pole);
        }
    }

}

CFMMTree::~CFMMTree() {
    delete root_;
}

} // end namespace psi