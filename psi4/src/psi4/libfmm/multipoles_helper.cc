#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_tree.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>

namespace psi {

int choose(int n, int r) {
    if (r < 0 || r > n) {
        return 0;
    }
    int small = std::min(n, n-r);
    int nCr = 1;
    for (int t = 0; t < small; t++) {
        nCr *= n;
        nCr /= (t+1);
        n -= 1;
    }
    return nCr;
}

int m_addr(int m) {
    /*- Return the unsigned (array) address of m -*/
    if (m <= 0) {
        // 0, 1s, 2s, 3s, ...
        return 2*(-m);
    } else {
        // 1c, 2c, 3c, ...
        return 2*m-1;
    }
}

MultipoleRotationFactory::MultipoleRotationFactory(Vector3 R_a, Vector3 R_b, int lmax) {
    lmax_ = lmax;
    Vector3 R_ab = R_a - R_b;
    double R = R_ab.norm();

    Vector3 z_axis = R_ab / R;
    Vector3 ca(z_axis);

    if (R_a[1] == R_b[1] && R_a[2] == R_a[2]) {
        ca[1] += 1.0;
    } else {
        ca[0] += 1.0;
    }

    double this_dot = ca.dot(z_axis);
    ca -= (z_axis * this_dot);

    Vector3 x_axis = ca / ca.norm();
    Vector3 y_axis = z_axis.cross(x_axis);

    Uz_ = std::make_shared<Matrix>("Uz Matrix", 3, 3);

    for (int i = 0; i < 3; i++) {
        Uz_->set(0, i, x_axis[i]);
        Uz_->set(1, i, y_axis[i]);
        Uz_->set(2, i, z_axis[i]);
    }

    for (int l = 0; l <= lmax_; l++) {
        D_cache_.push_back(nullptr);
    }

}

double MultipoleRotationFactory::U(int l, int m, int M) {
    return P(0, l, m, M);
}

double MultipoleRotationFactory::V(int l, int m, int M) {
    double val;
    if (m == 0) {
        val = P(1, l, 1, M) + P(-1, l, -1, M);
    } else if (m > 0) {
        if (m == 1) {
            val = std::sqrt(2.0) * P(1, l, m-1, M);
        } else {
            val = P(1, l, m-1, M) - P(-1, 1, -m+1, M);
        }
    } else {
        if (m == -1) {
            val = std::sqrt(2.0) * P(-1, l, -m-1, M);
        } else {
            val = P(1, l, m+1, M) + P(-1, l, -m-1, M);
        }
    }
    return val;
}

double MultipoleRotationFactory::W(int l, int m, int M) {
    double val;
    if (m == 0) {
        val = 0.0;
    } else if (m > 0) {
        val = P(1, l, m+1, M) + P(-1, l, -m-1, M);
    } else {
        val = P(1, l, m-1, M) - P(-1, l, -m+1, M);
    }
    return val;
}

double MultipoleRotationFactory::P(int i, int l, int mu, int M) {
    int I = m_addr(i);
    SharedMatrix D1 = get_D(1);
    SharedMatrix Dl1 = get_D(l-1);
    if (std::abs(M) < l) {
        return D1->get(I, 0) * Dl1->get(m_addr(mu), m_addr(M));
    } else if (M == l) {
        return D1->get(I, 1) * Dl1->get(m_addr(mu), m_addr(M-1)) - D1->get(I, 2) * Dl1->get(m_addr(mu), m_addr(-M+1));
    } else {
        return D1->get(I, 1) * Dl1->get(m_addr(mu), m_addr(M+1)) + D1->get(I, 2) * Dl1->get(m_addr(mu), m_addr(-M-1));
    }
}

SharedMatrix MultipoleRotationFactory::get_D(int l) {
    if (l > lmax_) {
        throw PsiException("Input l is larger than the set lmax for the Rotation Matrix", __FILE__, __LINE__);
    }

    if (D_cache_[l]) {
        return D_cache_[l];
    }

    SharedMatrix Drot;

    if (l == 0) {
        Drot = std::make_shared<Matrix>("D Rotation Matrix", 1, 1);
        Drot->set(0, 0, 1.0);
    } else if (l == 1) {
        Drot = std::make_shared<Matrix>("D Rotation Matrix", 3, 3);
        std::vector<int> permute {2, 0, 1};
        for (int i = 0; i < 3; i++) {
            int ip = permute[i];
            for (int j = 0; j < 3; j++) {
                int jp = permute[j];
                Drot->set(i, j, Uz_->get(ip, jp));
            }
        }
    } else {
        Drot = std::make_shared<Matrix>("D Rotation Matrix", 2*l+1, 2*l+1);
        for (int m1 = -l; m1 <= l; m1++) {
            int k1 = m_addr(m1);
            for (int m2 = -l; m2 <= l; m2++) {
                int k2 = m_addr(m2);
                double Uterm = u(l, m1, m2);
                if (Uterm != 0.0) Uterm *= U(l, m1, m2);
                double Vterm = v(l, m1, m2);
                if (Vterm != 0.0) Vterm *= V(l, m1, m2);
                double Wterm = w(l, m1, m2);
                if (Wterm != 0.0) Wterm *= W(l, m1, m2);
                Drot->set(k1, k2, Uterm + Vterm + Wterm);
            }
        }
    }

    D_cache_[l] = Drot;
    return D_cache_[l];
}

HarmonicCoefficients::HarmonicCoefficients(int lmax, SolidHarmonicsType type) {
    lmax_ = lmax;
    type_ = type;
    if (type_ == Regular) compute_terms_regular();
    if (type_ == Irregular) compute_terms_irregular();
}

void HarmonicCoefficients::compute_terms_irregular() {
    throw FeatureNotImplemented("libfmm", "RealSolidHarmonics::compute_terms_irregular()", __FILE__, __LINE__);
}

void HarmonicCoefficients::compute_terms_regular() {

    Rc_.resize(lmax_+1);
    Rs_.resize(lmax_+1);
    mpole_terms_.resize(lmax_+1);

    for (int l = 0; l <= lmax_; l++) {
        Rc_[l].resize(l+1);
        Rs_[l].resize(l+1);
        mpole_terms_[l].resize(2*l+1);

        if (l == 0) {
            Rc_[0][0].push_back(std::make_tuple(1.0, 0, 0, 0));
            Rs_[0][0].push_back(std::make_tuple(0.0, 0, 0, 0));
        }

        else {
            // m < l-1 terms
            for (int m = 0; m < l-1; m++) {
                int denom = (l+m)*(l-m);

                // Rc_[l-1][m] contributions to Rc_[l][m]
                for (int ind = 0; ind < Rc_[l-1][m].size(); ind++) {
                    auto term_tuple = Rc_[l-1][m][ind];
                    double coef = std::get<0>(term_tuple);
                    int a = std::get<1>(term_tuple);
                    int b = std::get<2>(term_tuple);
                    int c = std::get<3>(term_tuple);

                    coef *= (2*l-1) / denom;
                    Rc_[l][m].push_back(std::make_tuple(coef, a, b, c+1));
                }

                // Rc_[l-2][m] contributions to Rc_[l][m]
                for (int ind = 0; ind < Rc_[l-2][m].size(); ind++) {
                    auto term_tuple = Rc_[l-2][m][ind];
                    double coef = std::get<0>(term_tuple);
                    int a = std::get<1>(term_tuple);
                    int b = std::get<2>(term_tuple);
                    int c = std::get<3>(term_tuple);

                    coef /= denom;
                    Rc_[l][m].push_back(std::make_tuple(-coef, a+2, b, c));
                    Rc_[l][m].push_back(std::make_tuple(-coef, a, b+2, c));
                    Rc_[l][m].push_back(std::make_tuple(-coef, a, b, c+2));
                }

                // Rs_[l-1][m] contributions to Rs_[l][m]
                for (int ind = 0; ind < Rs_[l-1][m].size(); ind++) {
                    auto term_tuple = Rs_[l-1][m][ind];
                    double coef = std::get<0>(term_tuple);
                    int a = std::get<1>(term_tuple);
                    int b = std::get<2>(term_tuple);
                    int c = std::get<3>(term_tuple);

                    coef *= (2*l-1) / denom;
                    Rs_[l][m].push_back(std::make_tuple(coef, a, b, c+1));
                }

                // Rs_[l-2][m] contributions to Rs_[l][m]
                for (int ind = 0; ind < Rs_[l-2][m].size(); ind++) {
                    auto term_tuple = Rs_[l-2][m][ind];
                    double coef = std::get<0>(term_tuple);
                    int a = std::get<1>(term_tuple);
                    int b = std::get<2>(term_tuple);
                    int c = std::get<3>(term_tuple);

                    coef /= denom;
                    Rs_[l][m].push_back(std::make_tuple(-coef, a+2, b, c));
                    Rs_[l][m].push_back(std::make_tuple(-coef, a, b+2, c));
                    Rs_[l][m].push_back(std::make_tuple(-coef, a, b, c+2));
                }
            }

            // => m = l-1 <= //

            // Rc[l][l-1]
            for (int ind = 0; ind < Rc_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rc_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rc_[l][l-1].push_back(std::make_tuple(coef, a, b, c+1));
            }

            // Rs[l][l-1]
            for (int ind = 0; ind < Rs_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rs_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rs_[l][l-1].push_back(std::make_tuple(coef, a, b, c+1));
            }

            // => m = l <= //

            // Rc[l-1][l-1] contribution to Rc[l][l]
            for (int ind = 0; ind < Rc_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rc_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rc_[l][l].push_back(std::make_tuple(-coef/(2*l), a+1, b, c));
            }

            // Rs[l-1][l-1] contribution to Rc[l][l]
            for (int ind = 0; ind < Rs_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rs_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rc_[l][l].push_back(std::make_tuple(coef/(2*l), a, b+1, c));
            }

            // Rc[l-1][l-1] contribution to Rs[l][l]
            for (int ind = 0; ind < Rc_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rc_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rs_[l][l].push_back(std::make_tuple(-coef/(2*l), a, b+1, c));
            }

            // Rs[l-1][l-1] contribution to Rs[l][l]
            for (int ind = 0; ind < Rs_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rs_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rs_[l][l].push_back(std::make_tuple(-coef/(2*l), a+1, b, c));
            }
        }

        for (int m = -l; m <= l; m++) {
            // m is signed address
            // mu is unsigned address
            int mu = m_addr(m);
            double prefactor = 1.0;
            if ((mu == 0) || (mu % 2 == 1)) {
                if (mu == 0) {
                    prefactor = std::tgamma(l+1);
                } else {
                    prefactor = std::pow(-1.0, (double) m) * std::sqrt(2.0 * std::tgamma(l-m+1) * std::tgamma(l+m+1));
                }
                for (int ind = 0; ind < Rc_[l][m].size(); ind++) {
                    auto term_tuple = Rc_[l][m][ind];
                    double coef = std::get<0>(term_tuple);
                    int a = std::get<1>(term_tuple);
                    int b = std::get<2>(term_tuple);
                    int c = std::get<3>(term_tuple);

                    mpole_terms_[l][mu].push_back(std::make_tuple(prefactor * coef, a, b, c));
                }
            } else {
                prefactor = std::pow(-1.0, (double) m) * std::sqrt(2.0 * std::tgamma(l-m+1) * std::tgamma(l+m+1));

                for (int ind = 0; ind < Rs_[l][-m].size(); ind++) {
                    auto term_tuple = Rs_[l][-m][ind];
                    double coef = std::get<0>(term_tuple);
                    int a = std::get<1>(term_tuple);
                    int b = std::get<2>(term_tuple);
                    int c = std::get<3>(term_tuple);

                    mpole_terms_[l][mu].push_back(std::make_tuple(prefactor * coef, a, b, c));
                }
            }
        }
    }
}

RealSolidHarmonics::RealSolidHarmonics(int lmax, Vector3 center, SolidHarmonicsType type) {
    lmax_ = lmax;
    center_ = center;
    type_ = type;

    // Set all multipoles to zero
    Ylm_.resize(lmax_+1);

    for (int l = 0; l <= lmax_; l++) {
        Ylm_[l].resize(2*l+1, 0.0);
    }
    
}

void RealSolidHarmonics::add(const RealSolidHarmonics& rsh) {
    for (int l = 0; l <= lmax_; l++) {
        for (int mu = 0; mu < 2*l+1; mu++) {
            Ylm_[l][mu] += rsh.Ylm_[l][mu];
        }
    }
}

void RealSolidHarmonics::add(const std::shared_ptr<RealSolidHarmonics>& rsh) {
    for (int l = 0; l <= lmax_; l++) {
        for (int mu = 0; mu < 2*l+1; mu++) {
            Ylm_[l][mu] += rsh->Ylm_[l][mu];
        }
    }
}

std::shared_ptr<RealSolidHarmonics> RealSolidHarmonics::translate(Vector3 new_center) {
    if (type_ == Regular) return translate_regular(new_center);
    if (type_ == Irregular) return translate_irregular(new_center);
}

std::shared_ptr<RealSolidHarmonics> RealSolidHarmonics::translate_irregular(Vector3 new_center) {
    auto trans_harmonics = std::make_shared<RealSolidHarmonics>(lmax_, new_center, Irregular);
    auto rotation_factory = std::make_shared<MultipoleRotationFactory>(center_, new_center, lmax_);

    Vector3 R_ab = new_center - center_;
    double R = R_ab.norm();

    std::vector<std::vector<double>> rot_mpoles(lmax_+1);
    std::vector<std::vector<double>> trans_rot_mpoles(lmax_+1);

    // Rotate Multipoles to direction of translation
    for (int l = 0; l <= lmax_; l++) {
        rot_mpoles[l].resize(2*l+1, 0.0);
        int dim = 2*l+1;
        SharedMatrix Dmat = rotation_factory->get_D(l);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                rot_mpoles[l][i] += Dmat->get(i, j) * Ylm_[l][j];
            }
        }
    }

    // Translate Rotated Multipoles
    for (int l = 0; l <= lmax_; l++) {
        trans_rot_mpoles[l].resize(2*l+1, 0.0);
        for (int j = l; j <= lmax_; j++) {
            for (int m = -l; m <= l; m++) {
                int mu = m_addr(m);
                double coef = std::sqrt((double) choose(j+m,l+m)*choose(j-m,l-m));

                trans_rot_mpoles[l][mu] += coef * std::pow(-R, j-l) * rot_mpoles[j][mu];
            }
        }
    }

    // Backrotation of Multipoles
    for (int l = 0; l <= lmax_; l++) {
        SharedMatrix Dmat = rotation_factory->get_D(l);
        int dim = 2*l+1;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                trans_harmonics->Ylm_[l][i] += Dmat->get(j, i) * trans_rot_mpoles[l][j];
            }
        }
    }

    return trans_harmonics;
}

std::shared_ptr<RealSolidHarmonics> RealSolidHarmonics::translate_regular(Vector3 new_center) {
    auto trans_harmonics = std::make_shared<RealSolidHarmonics>(lmax_, new_center, Regular);
    auto rotation_factory = std::make_shared<MultipoleRotationFactory>(center_, new_center, lmax_);

    Vector3 R_ab = new_center - center_;
    double R = R_ab.norm();

    std::vector<std::vector<double>> rot_mpoles(lmax_+1);
    std::vector<std::vector<double>> trans_rot_mpoles(lmax_+1);

    // Rotate Multipoles to direction of translation
    for (int l = 0; l <= lmax_; l++) {
        rot_mpoles[l].resize(2*l+1, 0.0);
        int dim = 2*l+1;
        SharedMatrix Dmat = rotation_factory->get_D(l);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                rot_mpoles[l][i] += Dmat->get(i, j) * Ylm_[l][j];
            }
        }
    }

    // Translate Rotated Multipoles
    for (int l = 0; l <= lmax_; l++) {
        trans_rot_mpoles[l].resize(2*l+1, 0.0);
        for (int j = 0; j <= l; j++) {
            for (int m = -j; m <= j; m++) {
                int mu = m_addr(m);
                double coef = std::sqrt((double) choose(l+m,j+m)*choose(l-m,j-m));

                trans_rot_mpoles[l][mu] += coef * std::pow(-R, l-j) * rot_mpoles[j][mu];
            }
        }
    }

    // Backrotation of Multipoles
    for (int l = 0; l <= lmax_; l++) {
        SharedMatrix Dmat = rotation_factory->get_D(l);
        int dim = 2*l+1;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                trans_harmonics->Ylm_[l][i] += Dmat->get(j, i) * trans_rot_mpoles[l][j];
            }
        }
    }

    return trans_harmonics;
}

// A helper method to compute the interaction tensor between aligned multipoles after rotation
SharedVector RealSolidHarmonics::build_T_spherical(int la, int lb, double R) {
    int lmin = std::min(la, lb);
    SharedVector Tvec = std::make_shared<Vector>(2*lmin+1);
    double denom = std::pow(R, la+lb+1);

    for (int m = -lmin; m <= lmin; lmin++) {
        int mu = m_addr(m);
        double Tval = std::pow(-1.0, (double) (lb-m)) * std::sqrt((double) choose(la+lb, la+m) * choose(la+lb, la-m)) / denom;
        Tvec->set(mu, Tval);
    }

    return Tvec;
}

std::shared_ptr<RealSolidHarmonics> RealSolidHarmonics::far_field_vector(Vector3 far_center) {
    
    Vector3 R_ab = far_center - center_;
    double R = R_ab.norm();

    auto Vff = std::make_shared<RealSolidHarmonics>(lmax_, far_center, Irregular);
    auto rotation_factory = std::make_shared<MultipoleRotationFactory>(far_center, center_, lmax_);

    for (int l = 0; l <= lmax_; l++) {
        for (int j = 0; j <= lmax_; j++) {
            SharedVector Tvec = build_T_spherical(l, j, R);
            int nterms = 2*std::min(l,j)+1;

            std::vector<double> rotated_mpole(2*j+1, 0.0);
            SharedMatrix Dmat = rotation_factory->get_D(j);
            for (int u = 0; u < 2*j+1; u++) {
                for (int v = 0; v < 2*j+1; v++) {
                    rotated_mpole[u] = Dmat->get(u, v) * Ylm_[j][v];
                }
            }

            std::vector<double> temp(nterms, 0.0);
            for (int u = 0; u < nterms; u++) {
                temp[u] = Tvec->get(u) * rotated_mpole[u];
            }

            SharedMatrix Dl = rotation_factory->get_D(l);
            for (int r = 0; r < 2*l+1; r++) {
                for (int s = 0; s < nterms; s++) {
                    Vff->Ylm_[l][r] += Dl->get(s, r) * temp[s];
                }
            }

        }
    }

    return Vff;
}

} // namespace psi