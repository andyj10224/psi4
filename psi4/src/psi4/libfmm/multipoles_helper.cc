#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>

namespace psi {

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

    Vector3 x_axis = ca / ca.norm();
    Vector3 y_axis = z_axis.cross(x_axis);

    Uz_ = std::make_shared<Matrix>("Uz Matrix", 3, 3);

    for (int i = 0; i < 3; i++) {
        Uz->set(0, i, x_axis[i]);
        Uz->set(1, i, y_axis[i]);
        Uz->set(2, i, z_axis[i]);
    }

    for (int l = 0; l <= lmax_; l++) {
        D_cache_.append(nullptr);
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

    if (D_cache_[l] != nullptr) {
        return D_cache_[l];
    }

    SharedMatrix Drot;

    if (l == 0) {
        Drot = std::make_shared<Matrix>("D Rotation Matrix", 1, 1);
        Drot->set(0, 0, 1.0);
    } else if (l == 1) {
        Drot = std::make_shared<Matrix>("D Rotation Matrix", 3, 3);
        std::vector<int> permute(2, 0, 1);
        for (int i = 0; i < 3; i++) {
            int ip = permute[i];
            for (int j = 0; j < 3; j++) {
                int jp = permute[j];
                Drot->set(i, j, Uz_[ip][jp]);
            }
        }
    } else {
        Drot = std::make_shared<Matrix>("D Rotation Matrix", 2*l+1, 2*l+1);
        for (int m1 = -l; m1 <= l; m1++) {
            int k1 = m_addr(m1);
            for (int m2 = -l; m2 <= l; m2++) {
                int k2 = m_addr(m2);
                double Uterm = u(l, m1, m2);
                if (Uterm != 0) Uterm *= U(l, m1, m2);
                double Vterm = v(l, m1, m2);
                if (Vterm != 0) Vterm *= V(l, m1, m2);
                double Wterm = w(l, m1, m2);
                if (Wterm != 0) Wterm *= W(l, m1, m2);
                Drot->set(k1, k2, Uterm + Vterm + Wterm);
            }
        }
    }

    D_cache_[l] = Drot;
    return D_cache_[l];
}

void RealSolidHarmonics::compute_terms_regular(double q, double x, double y, double z) {
    
    double r2 = x*x + y*y + z*z;

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

                Rc_[l][l-1].push_back(coef, a, b, c+1);
            }

            // Rs[l][l-1]
            for (int ind = 0; ind < Rs_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rs_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rs_[l][l-1].push_back(coef, a, b, c+1);
            }

            // => m = l <= //

            // Rc[l-1][l-1] contribution to Rc[l][l]
            for (int ind = 0; ind < Rc_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rc_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rc_[l][l].push_back(-coef/(2*l), a+1, b, c);
            }

            // Rs[l-1][l-1] contribution to Rc[l][l]
            for (int ind = 0; ind < Rs_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rs_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rc_[l][l].push_back(coef/(2*l), a, b+1, c);
            }

            // Rc[l-1][l-1] contribution to Rs[l][l]
            for (int ind = 0; ind < Rc_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rc_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rs_[l][l].push_back(-coef/(2*l), a, b+1, c);
            }

            // Rs[l-1][l-1] contribution to Rs[l][l]
            for (int ind = 0; ind < Rs_[l-1][l-1].size(); ind++) {
                auto term_tuple = Rs_[l-1][l-1][ind];
                double coef = std::get<0>(term_tuple);
                int a = std::get<1>(term_tuple);
                int b = std::get<2>(term_tuple);
                int c = std::get<3>(term_tuple);

                Rs_[l][l].push_back(-coef/(2*l), a+1, b, c);
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

                    mpole_terms_[l][mu].push_back(prefactor * coef, a, b, c);
                }
            } else {
                prefactor = std::pow(-1.0, (double) m) * std::sqrt(2.0 * std::tgamma(l-m+1) * std::tgamma(l+m+1));

                for (int ind = 0; ind < Rs_[l][-m].size(); ind++) {
                    auto term_tuple = Rs_[l][-m][ind];
                    double coef = std::get<0>(term_tuple);
                    int a = std::get<1>(term_tuple);
                    int b = std::get<2>(term_tuple);
                    int c = std::get<3>(term_tuple);

                    mpole_terms_[l][mu].push_back(prefactor * coef, a, b, c);
                }
            }
        }
    }


}

} // namespace psi