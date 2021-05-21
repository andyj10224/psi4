#ifndef libfmm_mpoles_helper_H
#define libfmm_mpoles_helper_H

#include "psi4/pragma.h"

#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

namespace psi {

enum SolidHarmonicsType {Regular, Irregular};

class MultipoleRotationFactory {

    protected:
      Vector3 R_a_;
      Vector3 R_b_;

      // New Z axis in rotated frame of reference
      SharedMatrix Uz_;

      // Maximal Angular Momentum
      int lmax_;

      // Cached Rotation Matrices in a vector of Matrices
      std::vector<SharedMatrix> D_cache_;

      double U(int l, int m, int M);
      double V(int l, int m, int M);
      double W(int l, int m, int M);
      double P(int i, int l, int mu, int M);

    public:
      // Constructor
      MultipoleRotationFactory(Vector3 R_a, Vector3 R_b, int lmax);

      SharedMatrix get_D(int l);

}; // End MultipoleRotationFactory

inline double MultipoleRotationFactory::u(int l, int m, int M) {
    if (std::abs(M) < l) {
        return std::sqrt((l+m)*(l-m) /((l+M)*(l-M)));
    } else {
        return std::sqrt((l+m)/(l-m)/(2*l*(2*l-1)));
    }
}

inline double MultipoleRotationFactory::v(int l, int m, int M) {
    double dm0 = 0.0;
    if (m == 0) dm0 = 1.0;
    if (std::abs(M) < l) {
        return 0.5 * (1.0 - 2.0*dm0) * std::sqrt((1.0+dm0)*(l+std::abs(m)-1)*(l+std::abs(m)) / ((l+M)*(l-M)));
    } else {
        return 0.5 * (1.0 - 2.0*dm0) * std::sqrt((1.0+dm0)*(l+std::abs(m)-1)*(l+std::abs(m)) / ((2*l)*(2*l-1)));
    }
}

inline double MultipoleRotationFactory::w(int l, int m, int M) {
    double dm0 = 0.0;
    if (m == 0) dm0 = 1.0;
    if (std::abs(M) < l) {
        return 0.5 * (dm0 - 1) * std::sqrt((l-std:;abs(m)-1)*(l-std::abs(m)) / ((l+M)*(l-M)));
    } else {
        return 0.5 * (dm0 - 1) * std::sqrt((l-std:;abs(m)-1)*(l-std::abs(m)) / ((2*l)*(2*l-1)));
    }
}
    
class RealSolidHarmonics {

    protected:
      // Ylm[l][m] = sum (coeff * x^a * y^b * z^c), stores a tuple of (coeff, a, b, c)
      std::vector<std::vector<std::vector<std::tuple<double, int, int, int>>>> mpole_terms_;
      // Helgaker Rs terms (used in generating mpole_terms_)
      std::vector<std::vector<std::vector<std::tuple<double, int, int, int>>>> Rc_;
      // Helgaker Rc terms (used in generating mpole_terms_)
      std::vector<std::vector<std::vector<std::tuple<double, int, int, int>>>> Rs_;
      // Real Solid Harmonics, normalized according to Stone's convention
      std::vector<std::vector<double>> Ylm_;
      // Maximum angular momentum
      const int lmax_;
      // Regular or Irregular?
      SolidHarmonicsType type_;
      // Center of the Harmonics
      Vector3 center_;

      // Compute terms if it were regular
      void compute_terms_regular();
      // Compute terms if it were irregular
      void compute_terms_irregular();

      // Return a translated copy of the multipoles if it were regular
      std::shared_ptr<RealSolidHarmonics> translate_regular(Vector3 new_center);
      // Return a translated copy of the multipoles if it were irregular
      std::shared_ptr<RealSolidHarmonics> translate_irregular(Vector3 new_center);

    public:
      // Constructor
      RealSolidHarmonics(int lmax, Vector3 center, SolidHarmonicsType type);

      // Adds two harmonics together
      void add(const RealSolidHarmonics& rsh);
      void add(std::shared_ptr<RealSolidHarmonics> rsh);
      
      // Returns a reference of Ylm
      std::vector<std::vector<double>>& get_multipoles() { return Ylm_; }

      // Compute the terms (regular)
      void compute_terms();
      // Translate the solid harmonics
      std::shared_ptr<RealSolidHarmonics> translate(Vector3 new_center);

}; // End RealSolidHarmonics class

} // namespace psi

#endif