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
      Vector3 Uz_;

      // Cached Rotation Matrices in a vector of Matrices
      std::vector<SharedMatrix> D_cache_;

      SharedMatrix U(int l, int m, int M);
      SharedMatrix V(int l, int m, int M);
      SharedMatrix W(int l, int m, int M);
      SharedMatrix P(int i, int l, int mu, int M);

      double u(int l, int m, int M);
      double v(int l, int m, int M);
      double w(int l, int m, int M);

    public:
      // Constructor
      MultipoleRotationFactory(Vector3 R_a, Vector3 R_b);

      SharedMatrix get_D();

}; // End MultipoleRotationFactory
    
class RealSolidHarmonics {

    protected:
      // Ylm[l][m] = sum (coeff * x^a * y^b * z^c), stores a tuple of (coeff, a, b, c)
      std::vector<std::vector<std::vector<std::tuple<double, int, int, int>>>> mpole_terms_;
      // Real Solid Harmonics, normalized according to Stone's convention
      std::vector<std::vector<double>> Ylm_;
      // Maximum angular momentum
      const int lmax_;
      // Regular or Irregular?
      SolidHarmonicsType type_;

      // Compute terms if it were regular
      void compute_terms_regular();
      // Compute terms if it were irregular
      void compute_terms_irregular();

      // Compute multipoles if it were regular
      void compute_mpoles_regular();
      // Compute multipoles if it were irregular
      void compute_mpoles_regular();

      // Translate multipoles if it were regular
      void translate_regular();
      // Translate multipoles if it were irregular
      void translate_irregular();

    public:
      // Constructor
      RealSolidHarmonics(int lmax, SolidHarmonicsType type);

      // Adds two harmonics together
      void add(const RealSolidHarmonics& rsh);

      // Compute the terms (regular)
      void compute_terms();
      // Compute multipoles from terms
      void compute_mpoles();
      // Translate the solid harmonics
      void translate();

}; // End RealSolidHarmonics class

} // namespace psi

#endif