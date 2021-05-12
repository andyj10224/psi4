#ifndef libfmm_mpoles_helper_H
#define libfmm_mpoles_helper_H

#include "psi4/pragma.h"
#include <functional>
#include <memory>
#include <tuple>
#include <vector>

namespace psi {
    
class RealSolidHarmonics {

    protected:
      // Ylm[l][m] = sum (coeff * x^a * y^b * z^c), stores a tuple of (coeff, a, b, c)
      std::vector<std::vector<std::vector<std::tuple<double, int, int, int>>>> mpole_terms_;
      // Real Solid Harmonics, normalized according to Stone's convention
      std::vector<std::vector<double>> Ylm_;
      // Maximum angular momentum
      const int lmax_;
      // Compute regular harmonics terms (mpole_terms)
      void compute_regular_terms();
      // Compute irregular harmonics terms (mpole_terms)
      void compute_irregular_terms();
      // Compute multipoles if it were regular
      
      // Compute multipoles if it were irregular

    public:
      // Constructor
      RealSolidHarmonics(int lmax);

      // Adds two harmonics together
      void add(const RealSolidHarmonics& rsh);
      // Compute multipoles from terms
      virtual void compute_mpoles();
      // Translate the solid harmonics
      virtual void translate();


}; // End RealSolidHarmonics class

class RealRegularHarmonics : public RealSolidHarmonics {

}

} // namespace psi

#endif