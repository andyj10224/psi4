#ifndef libfmm_mpoles_helper_H
#define libfmm_mpoles_helper_H

#include "psi4/pragma.h"
#include <functional>
#include <memory>
#include <vector>

namespace psi {
    
class RealSolidHarmonics {

    protected:
      // Real Solid Harmonics, normalized according to Stone's convention
      std::vector<std::vector<double>> Ylm_;
      // Maximum angular momentum
      const int lmax_;

    public:
      // Constructor
      RealSolidHarmonics(int lmax);

      // Adds two harmonics together
      void add(const RealSolidHarmonics& rsh);
      // Compute regular harmonics for an orbital
      def compute_regular();



}; // End RealSolidHarmonics class

} // namespace psi

#endif