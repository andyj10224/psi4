/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "dlpno.h"
#include "sparse.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <numeric>
#include <sstream>

namespace psi {
namespace dlpno {

namespace {

// Diagonal expectations, without allocating an nMO x nMO matrix or changing
// the orbital gauge. Local orbitals need not diagonalize their Fock matrix.
SharedVector dlpno_orbital_energies(const SharedMatrix& C, const SharedMatrix& F) {
    auto FC = linalg::doublet(F, C);
    auto energies = std::make_shared<Vector>(C->ncol());
    for (int a = 0; a < C->ncol(); ++a) {
        double energy = 0.0;
        for (int mu = 0; mu < C->nrow(); ++mu) energy += C->get(mu, a) * FC->get(mu, a);
        energies->set(a, energy);
    }
    return energies;
}

std::string dlpno_orbital_tuple_name(const std::string& prefix, std::initializer_list<int> indices) {
    std::string name = prefix;
    for (int i : indices) name += "_" + std::to_string(i + 1);
    return name;
}

}  // namespace

// Stream one domain at a time. Retaining AO coefficients for all tuples at once
// would undo much of the memory advantage of a local-correlation calculation.
class DLPNOOrbitalWriter {
   private:
    std::filesystem::path root_;
    std::string header_;
    std::vector<int> source_row_;
    std::vector<double> ao_scale_;
    std::ofstream index_;
    size_t file_count_ = 0;

    static bool make_directory(const std::filesystem::path& path) {
        std::error_code error;
        const bool created = std::filesystem::create_directory(path, error);
        if (error || !std::filesystem::is_directory(path))
            throw PSIEXCEPTION("DLPNO orbital export cannot create directory " + path.string() +
                               (error ? ": " + error.message() : ""));
        return created;
    }

   public:
    DLPNOOrbitalWriter(const std::shared_ptr<BasisSet>& basis, const std::shared_ptr<Molecule>& mol,
                      const SharedMatrix& overlap) {
        const int nbf = basis->nbf();
        source_row_.resize(nbf);
        std::iota(source_row_.begin(), source_row_.end(), 0);
        ao_scale_.resize(nbf);
        for (int mu = 0; mu < nbf; ++mu) {
            // Same conversion as Wavefunction.write_molden: Molden uses unit
            // normalized AOs, including the individual Cartesian components.
            const double norm2 = overlap->get(mu, mu);
            if (!(norm2 > 0.0) || !std::isfinite(norm2))
                throw PSIEXCEPTION("DLPNO orbital export encountered an invalid AO norm.");
            ao_scale_[mu] = std::sqrt(norm2);
        }

        // Psi4 source row -> Molden destination row (writer.py conventions).
        const std::vector<std::vector<int>> order = {
            {}, {2, 0, 1}, {0, 3, 4, 1, 5, 2}, {0, 4, 5, 3, 9, 6, 1, 8, 7, 2},
            {0, 3, 4, 9, 12, 10, 5, 13, 14, 7, 1, 6, 11, 8, 2}};
        std::array<int, 5> shell_kind{};
        for (int s = 0; s < basis->nshell(); ++s) {
            const auto& shell = basis->shell(s);
            const int am = shell.am();
            if (am > 4)
                throw PSIEXCEPTION("DLPNO Molden export supports orbital basis shells through G.");
            const int kind = shell.is_pure() ? 1 : 2;
            if (am >= 2 && shell_kind[am] && shell_kind[am] != kind)
                throw PSIEXCEPTION("Molden cannot mix pure and Cartesian shells of the same angular momentum.");
            shell_kind[am] = kind;
            if ((am == 1 && basis->has_puream()) || (am >= 2 && shell.is_cartesian())) {
                for (int j = 0; j < shell.nfunction(); ++j)
                    source_row_[shell.function_index() + order[am][j]] = shell.function_index() + j;
            }
        }

        std::ostringstream header;
        header.imbue(std::locale::classic());
        header << std::scientific << std::setprecision(16);
        header << "[Atoms] (AU)\n";
        for (int atom = 0; atom < mol->natom(); ++atom)
            header << mol->symbol(atom) << " " << atom + 1 << " " << static_cast<int>(mol->Z(atom)) << " "
                   << mol->x(atom) << " " << mol->y(atom) << " " << mol->z(atom) << "\n";
        header << "[GTO]\n";
        for (int atom = 0; atom < mol->natom(); ++atom) {
            header << atom + 1 << " 0\n";
            for (int local_shell = 0; local_shell < basis->nshell_on_center(atom); ++local_shell) {
                const auto& shell = basis->shell(basis->shell_on_center(atom, local_shell));
                header << shell.amchar() << " " << shell.nprimitive() << " 1.00\n";
                for (int p = 0; p < shell.nprimitive(); ++p)
                    header << shell.exp(p) << " " << shell.original_coef(p) << "\n";
            }
            header << "\n";
        }
        // D and F flags are coupled in Molden; G is independent. This also
        // describes bases containing pure D and Cartesian F (or the reverse).
        if (shell_kind[2] == 1) header << (shell_kind[3] == 2 ? "[5D10F]\n" : "[5D7F]\n");
        else if (shell_kind[3] == 1) header << "[7F]\n";
        if (shell_kind[4] == 1) header << "[9G]\n";
        header_ = header.str();

        root_ = std::filesystem::absolute("dlpno_info");
        if (!make_directory(root_)) {
            const auto base = root_;
            // Preserve earlier results, including any incomplete export. Never
            // delete a user's files or mix tuples from different calculations.
            for (size_t run = 2;; ++run) {
                root_ = base / ("run_" + std::to_string(run));
                if (make_directory(root_)) break;
            }
        }
        index_.open(root_ / "index.tsv");
        index_.imbue(std::locale::classic());
        if (!index_) throw PSIEXCEPTION("Cannot open DLPNO orbital index in " + root_.string());
        index_ << "file\torbitals\n";
        outfile->Printf("\n  ==> DLPNO Orbital Export <==\n\n    Molden files: %s\n", root_.string().c_str());
    }

    void write(const std::string& kind, const std::string& name, const SharedMatrix& C,
               const SharedVector& energies, const std::vector<double>& occupations = {}) {
        if (!C || C->nirrep() != 1 || C->nrow() != static_cast<int>(source_row_.size()) || !energies ||
            energies->dim() != C->ncol() || (!occupations.empty() && occupations.size() != C->ncol()))
            throw PSIEXCEPTION("Inconsistent dimensions in DLPNO orbital export: " + name);
        if (C->ncol() == 0) return;
        make_directory(root_ / kind);
        const std::string relative = kind + "/" + name + ".molden";
        std::ofstream out(root_ / relative);
        out.imbue(std::locale::classic());
        if (!out) throw PSIEXCEPTION("Cannot open DLPNO orbital file " + (root_ / relative).string());
        out << std::scientific << std::setprecision(16);
        out << "[Molden Format]\n[Title]\nDLPNO " << name << " (AO coefficients)\n" << header_ << "[MO]\n";
        for (int a = 0; a < C->ncol(); ++a) {
            out << " Sym= A\n Ene= " << energies->get(a) << "\n Spin= Alpha\n Occup= "
                << (occupations.empty() ? 0.0 : occupations[a]) << "\n";
            for (int mu = 0; mu < C->nrow(); ++mu) {
                const int source = source_row_[mu];
                out << mu + 1 << " " << ao_scale_[source] * C->get(source, a) << "\n";
            }
        }
        out.close();
        if (!out) throw PSIEXCEPTION("Failed writing DLPNO orbital file " + (root_ / relative).string());
        index_ << relative << "\t" << C->ncol() << "\n";
        ++file_count_;
    }

    void finish() {
        index_.close();
        if (!index_) throw PSIEXCEPTION("Failed writing DLPNO orbital index in " + root_.string());
        outfile->Printf("    Wrote %zu orbital sets; index.tsv lists files and orbital counts.\n"
                        "    Tuple indices are 1-based active LMOs (columns of LMO/lmo.molden).\n"
                        "    Natural-orbital spaces have Occup=0 (not density eigenvalues).\n\n", file_count_);
    }
};

SharedMatrix DLPNO::orbital_coefficients_ao(const std::vector<int>& domain, const SharedMatrix& X) const {
    if (!X || X->nirrep() != 1 || X->nrow() != static_cast<int>(domain.size()))
        throw PSIEXCEPTION("DLPNO orbital export: PAO domain and orbital transformation have different dimensions.");
    for (int u : domain) {
        if (u < 0 || u >= C_pao_->ncol())
            throw PSIEXCEPTION("DLPNO orbital export: invalid PAO/SOMO domain index.");
    }
    // The ROHF final maps include the appended SOMO columns of C_pao_.
    return linalg::doublet(submatrix_cols(*C_pao_, domain), X);
}

void DLPNO::write_orbitals() const {
    if (!options_.get_bool("DLPNO_WRITE_ORBITALS")) return;
    if (brueckner_orbs_ && !brueckner_converged_)
        throw PSIEXCEPTION("DLPNO orbitals can only be exported after Brueckner convergence.");

    DLPNOOrbitalWriter writer(basisset_, molecule_, reference_wavefunction_->S());
    auto F_average = F_reference_a_->clone();
    F_average->add(F_reference_b_);
    F_average->scale(0.5);
    std::vector<double> occupied(C_lmo_->ncol(), 1.0);
    for (int i = 0; i < nbeta_ - nfrzc(); ++i) occupied[i] = 2.0;
    writer.write("LMO", "lmo", C_lmo_, dlpno_orbital_energies(C_lmo_, F_average), occupied);

    if (brueckner_orbs_) {
        // Ca_ is the final rotated determinant, including frozen occupied and
        // virtual MOs. epsilon_a_ still belongs to the input SCF determinant;
        // compute diagonal Fock expectations in the actual exported orbitals.
        std::vector<double> occupation(Ca_->ncol(), 0.0);
        for (int p = 0; p < nalpha_; ++p) occupation[p] += 1.0;
        for (int p = 0; p < nbeta_; ++p) occupation[p] += 1.0;
        writer.write("Brueckner", "brueckner", Ca_, dlpno_orbital_energies(Ca_, F_average), occupation);
    }
    write_natural_orbitals(writer);
    writer.finish();
}

void DLPNO::write_natural_orbitals(DLPNOOrbitalWriter& writer) const {
    for (size_t ij = 0; ij < ij_to_i_j_.size(); ++ij) {
        const auto& [i, j] = ij_to_i_j_[ij];
        if (i > j) continue;  // ij and ji share the same spatial PNOs.
        if (!X_pno_.at(ij) || X_pno_[ij]->ncol() == 0) continue;
        writer.write("PNO", dlpno_orbital_tuple_name("pno", {i, j}),
                     orbital_coefficients_ao(lmopair_to_paos_.at(ij), X_pno_[ij]), e_pno_.at(ij));
    }
}

void DLPNOCCSD_T::write_natural_orbitals(DLPNOOrbitalWriter& writer) const {
    DLPNO::write_natural_orbitals(writer);
    for (size_t ijk = 0; ijk < ijk_to_i_j_k_.size(); ++ijk) {
        const auto& [i, j, k] = ijk_to_i_j_k_[ijk];
        if (!X_tno_.at(ijk) || X_tno_[ijk]->ncol() == 0) continue;
        writer.write("TNO", dlpno_orbital_tuple_name("tno", {i, j, k}),
                     orbital_coefficients_ao(lmotriplet_to_paos_.at(ijk), X_tno_[ijk]), e_tno_.at(ijk));
    }
}

void RO_DLPNOCCSD_T::write_natural_orbitals(DLPNOOrbitalWriter& writer) const {
    DLPNO::write_natural_orbitals(writer);
    auto F_average = F_pao_a_->clone();
    F_average->add(F_pao_b_);
    F_average->scale(0.5);
    for (size_t ijk = 0; ijk < ijk_to_i_j_k_ro_.size(); ++ijk) {
        // Keep oriented RO tuples: permutations need not represent the same
        // retained spin cases. Each file holds the common spatial TNO basis.
        const auto& [i, j, k] = ijk_to_i_j_k_ro_[ijk];
        const auto& X = X_tno_ro_.at(ijk);
        if (!X || X->ncol() == 0) continue;
        const auto& domain = lmotriplet_to_paos_ro_.at(ijk);
        // F_tno_spin_ has spin-forbidden SOMO slots masked to zero; use the
        // unmasked physical Fock matrix for meaningful display energies.
        auto F_domain = submatrix_rows_and_cols(*F_average, domain, domain);
        writer.write("TNO", dlpno_orbital_tuple_name("tno", {i, j, k}),
                     orbital_coefficients_ao(domain, X), dlpno_orbital_energies(X, F_domain));
    }
}

#ifdef USING_Einsums
void DLPNOCCSDT_Q::write_natural_orbitals(DLPNOOrbitalWriter& writer) const {
    DLPNOCCSD_T::write_natural_orbitals(writer);
    for (size_t ijkl = 0; ijkl < ijkl_to_i_j_k_l_.size(); ++ijkl) {
        const auto& [i, j, k, l] = ijkl_to_i_j_k_l_[ijkl];
        if (!X_qno_.at(ijkl) || X_qno_[ijkl]->ncol() == 0) continue;
        writer.write("QNO", dlpno_orbital_tuple_name("qno", {i, j, k, l}),
                     orbital_coefficients_ao(lmoquadruplet_to_paos_.at(ijkl), X_qno_[ijkl]), e_qno_.at(ijkl));
    }
}
#endif

}  // namespace dlpno
}  // namespace psi
