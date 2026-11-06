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

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/twobody.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace psi {
namespace dlpno {
namespace {

// One generation of triplet blocks. A Jacobi sweep reads only from the old
// generation, including in the PSIO case. Each parallel worker owns its output row.
class TriplesBlocks {
   public:
    TriplesBlocks(const std::vector<int>& ranks, std::string label, std::shared_ptr<PSIO> disk,
                  std::vector<SharedMatrix> initial = {})
        : ranks_(ranks), label_(std::move(label)), disk_(std::move(disk)), blocks_(std::move(initial)) {
        if (!disk_) blocks_.resize(ranks_.size());
    }

    SharedMatrix read(int row) {
        if (!disk_) return blocks_[row];
        int n = ranks_[row];
        auto result = std::make_shared<Matrix>(key(row), n, n * n);
#pragma omp critical(dlpno_response_io)
        result->load(disk_, PSIF_DLPNO_TRIPLES, Matrix::SubBlocks);
        return result;
    }

    void write(int row, SharedMatrix block) {
        if (disk_) {
            block->set_name(key(row));
#pragma omp critical(dlpno_response_io)
            block->save(disk_, PSIF_DLPNO_TRIPLES, Matrix::SubBlocks);
        } else {
            blocks_[row] = std::move(block);
        }
    }

    void swap_generation(TriplesBlocks& other) {
        blocks_.swap(other.blocks_);
        label_.swap(other.label_);
    }

   private:
    std::string key(int row) const { return label_ + " " + std::to_string(row); }
    const std::vector<int>& ranks_;
    std::string label_;
    std::shared_ptr<PSIO> disk_;
    std::vector<SharedMatrix> blocks_;
};

// Forward row r contains -F(l,m) S(r,s)^(tensor 3) P(ordered_source) t_s.
// Several slots of r may reference the same stored s for repeated indices.
struct TriplesEdge {
    int target, source, l, m;
    std::array<int, 3> ordered_source;
};

}  // namespace

void DLPNOCCSD_T::compute_gradient_model_triples() {
    outfile->Printf("\n  ==> RHF DLPNO-CCSD(T) gradient model: single TNO space <==\n\n");
    ijk_to_i_j_k_.clear();
    i_j_k_to_ijk_.clear();
    triples_sparsity(TripletStage::SingleSpace);
    tno_transform(options_.get_double("T_CUT_TNO"));
    estimate_triples_memory();

    auto evaluate = [&](bool right_only) {
        psio_->open(PSIF_DLPNO_TRIPLES, PSIO_OPEN_NEW);
        try {
            const double t0 = compute_lccsd_t0(true).first;
            if (!right_only) set_scalar_variable("DLPNO SEMICANONICAL (T0) ENERGY", t0);
            solve_gradient_model_triples(right_only);
        } catch (...) {
            psio_->close(PSIF_DLPNO_TRIPLES, 0);
            throw;
        }
        psio_->close(PSIF_DLPNO_TRIPLES, 0);
        return E_T_;
    };

    std::array<double, 2> numerical{};
    double frozen2 = 0.;
    if (debug_) {
        const auto ranks = n_tno_;
        constexpr double h = 3.e-6;
        auto fixed = [&]() { return evaluate(true); };
        numerical[0] = probe_amplitudes(true, amplitude_probe(true), h, fixed);
        const auto direction = amplitude_probe(false);
        frozen2 = probe_amplitudes(false, direction, h, fixed);
        numerical[1] = frozen2;
        if (std::any_of(tno_selection_.begin(), tno_selection_.end(), [](const auto& s) { return bool(s.natural); })) {
            numerical[1] = probe_amplitudes(false, direction, h, [&]() {
                tno_transform(options_.get_double("T_CUT_TNO"), false, &ranks);
                return evaluate(true);
            });
            tno_transform(options_.get_double("T_CUT_TNO"));
        }
    }

    evaluate(false);
    if (debug_) {
        const double tno_error =
            std::abs(scalar_variable("DLPNO (T) L2 TNO SOURCE DIRECTION") - (numerical[1] - frozen2));
        set_scalar_variable("DLPNO (T) L2 TNO DIRECTION ERROR", tno_error);
        for (int k = 0; k < 2; ++k) {
            const auto probe = amplitude_probe(k == 0);
            const auto& source = k == 0 ? triples_source1_ : triples_source2_;
            double analytic = 0.;
            for (int m = 0; m < probe.size(); ++m) analytic += source[m]->vector_dot(probe[m]);
            if (k == 1) analytic *= 0.5;
            const double err = std::abs(analytic - numerical[k]);
            outfile->Printf("    (T) source %d: analytic % .12e, numerical % .12e, error %.3e\n", k + 1, analytic,
                            numerical[k], err);
            set_scalar_variable(k == 0 ? "DLPNO (T) L1 SOURCE DIRECTION ERROR" : "DLPNO (T) L2 SOURCE DIRECTION ERROR",
                                err);
        }
    }

    tno_selection_.clear();  // its doubles source has been accumulated
    e_lccsd_t_ = e_lccsd_ + E_T_;
    const double corr = scalar_variable("CCSD CORRELATION ENERGY") + E_T_;
    const double total = reference_energy_ + corr;
    set_scalar_variable("(T) CORRECTION ENERGY", E_T_);
    set_scalar_variable("DLPNO SCREENED TRIPLETS ENERGY", 0.0);
    set_scalar_variable("DLPNO GRADIENT MODEL TOTAL ENERGY", total);
    set_scalar_variable("DLPNO-CCSD(T) TOTAL ENERGY", total);
    set_scalar_variable("DLPNO-CCSD(T) CORRELATION ENERGY", corr);
    set_scalar_variable("CCSD(T) TOTAL ENERGY", total);
    set_scalar_variable("CCSD(T) CORRELATION ENERGY", corr);
    set_scalar_variable("CURRENT CORRELATION ENERGY", corr);
    set_scalar_variable("CURRENT ENERGY", total);
    outfile->Printf("    Single-space (T) correction: %20.12f\n", E_T_);
    outfile->Printf("    RHF DLPNO-CCSD(T) gradient-model total energy: %20.12f\n", total);
}

void DLPNOCCSD_T::solve_gradient_model_triples(bool right_only) {
    const int count = n_tno_.size(), naocc = nalpha_ - nfrzc();
    const double f_cut = options_.get_double("F_CUT_T");
    const int maxiter = options_.get_int("DLPNO_MAXITER");
    const double tol = options_.get_double("R_CONVERGENCE");

    std::vector<std::vector<TriplesEdge>> forward(count), transpose(count);
    for (int r = 0; r < count; ++r) {
        if (!n_tno_[r]) continue;
        auto [i, j, k] = ijk_to_i_j_k_[r];
        const std::array<int, 3> occ{i, j, k};
        for (int slot = 0; slot < 3; ++slot) {
            const int pair = i_j_to_ij_[occ[(slot + 1) % 3]][occ[(slot + 2) % 3]];
            // This pair domain contains every occupied l that can form the new triplet.
            for (int l : lmopair_to_lmos_[pair]) {
                const int m = occ[slot];
                if (l == m || std::abs((*F_lmo_)(l, m)) < f_cut) continue;
                auto ordered = occ;
                ordered[slot] = l;
                auto it = i_j_k_to_ijk_.find(triplet_key(ordered[0], ordered[1], ordered[2], naocc));
                if (it == i_j_k_to_ijk_.end() || !n_tno_[it->second]) continue;
                TriplesEdge edge{r, it->second, l, m, ordered};
                forward[r].push_back(edge);
                transpose[it->second].push_back(edge);
            }
        }
    }

    auto overlap = [&](const TriplesEdge& edge) {
        const int r = edge.target, s = edge.source;
        auto metric = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[r], lmotriplet_to_paos_[s]);
        return linalg::triplet(X_tno_[r], metric, X_tno_[s], true, false, false);
    };

    auto projected = [&](const TriplesEdge& edge, const SharedMatrix& x, bool left) {
        auto S = overlap(edge);
        const auto& p = edge.ordered_source;
        const int nr = n_tno_[edge.target], ns = n_tno_[edge.source];
        if (left) {
            auto y = matmul_3d(x, S->transpose(), nr, ns);
            return triples_permuter(y, p[0], p[1], p[2], true);
        }
        return matmul_3d(triples_permuter(x, p[0], p[1], p[2]), S, ns, nr);
    };

    if (debug_ && !right_only) {
        // Probe the actual packed operator on unrestricted test vectors. Checking
        // just the converged physical tensors can hide an inverse-permutation error.
        auto probe = [&](int row, int seed) {
            const int n = n_tno_[row];
            auto x = std::make_shared<Matrix>(n, n * n);
            for (size_t v = 0; v < static_cast<size_t>(n) * n * n; ++v)
                x->get_pointer()[v] = std::sin((row + 1) * 0.137 + (v + 1) * 0.173 * seed);
            return x;
        };
        double right = 0.0, left = 0.0;
        for (int r = 0; r < count; ++r) {
            if (!n_tno_[r]) continue;
            auto x = probe(r, 2), y = probe(r, 5);
            for (const auto& e : forward[r])
                right -= (*F_lmo_)(e.l, e.m) * y->vector_dot(projected(e, probe(e.source, 2), false));
            for (const auto& e : transpose[r])
                left -= (*F_lmo_)(e.l, e.m) * x->vector_dot(projected(e, probe(e.target, 5), true));
        }
        const double scale = std::max({1.0, std::abs(right), std::abs(left)});
        set_scalar_variable("DLPNO (T) TRANSPOSE DOT ERROR", std::abs(right - left) / scale);
    }

    // Eq. 53, in the same packed triplet coordinates as compute_t_iteration_energy.
    // P^T is needed when differentiating with respect to V (especially for ties).
    auto energy_moment = [&](int r, const SharedMatrix& v, bool adjoint) {
        auto [i, j, k] = ijk_to_i_j_k_[r];
        auto q = v->clone();
        q->scale(8.0);
        q->axpy(-4.0, triples_permuter(v, k, j, i, adjoint));
        q->axpy(-4.0, triples_permuter(v, i, k, j, adjoint));
        q->axpy(-4.0, triples_permuter(v, j, i, k, adjoint));
        q->axpy(2.0, triples_permuter(v, j, k, i, adjoint));
        q->axpy(2.0, triples_permuter(v, k, i, j, adjoint));
        q->scale(i == k ? 1.0 / 6.0 : (i == j || j == k ? 0.5 : 1.0));
        return q;
    };

    auto moments_disk = write_intermediates_ ? psio_ : nullptr;
    auto amps_disk = write_amplitudes_ ? psio_ : nullptr;
    TriplesBlocks W(n_tno_, "W", moments_disk, std::move(W_iajbkc_));
    TriplesBlocks V(n_tno_, "V", moments_disk, std::move(V_iajbkc_));
    TriplesBlocks T(n_tno_, "T", amps_disk, std::move(T_iajbkc_));
    TriplesBlocks Z(n_tno_, "Z", amps_disk);

    auto solve = [&](TriplesBlocks& amplitudes, bool left) {
        TriplesBlocks next(n_tno_, left ? "Z next" : "T next", amps_disk);
        const auto& rows = left ? transpose : forward;
        double residual = 0.0;
        if (count == 0) return residual;
        for (int iter = 0; iter <= maxiter; ++iter) {
            residual = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(max : residual)
            for (int r = 0; r < count; ++r) {
                const int n = n_tno_[r];
                if (!n) continue;
                auto rhs = left ? energy_moment(r, V.read(r), false) : W.read(r)->clone();
                for (const auto& edge : rows[r]) {
                    const auto x = amplitudes.read(left ? edge.target : edge.source);
                    rhs->axpy(-(*F_lmo_)(edge.l, edge.m), projected(edge, x, left));
                }

                auto [i, j, k] = ijk_to_i_j_k_[r];
                const double occ = (*F_lmo_)(i, i) + (*F_lmo_)(j, j) + (*F_lmo_)(k, k);
                const auto& eps = *e_tno_[r];
                auto x = amplitudes.read(r);
                for (int a = 0; a < n; ++a)
                    for (int b = 0; b < n; ++b)
                        for (int c = 0; c < n; ++c) {
                            auto& value = (*rhs)(a, b * n + c);
                            double d = eps(a) + eps(b) + eps(c) - occ;
                            double err = value + d * (*x)(a, b * n + c);
                            double update = -value / d;
                            residual = (std::isfinite(err) && std::isfinite(update))
                                           ? std::max(residual, std::abs(err))
                                           : std::numeric_limits<double>::infinity();
                            value = update;
                        }
                next.write(r, std::move(rhs));
            }
            outfile->Printf("    %-2s %3d       max |residual| %10.3e\n", left ? "Z3" : "T3", iter, residual);
            if (residual <= tol) return residual;  // The measured generation satisfies every equation.
            if (!std::isfinite(residual)) break;
            amplitudes.swap_generation(next);
        }
        throw PSIEXCEPTION(std::string("Single-space DLPNO ") + (left ? "Z3" : "T3") + " equations did not converge.");
    };

    const double right_residual = solve(T, false);
    if (right_only) {
        E_T_ = 0.;
        for (int r = 0; r < count; ++r)
            if (n_tno_[r]) E_T_ += energy_moment(r, V.read(r), false)->vector_dot(T.read(r));
        return;
    }
    set_scalar_variable("DLPNO (T) RIGHT RESIDUAL", right_residual);
    for (int r = 0; r < count; ++r) {
        int n = n_tno_[r];
        if (n) Z.write(r, std::make_shared<Matrix>(n, n * n));
    }
    set_scalar_variable("DLPNO (T) ADJOINT RESIDUAL", solve(Z, true));

    // With R = W + B t and L = <P V,t> + <z,R>, B^T z = -P V.
    // Thus <z,W> independently reconstructs the ordinary (T) energy. At fixed
    // local spaces its variation is <P^T t,dV> + <z,dW> + <z,(dB)t>.
    E_T_ = 0.0;
    double dual_energy = 0.0;
    for (int r = 0; r < count; ++r) {
        if (!n_tno_[r]) continue;
        auto t = T.read(r);
        e_ijk_[r] = energy_moment(r, V.read(r), false)->vector_dot(t);
        E_T_ += e_ijk_[r];
        dual_energy += Z.read(r)->vector_dot(W.read(r));
    }
    set_scalar_variable("DLPNO (T) ADJOINT ENERGY", dual_energy);

    triples_source1_.clear();
    triples_source2_.clear();
    for (const auto& t : T_ia_) triples_source1_.push_back(std::make_shared<Matrix>(t->nrow(), 1));
    for (const auto& t : T_iajb_) triples_source2_.push_back(std::make_shared<Matrix>(t->nrow(), t->ncol()));

    std::vector<SharedMatrix> x_bar(count);
    for (int r = 0; r < count; ++r) {
        if (!tno_selection_[r].natural) continue;  // full kept spaces have zero projector response
        auto t = T.read(r), z = Z.read(r);
        auto f_bar = linalg::doublet(z, t, false, true);
        // All three virtual Fock legs. Retaining off-diagonal variations here
        // permits differentiating the subspace in an arbitrary kept-orbital gauge.
        for (auto perm : {std::array<int, 3>{1, 0, 2}, std::array<int, 3>{2, 1, 0}}) {
            auto zp = triples_permuter(z, perm[0], perm[1], perm[2]);
            auto tp = triples_permuter(t, perm[0], perm[1], perm[2]);
            f_bar->add(linalg::doublet(zp, tp, false, true));
        }
        f_bar->hermitivitize();
        f_bar->scale(2.);
        auto F = submatrix_rows_and_cols(*F_pao_, lmotriplet_to_paos_[r], lmotriplet_to_paos_[r]);
        x_bar[r] = linalg::triplet(F, X_tno_[r], f_bar);
    }
    for (int r = 0; r < count; ++r) {
        for (const auto& edge : forward[r]) {
            int s = edge.source;
            if (!x_bar[r] && !x_bar[s]) continue;
            const auto& p = edge.ordered_source;
            auto t = triples_permuter(T.read(s), p[0], p[1], p[2]);
            auto g = tno_overlap_adjoint(overlap(edge), t, Z.read(r));
            g->scale(-(*F_lmo_)(edge.l, edge.m));
            auto M = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[r], lmotriplet_to_paos_[s]);
            if (x_bar[r]) x_bar[r]->add(linalg::triplet(M, X_tno_[s], g, false, false, true));
            if (x_bar[s]) x_bar[s]->add(linalg::triplet(M, X_tno_[r], g, true, false, false));
        }
    }

    // Accumulate dL/dJ once per auxiliary shell pair. Local domains may overlap.
    // Store the true derivative (including its sign), with both PQ and QP
    // contributions folded into each off-diagonal shell block.
    std::vector<std::map<int, SharedMatrix>> density(ribasis_->nshell());
    std::vector<ThreeCenterDensity> three_center(ribasis_->nbf());
    const auto probe = debug_ ? amplitude_probe(false) : std::vector<SharedMatrix>{};
    double tno_direction = 0.;
    double metric_scale = 0.0;
    for (int r = 0; r < count; ++r) {
        if (!n_tno_[r]) continue;
        auto v_bar = energy_moment(r, T.read(r), true);
        auto w_bar = Z.read(r)->clone();
        w_bar->add(v_bar);  // V = W + the singles and F_ov terms.
        auto g = triples_df_response(r, w_bar, v_bar, three_center, x_bar[r]);
        if (x_bar[r]) tno_direction += add_tno_density_source(r, x_bar[r], probe);
        const auto& aux = lmotriplet_to_ribfs_[r];
        for (int u = 0; u < aux.size(); ++u) {
            const int P = ribasis_->function_to_shell(aux[u]);
            const int p = aux[u] - ribasis_->shell_to_basis_function(P);
            for (int v = 0; v < aux.size(); ++v) {
                metric_scale += (*g)(u, v) * (*full_metric_)(aux[u], aux[v]);
                const int Q = ribasis_->function_to_shell(aux[v]);
                if (Q > P) continue;
                auto& block = density[P][Q];
                if (!block)
                    block = std::make_shared<Matrix>(ribasis_->shell(P).nfunction(), ribasis_->shell(Q).nfunction());
                const int q = aux[v] - ribasis_->shell_to_basis_function(Q);
                (*block)(p, q) += (P == Q ? 1.0 : 2.0) * (*g)(u, v);
            }
        }
    }

    if (debug_) set_scalar_variable("DLPNO (T) L2 TNO SOURCE DIRECTION", tno_direction);

    // Stored CC Lambda2 is dual to half the ordered-pair residual sum.
    // Including both orientations (also for ii) gives its source.
    for (int ij = 0; ij < triples_source2_.size(); ++ij) {
        int ji = ij_to_ji_[ij];
        if (ij > ji) continue;
        auto g = triples_source2_[ij]->clone();
        g->add(triples_source2_[ji]->transpose());
        triples_source2_[ij] = g;
        if (ij != ji) triples_source2_[ji] = g->transpose();
    }

    auto gradient = std::make_shared<Matrix>("DLPNO (T) AUXILIARY METRIC GRADIENT", molecule_->natom(), 3);
    // Each shell derivative is evaluated once for the complete set of nuclei.
    auto zero = BasisSet::zero_ao_basis_set();
    IntegralFactory factory(ribasis_, zero, ribasis_, zero);
    std::unique_ptr<TwoBodyAOInt> ints(factory.eri(1));
    for (int P = 0; P < density.size(); ++P) {
        for (const auto& [Q, block] : density[P]) {
            const int atomP = ribasis_->shell_to_center(P), atomQ = ribasis_->shell_to_center(Q);
            if (atomP == atomQ) continue;  // a common translation of both centers
            ints->compute_shell_deriv1(P, 0, Q, 0);
            const auto& buf = ints->buffers();
            for (int xyz = 0; xyz < 3; ++xyz) {
                const int size = block->nrow() * block->ncol();
                (*gradient)(atomP, xyz) += C_DDOT(size, block->get_pointer(), 1, buf[xyz], 1);
                (*gradient)(atomQ, xyz) += C_DDOT(size, block->get_pointer(), 1, buf[3 + xyz], 1);
            }
        }
    }
    set_array_variable("DLPNO (T) AUXILIARY METRIC GRADIENT", gradient);
    set_array_variable("DLPNO (T) THREE-CENTER GRADIENT", contract_three_center(three_center));
    set_scalar_variable("DLPNO (T) AUXILIARY METRIC SCALE DERIVATIVE", metric_scale);
}

// Derivative of <z,S^(tensor 3) x> with respect to S. Move each
// differentiated leg to the first axis and project only the other two.
SharedMatrix DLPNOCCSD_T::tno_overlap_adjoint(const SharedMatrix& S, const SharedMatrix& x, const SharedMatrix& z) {
    const int nr = S->nrow(), ns = S->ncol();
    auto g = std::make_shared<Matrix>(nr, ns);
    for (auto perm : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{1, 0, 2}, std::array<int, 3>{2, 1, 0}}) {
        auto y = triples_permuter(x, perm[0], perm[1], perm[2]);
        auto zp = triples_permuter(z, perm[0], perm[1], perm[2]);
        auto partial = std::make_shared<Matrix>(ns, nr * nr);
        auto slab = std::make_shared<Matrix>(ns, ns);
        for (int a = 0; a < ns; ++a) {
            std::copy_n(&(*y)(a, 0), ns * ns, slab->get_pointer());
            auto proj = linalg::triplet(S, slab, S, false, false, true);
            std::copy_n(proj->get_pointer(), nr * nr, &(*partial)(a, 0));
        }
        g->add(linalg::doublet(zp, partial, false, true));
    }
    return g;
}

double DLPNOCCSD_T::add_tno_density_source(int r, const SharedMatrix& x_bar, const std::vector<SharedMatrix>& probe) {
    double directional = 0.;
    const auto& sel = tno_selection_[r];
    const int kept = n_tno_[r], n = sel.natural->ncol();
    Dimension zero(1), k(1), all(1);
    k.fill(kept);
    all.fill(n);
    auto C = sel.natural->get_block({zero, sel.natural->rowspi()}, {zero, k});
    auto X = sel.natural->get_block({zero, sel.natural->rowspi()}, {k, all});
    auto multiplier = linalg::triplet(X, x_bar, sel.canonical, true, false, true);
    double occ_scale = std::max(std::abs((*sel.occupations)(0)), std::abs((*sel.occupations)(n - 1)));
    for (int d = 0; d < n - kept; ++d)
        for (int a = 0; a < kept; ++a) {
            double gap = (*sel.occupations)(a) - (*sel.occupations)(kept + d);
            if (std::abs(gap) <= 32. * std::numeric_limits<double>::epsilon() * occ_scale)
                throw PSIEXCEPTION("Undefined DLPNO TNO response: the retained rank splits a degenerate occupation.");
            (*multiplier)(d, a) /= gap;
        }
    const auto [i, j, l] = ijk_to_i_j_k_[r];
    for (int ij : {i_j_to_ij_[i][j], i_j_to_ij_[j][l], i_j_to_ij_[i][l]}) {
        auto metric = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmotriplet_to_paos_[r]);
        auto P = linalg::doublet(X_pno_[ij], metric, true, false);
        auto H = linalg::triplet(linalg::doublet(P, X), multiplier, linalg::doublet(P, C), false, false, true);
        H->hermitivitize();
        const auto [p, q] = ij_to_i_j_[ij];
        H->scale(p == q ? 1. / 6. : 1. / 3.);
        // D = U t^T + U^T t, U = 2t - t^T.
        auto u_bar = linalg::doublet(H, T_iajb_[ij]);
        u_bar->add(linalg::doublet(T_iajb_[ij], H));
        auto dt = u_bar->transpose();
        dt->scale(-1.);
        dt->axpy(2., u_bar);
        dt->add(linalg::doublet(H, Tt_iajb_[ij]));
        dt->add(linalg::doublet(Tt_iajb_[ij], H));
        triples_source2_[ij]->add(dt);
        if (!probe.empty()) directional += probe[ij]->vector_dot(dt);
    }
    return directional;
}

}  // namespace dlpno
}  // namespace psi
