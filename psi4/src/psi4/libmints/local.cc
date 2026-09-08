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

#include "local.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

#include "psi4/libqt/qt.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/thc_eri.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

using namespace psi;

namespace psi {

Localizer::Localizer(std::shared_ptr<BasisSet> primary, std::shared_ptr<Matrix> C) : primary_(primary), C_(C) {
    if (C->nirrep() != 1) {
        throw PSIEXCEPTION("Localizer: C matrix is not C1");
    }
    if (C->rowspi()[0] != primary->nbf()) {
        throw PSIEXCEPTION("Localizer: C matrix does not match basis");
    }
    common_init();
}
Localizer::~Localizer() {}
void Localizer::common_init() {
    print_ = 0;
    debug_ = 0;
    bench_ = 0;
    convergence_ = 1.0E-8;
    gradient_convergence_ = 1.0E-8;
    maxiter_ = 50;
    use_augmented_hessian_ = true;
    augmented_hessian_start_ = 3;
    augmented_hessian_max_rotations_ = 512;
    augmented_hessian_trust_radius_ = 0.25;
    saddle_tolerance_ = 1.0E-8;
    converged_ = false;
}
std::shared_ptr<Localizer> Localizer::build(const std::string& type, std::shared_ptr<BasisSet> primary,
                                            std::shared_ptr<Matrix> C, Options& options) {
    std::shared_ptr<Localizer> local;

    if (type == "BOYS") {
        local = std::make_shared<BoysLocalizer>(primary, C);
    } else if (type == "PIPEK_MEZEY") {
        local = std::make_shared<PMLocalizer>(primary, C);
    } else {
        throw PSIEXCEPTION("Localizer: Unrecognized localization algorithm");
    }

    local->set_print(options.get_int("PRINT"));
    local->set_debug(options.get_int("DEBUG"));
    local->set_bench(options.get_int("BENCH"));
    local->set_convergence(options.get_double("LOCAL_CONVERGENCE"));
    local->set_maxiter(options.get_int("LOCAL_MAXITER"));
    if (options.exists("LOCAL_GRADIENT_CONVERGENCE"))
        local->set_gradient_convergence(options.get_double("LOCAL_GRADIENT_CONVERGENCE"));
    if (options.exists("LOCAL_USE_AUGMENTED_HESSIAN"))
        local->set_use_augmented_hessian(options.get_bool("LOCAL_USE_AUGMENTED_HESSIAN"));
    if (options.exists("LOCAL_AH_START"))
        local->set_augmented_hessian_start(options.get_int("LOCAL_AH_START"));
    if (options.exists("LOCAL_AH_MAX_ROTATIONS"))
        local->set_augmented_hessian_max_rotations(options.get_int("LOCAL_AH_MAX_ROTATIONS"));
    if (options.exists("LOCAL_AH_TRUST_RADIUS"))
        local->set_augmented_hessian_trust_radius(options.get_double("LOCAL_AH_TRUST_RADIUS"));
    if (options.exists("LOCAL_SADDLE_TOLERANCE"))
        local->set_saddle_tolerance(options.get_double("LOCAL_SADDLE_TOLERANCE"));

    return local;
}
std::shared_ptr<Localizer> Localizer::build(const std::string& type, std::shared_ptr<BasisSet> primary,
                                            std::shared_ptr<Matrix> C) {
    return Localizer::build(type, primary, C, Process::environment.options);
}
std::shared_ptr<Localizer> Localizer::build(std::shared_ptr<BasisSet> primary, std::shared_ptr<Matrix> C,
                                            Options& options) {
    return Localizer::build(options.get_str("LOCAL_TYPE"), primary, C, options);
}
std::shared_ptr<Matrix> Localizer::fock_update(std::shared_ptr<Matrix> Fc) {
    if (!L_ || !U_) {
        throw PSIEXCEPTION("Localizer: run compute() first");
    }

    int nso = L_->rowspi()[0];
    int nmo = L_->colspi()[0];

    if (nmo < 1) return Fc;

    std::shared_ptr<Matrix> Fl = linalg::triplet(U_, Fc, U_, true, false, false);
    double** Fp = Fl->pointer();
    double** Lp = L_->pointer();
    double** Up = U_->pointer();

    std::vector<std::pair<double, int> > order;
    for (int i = 0; i < nmo; i++) {
        order.push_back(std::pair<double, int>(Fp[i][i], i));
    }
    std::sort(order.begin(), order.end());

    std::shared_ptr<Matrix> Fl2(Fl->clone());
    Fl2->copy(Fl);
    double** F2p = Fl2->pointer();
    for (int i = 0; i < nmo; i++) {
        for (int j = 0; j < nmo; j++) {
            Fp[i][j] = F2p[order[i].second][order[j].second];
        }
    }

    std::shared_ptr<Matrix> L2(L_->clone());
    L2->copy(L_);
    double** L2p = L2->pointer();
    std::shared_ptr<Matrix> U2(U_->clone());
    U2->copy(U_);
    double** U2p = U2->pointer();
    for (int i = 0; i < nmo; i++) {
        C_DCOPY(nso, &L2p[0][order[i].second], nmo, &Lp[0][i], nmo);
        C_DCOPY(nmo, &U2p[0][order[i].second], nmo, &Up[0][i], nmo);
    }

    return Fl;
}

namespace {

// Boys and generalized Pipek--Mezey localization can both be written as
//
//     max_U sum_K sum_p [(U^T A^K U)_pp]^2.
//
// For generalized PM, A^K is an atomic population operator; see
// Lehtola and Jonsson, JCTC 2014, doi:10.1021/ct401016x.  Keeping the
// optimizer independent of the population partition lets Mulliken and
// real-space stockholder analyses share the same derivatives and safeguards.

struct LocalizationPair {
    int i;
    int j;
};

struct LocalizationDiagnostics {
    std::vector<double> gradient;
    double max_gradient = 0.0;
    double max_pair_gain = 0.0;
};

double localization_objective(const std::vector<std::shared_ptr<Matrix>>& operators) {
    if (operators.empty()) return 0.0;
    const int nmo = operators.front()->nrow();
    double value = 0.0;
    for (const auto& op : operators) {
        double** Ap = op->pointer();
        value += C_DDOT(nmo, Ap[0], nmo + 1, Ap[0], nmo + 1);
    }
    return value;
}

std::vector<LocalizationPair> localization_pairs(int nmo) {
    std::vector<LocalizationPair> pairs;
    pairs.reserve(nmo * (nmo - 1) / 2);
    for (int i = 0; i < nmo - 1; ++i) {
        for (int j = i + 1; j < nmo; ++j) pairs.push_back({i, j});
    }
    return pairs;
}

LocalizationDiagnostics localization_diagnostics(const std::vector<std::shared_ptr<Matrix>>& operators,
                                                   const std::vector<LocalizationPair>& pairs) {
    LocalizationDiagnostics result;
    result.gradient.assign(pairs.size(), 0.0);

    for (size_t pq = 0; pq < pairs.size(); ++pq) {
        const int i = pairs[pq].i;
        const int j = pairs[pq].j;
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        for (const auto& op : operators) {
            double** Ap = op->pointer();
            const double Ad = Ap[i][i] - Ap[j][j];
            const double Ao = 2.0 * Ap[i][j];
            a += Ad * Ad;
            b += Ao * Ao;
            c += Ad * Ao;
        }
        const double Hd = a - b;
        const double Ho = 2.0 * c;
        result.gradient[pq] = Ho;
        result.max_gradient = std::max(result.max_gradient, std::fabs(Ho));
        result.max_pair_gain = std::max(result.max_pair_gain, 0.25 * (std::hypot(Hd, Ho) - Hd));
    }
    return result;
}

void apply_jacobi_rotation(std::vector<std::shared_ptr<Matrix>>& operators, const std::shared_ptr<Matrix>& U, int i,
                           int j, double theta) {
    const int nmo = U->nrow();
    const double cc = std::cos(theta);
    const double ss = std::sin(theta);
    for (const auto& op : operators) {
        double** Ap = op->pointer();
        // A <- R^T A R, where R_ij is the orbital rotation associated with theta.
        C_DROT(nmo, &Ap[i][0], 1, &Ap[j][0], 1, cc, ss);
        C_DROT(nmo, &Ap[0][i], nmo, &Ap[0][j], nmo, cc, ss);
    }
    // Store the direct input-MO -> localized-MO transformation: U <- U R.
    double** Up = U->pointer();
    C_DROT(nmo, &Up[0][i], nmo, &Up[0][j], nmo, cc, ss);
}

double operator_rotation_derivative(const std::shared_ptr<Matrix>& op, int p, int q,
                                    const LocalizationPair& direction) {
    const int k = direction.i;
    const int l = direction.j;
    double value = 0.0;
    // dA/dx = A K - K A for K_kl = -1 and K_lk = +1.
    if (q == k) value += op->get(p, l);
    if (q == l) value -= op->get(p, k);
    if (p == k) value += op->get(l, q);
    if (p == l) value -= op->get(k, q);
    return value;
}

double gradient_derivative(const std::vector<std::shared_ptr<Matrix>>& operators,
                           const LocalizationPair& gradient_pair, const LocalizationPair& direction) {
    const int i = gradient_pair.i;
    const int j = gradient_pair.j;
    double value = 0.0;
    for (const auto& op : operators) {
        const double Aii = op->get(i, i);
        const double Ajj = op->get(j, j);
        const double Aij = op->get(i, j);
        const double dAii = operator_rotation_derivative(op, i, i, direction);
        const double dAjj = operator_rotation_derivative(op, j, j, direction);
        const double dAij = operator_rotation_derivative(op, i, j, direction);
        value += 4.0 * ((dAii - dAjj) * Aij + (Aii - Ajj) * dAij);
    }
    return value;
}

std::shared_ptr<Matrix> localization_hessian(const std::vector<std::shared_ptr<Matrix>>& operators,
                                             const std::vector<LocalizationPair>& pairs) {
    const int nrot = pairs.size();
    auto H = std::make_shared<Matrix>("Localization Hessian", nrot, nrot);
    for (int pq = 0; pq < nrot; ++pq) {
        for (int rs = 0; rs <= pq; ++rs) {
            // The derivative of the coordinate gradient is not symmetric away from the
            // expansion point because finite rotations do not commute.  The symmetric
            // exponential-coordinate Hessian is its Jordan symmetrization.
            const double value = 0.5 * (gradient_derivative(operators, pairs[pq], pairs[rs]) +
                                        gradient_derivative(operators, pairs[rs], pairs[pq]));
            H->set(pq, rs, value);
            H->set(rs, pq, value);
        }
    }
    return H;
}

std::shared_ptr<Matrix> localization_rotation(const std::vector<LocalizationPair>& pairs,
                                              const std::vector<double>& step, double scale, int nmo) {
    auto R = std::make_shared<Matrix>("Localization rotation generator", nmo, nmo);
    for (size_t pq = 0; pq < pairs.size(); ++pq) {
        const double value = scale * step[pq];
        R->set(pairs[pq].i, pairs[pq].j, -value);
        R->set(pairs[pq].j, pairs[pq].i, value);
    }
    R->expm(6, true);
    return R;
}

struct AugmentedHessianResult {
    bool accepted = false;
    bool stable = false;
    double largest_curvature = 0.0;
    double step_norm = 0.0;
};

AugmentedHessianResult augmented_hessian_step(std::vector<std::shared_ptr<Matrix>>& operators,
                                              std::shared_ptr<Matrix>& U,
                                              const std::vector<LocalizationPair>& pairs,
                                              const LocalizationDiagnostics& diagnostics, double objective,
                                              double saddle_tolerance, double& trust_radius, int debug) {
    // The dense second-order model follows the robust-optimization motivation
    // of Clement, Wang, and Valeev, JCTC 2021, doi:10.1021/acs.jctc.1c00238.
    // Here an augmented-Hessian root replaces a quasi-Newton step so positive
    // curvature is available explicitly for saddle-point control.
    AugmentedHessianResult result;
    const int nrot = pairs.size();
    if (nrot == 0) {
        result.stable = true;
        return result;
    }

    auto H = localization_hessian(operators, pairs);
    auto Hwork = H->clone();
    auto Hvectors = std::make_shared<Matrix>("Localization Hessian eigenvectors", nrot, nrot);
    auto Hvalues = std::make_shared<Vector>("Localization Hessian eigenvalues", nrot);
    Hwork->diagonalize(*Hvectors, *Hvalues, descending);
    result.largest_curvature = Hvalues->get(0);
    const double scaled_saddle_tolerance = saddle_tolerance * std::max(1.0, std::fabs(objective));
    result.stable = result.largest_curvature <= scaled_saddle_tolerance;

    std::vector<double> step(nrot, 0.0);
    const bool stationary = diagnostics.max_gradient < std::sqrt(std::numeric_limits<double>::epsilon());
    if (stationary && !result.stable) {
        // A stationary point with positive curvature is a minimum/saddle of the
        // maximization functional.  Follow the most positive Hessian mode away
        // from it; both signs are tested below.
        for (int pq = 0; pq < nrot; ++pq) step[pq] = Hvectors->get(pq, 0);
    } else if (!stationary) {
        auto AH = std::make_shared<Matrix>("Localization augmented Hessian", nrot + 1, nrot + 1);
        for (int pq = 0; pq < nrot; ++pq) {
            AH->set(0, pq + 1, diagnostics.gradient[pq]);
            AH->set(pq + 1, 0, diagnostics.gradient[pq]);
            for (int rs = 0; rs < nrot; ++rs) AH->set(pq + 1, rs + 1, H->get(pq, rs));
        }
        auto AHvectors = std::make_shared<Matrix>("Localization augmented-Hessian eigenvectors", nrot + 1,
                                                  nrot + 1);
        auto AHvalues = std::make_shared<Vector>("Localization augmented-Hessian eigenvalues", nrot + 1);
        AH->diagonalize(*AHvectors, *AHvalues, descending);
        const double reference_component = AHvectors->get(0, 0);
        if (std::fabs(reference_component) > 1.0E-10) {
            for (int pq = 0; pq < nrot; ++pq) step[pq] = AHvectors->get(pq + 1, 0) / reference_component;
        } else {
            // The selected root has become a pure curvature mode.  It is still
            // a useful saddle-escape direction, but it cannot be projectively
            // normalized by the augmented-Hessian reference component.
            for (int pq = 0; pq < nrot; ++pq) step[pq] = Hvectors->get(pq, 0);
        }
    } else {
        return result;
    }

    double norm = std::sqrt(std::inner_product(step.begin(), step.end(), step.begin(), 0.0));
    if (!std::isfinite(norm) || norm < std::numeric_limits<double>::epsilon()) return result;
    if (norm > trust_radius) {
        const double scale = trust_radius / norm;
        for (double& value : step) value *= scale;
        norm = trust_radius;
    }

    const int nmo = U->nrow();
    const double acceptance_tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                                        std::max(1.0, std::fabs(objective));
    double best_objective = objective;
    std::vector<std::shared_ptr<Matrix>> best_operators;
    std::shared_ptr<Matrix> best_U;
    double accepted_scale = 0.0;

    // A compact trust-region line search makes the initial implementation safe:
    // only rotations that increase the actual localization functional are kept.
    for (int trial = 0; trial < 8 && !result.accepted; ++trial) {
        const double line_scale = std::ldexp(1.0, -trial);
        for (double sign : {1.0, -1.0}) {
            auto R = localization_rotation(pairs, step, sign * line_scale, nmo);
            std::vector<std::shared_ptr<Matrix>> trial_operators;
            trial_operators.reserve(operators.size());
            for (const auto& op : operators)
                trial_operators.push_back(linalg::triplet(R, op, R, true, false, false));
            const double trial_objective = localization_objective(trial_operators);
            if (trial_objective > best_objective + acceptance_tolerance) {
                best_objective = trial_objective;
                best_operators = std::move(trial_operators);
                best_U = linalg::doublet(U, R, false, false);
                accepted_scale = line_scale;
                result.accepted = true;
            }
        }
    }

    if (result.accepted) {
        operators = std::move(best_operators);
        U = std::move(best_U);
        result.step_norm = norm * accepted_scale;
        if (accepted_scale == 1.0)
            trust_radius = std::min(0.5, 1.5 * trust_radius);
        else
            trust_radius = std::max(1.0E-4, accepted_scale * trust_radius);
    } else {
        trust_radius = std::max(1.0E-4, 0.5 * trust_radius);
    }

    if (debug > 1) {
        outfile->Printf("    Localization AH: lambda(max) = %11.3E, step = %11.3E, accepted = %s\n",
                        result.largest_curvature, result.step_norm, result.accepted ? "yes" : "no");
    }
    return result;
}

}  // namespace

void Localizer::localize_matrix_objective(std::vector<std::shared_ptr<Matrix>> operators,
                                          const std::string& label) {
    const int nmo = C_->ncol();
    if (gradient_convergence_ <= 0.0 || augmented_hessian_max_rotations_ < 0 ||
        augmented_hessian_trust_radius_ <= 0.0 || saddle_tolerance_ < 0.0)
        throw PSIEXCEPTION("Localizer: invalid convergence or augmented-Hessian control parameter");
    L_ = C_->clone();
    U_ = std::make_shared<Matrix>("MO -> localized-MO transformation", nmo, nmo);
    U_->identity();
    converged_ = false;

    if (nmo < 2) {
        converged_ = true;
        return;
    }
    if (operators.empty()) throw PSIEXCEPTION("Localizer: the localization objective has no operators");
    for (const auto& op : operators) {
        if (!op || op->nirrep() != 1 || op->nrow() != nmo || op->ncol() != nmo)
            throw PSIEXCEPTION("Localizer: invalid matrix-objective operator dimensions");
        op->hermitivitize();
    }

    const auto pairs = localization_pairs(nmo);
    const bool dense_hessian_available = pairs.size() <= static_cast<size_t>(augmented_hessian_max_rotations_);
    const bool stability_will_be_checked = use_augmented_hessian_ && dense_hessian_available &&
                                           augmented_hessian_start_ <= maxiter_;
    if (use_augmented_hessian_ && !dense_hessian_available) {
        outfile->Printf("    Dense AH stability analysis disabled for %zu rotations (limit %d).\n\n", pairs.size(),
                        augmented_hessian_max_rotations_);
    }

    std::mt19937 generator(0);
    std::vector<int> order(nmo);
    std::iota(order.begin(), order.end(), 0);
    double objective = localization_objective(operators);
    double old_objective = objective;
    double trust_radius = augmented_hessian_trust_radius_;
    auto diagnostics = localization_diagnostics(operators, pairs);
    bool hessian_stable_at_convergence = false;

    outfile->Printf("    Iteration %24s %14s %14s %14s\n", "Metric", "Rel. change", "Max |grad|", "Max curvature");
    outfile->Printf("    @%-4s %4d %24.16E %14s %14.6E %14s\n", label.c_str(), 0, objective, "-",
                    diagnostics.max_gradient, "-");

    for (int iter = 1; iter <= maxiter_; ++iter) {
        std::shuffle(order.begin(), order.end(), generator);

        // Exact two-orbital maximizations retain the inexpensive and very robust
        // Jacobi behavior of the original Boys/PM implementations.
        for (int p = 0; p < nmo - 1; ++p) {
            for (int q = p + 1; q < nmo; ++q) {
                const int i = order[p];
                const int j = order[q];
                double a = 0.0;
                double b = 0.0;
                double c = 0.0;
                for (const auto& op : operators) {
                    double** Ap = op->pointer();
                    const double Ad = Ap[i][i] - Ap[j][j];
                    const double Ao = 2.0 * Ap[i][j];
                    a += Ad * Ad;
                    b += Ao * Ao;
                    c += Ad * Ao;
                }
                const double Hd = a - b;
                const double Ho = 2.0 * c;
                const double gain = 0.25 * (std::hypot(Hd, Ho) - Hd);
                if (gain <= 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, objective)) continue;
                const double theta = 0.25 * std::atan2(Ho, Hd);
                if (debug_ > 3)
                    outfile->Printf("    @Rotation i = %4d, j = %4d, theta = %24.16E, gain = %11.3E\n", i, j,
                                    theta, gain);
                apply_jacobi_rotation(operators, U_, i, j, theta);
            }
        }

        objective = localization_objective(operators);
        diagnostics = localization_diagnostics(operators, pairs);
        double relative_change = std::fabs(objective - old_objective) / std::max(1.0, std::fabs(old_objective));

        bool stability_checked = false;
        bool stable = false;
        bool ah_accepted = false;
        double largest_curvature = 0.0;
        const bool ah_iteration = use_augmented_hessian_ && dense_hessian_available &&
                                  iter >= std::max(1, augmented_hessian_start_);
        if (ah_iteration) {
            stability_checked = true;
            const auto ah = augmented_hessian_step(operators, U_, pairs, diagnostics, objective, saddle_tolerance_,
                                                   trust_radius, debug_);
            stable = ah.stable;
            ah_accepted = ah.accepted;
            largest_curvature = ah.largest_curvature;
            if (ah_accepted) {
                objective = localization_objective(operators);
                diagnostics = localization_diagnostics(operators, pairs);
                relative_change =
                    std::fabs(objective - old_objective) / std::max(1.0, std::fabs(old_objective));
            }
        }

        if (stability_checked) {
            outfile->Printf("    @%-4s %4d %24.16E %14.6E %14.6E %14.6E%s\n", label.c_str(), iter, objective,
                            relative_change, diagnostics.max_gradient, largest_curvature,
                            ah_accepted ? "  AH" : "");
        } else {
            outfile->Printf("    @%-4s %4d %24.16E %14.6E %14.6E %14s\n", label.c_str(), iter, objective,
                            relative_change, diagnostics.max_gradient, "-");
        }

        const double objective_scale = std::max(1.0, std::fabs(objective));
        const double pair_gain_tolerance =
            std::max(convergence_, 64.0 * std::numeric_limits<double>::epsilon()) * objective_scale;
        const bool pairwise_stable = diagnostics.max_pair_gain < pair_gain_tolerance;
        const bool first_order_converged = diagnostics.max_gradient < gradient_convergence_ && pairwise_stable;
        const bool objective_converged = relative_change < convergence_;
        if (!ah_accepted && first_order_converged && objective_converged &&
            (!stability_will_be_checked || (stability_checked && stable))) {
            converged_ = true;
            hessian_stable_at_convergence = stability_checked && stable;
            break;
        }
        old_objective = objective;
    }

    outfile->Printf("\n");
    if (converged_ && hessian_stable_at_convergence)
        outfile->Printf("    %s Localizer converged to a Hessian-stable maximum.\n\n", label.c_str());
    else if (converged_)
        outfile->Printf("    %s Localizer converged (dense Hessian stability was not tested).\n\n", label.c_str());
    else
        outfile->Printf("    %s Localizer failed to reach a Hessian-stable maximum.\n\n", label.c_str());

    L_ = linalg::doublet(C_, U_, false, false);
}

BoysLocalizer::BoysLocalizer(std::shared_ptr<BasisSet> primary, std::shared_ptr<Matrix> C) : Localizer(primary, C) {
    common_init();
}
BoysLocalizer::~BoysLocalizer() {}
void BoysLocalizer::common_init() {}
void BoysLocalizer::print_header() const {
    outfile->Printf("  ==> Boys Localizer <==\n\n");
    outfile->Printf("    Objective convergence = %11.3E\n", convergence_);
    outfile->Printf("    Gradient convergence  = %11.3E\n", gradient_convergence_);
    outfile->Printf("    Maxiter               = %11d\n", maxiter_);
    outfile->Printf("    Augmented Hessian     = %11s\n", use_augmented_hessian_ ? "enabled" : "disabled");
    if (use_augmented_hessian_) {
        outfile->Printf("    AH start iteration     = %11d\n", augmented_hessian_start_);
        outfile->Printf("    AH trust radius        = %11.3E\n", augmented_hessian_trust_radius_);
    }
    outfile->Printf("\n");
}
void BoysLocalizer::localize() {
    print_header();

    auto factory = std::make_shared<IntegralFactory>(primary_);
    auto dipole = factory->ao_dipole();
    std::vector<std::shared_ptr<Matrix>> dipole_ao;
    for (int xyz = 0; xyz < 3; ++xyz)
        dipole_ao.push_back(std::make_shared<Matrix>("AO dipole", primary_->nbf(), primary_->nbf()));
    dipole->compute(dipole_ao);

    std::vector<std::shared_ptr<Matrix>> dipole_mo;
    dipole_mo.reserve(3);
    for (const auto& component : dipole_ao)
        dipole_mo.push_back(linalg::triplet(C_, component, C_, true, false, false));
    localize_matrix_objective(std::move(dipole_mo), "Boys");
}

PMLocalizer::PMLocalizer(std::shared_ptr<BasisSet> primary, std::shared_ptr<Matrix> C) : Localizer(primary, C) {
    common_init();
}
PMLocalizer::PMLocalizer(std::shared_ptr<BasisSet> primary, std::shared_ptr<Matrix> C,
                         const std::vector<std::shared_ptr<Matrix>>& population_matrices,
                         const std::string& population_method)
    : Localizer(primary, C), population_method_(population_method) {
    population_matrices_.reserve(population_matrices.size());
    for (const auto& population : population_matrices) {
        if (!population) throw PSIEXCEPTION("PMLocalizer: null population matrix");
        population_matrices_.push_back(population->clone());
    }
    common_init();
}
PMLocalizer::~PMLocalizer() {}
void PMLocalizer::common_init() {
    if (population_method_.empty()) population_method_ = "Mulliken";
}
void PMLocalizer::print_header() const {
    outfile->Printf("  ==> Generalized Pipek-Mezey Localizer <==\n\n");
    outfile->Printf("    Population partition  = %11s\n", population_method_.c_str());
    outfile->Printf("    Objective convergence = %11.3E\n", convergence_);
    outfile->Printf("    Gradient convergence  = %11.3E\n", gradient_convergence_);
    outfile->Printf("    Maxiter               = %11d\n", maxiter_);
    outfile->Printf("    Augmented Hessian     = %11s\n", use_augmented_hessian_ ? "enabled" : "disabled");
    if (use_augmented_hessian_) {
        outfile->Printf("    AH start iteration     = %11d\n", augmented_hessian_start_);
        outfile->Printf("    AH trust radius        = %11.3E\n", augmented_hessian_trust_radius_);
    }
    outfile->Printf("\n");
}
void PMLocalizer::localize() {
    print_header();

    if (population_matrices_.empty()) {
        const int nbf = C_->nrow();
        const int nmo = C_->ncol();
        const int natom = primary_->molecule()->natom();
        auto factory = std::make_shared<IntegralFactory>(primary_);
        auto overlap_integral = factory->ao_overlap();
        auto overlap = std::make_shared<Matrix>("AO overlap", nbf, nbf);
        overlap_integral->compute(overlap);
        auto SC = linalg::doublet(overlap, C_, false, false);

        population_matrices_.reserve(natom);
        for (int atom = 0; atom < natom; ++atom)
            population_matrices_.push_back(std::make_shared<Matrix>("Mulliken population", nmo, nmo));

        // Symmetric Mulliken population operators.  Writing PM this way exposes
        // the same generalized objective used by real-space stockholder schemes.
        for (int mu = 0; mu < nbf; ++mu) {
            auto population = population_matrices_[primary_->function_to_center(mu)];
            for (int i = 0; i < nmo; ++i) {
                for (int j = 0; j <= i; ++j) {
                    const double value = 0.5 * (SC->get(mu, i) * C_->get(mu, j) +
                                                C_->get(mu, i) * SC->get(mu, j));
                    population->add(i, j, value);
                    if (i != j) population->add(j, i, value);
                }
            }
        }
    }

    localize_matrix_objective(population_matrices_, "PM");
}

ERLocalizer::ERLocalizer(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, std::shared_ptr<Matrix> C) : Localizer(primary, C) {
    auxiliary_ = auxiliary;
    common_init();
}
ERLocalizer::~ERLocalizer() {}
void ERLocalizer::common_init() {}
void ERLocalizer::print_header() const {
    outfile->Printf("  ==> Edmiston-Ruedenberg (ER) Localizer <==\n");
    outfile->Printf("    By: Andy Jiang and Nate Kitzmiller    \n\n");
    outfile->Printf("    Convergence = %11.3E\n", convergence_);
    outfile->Printf("    Maxiter     = %11d\n", maxiter_);
    outfile->Printf("\n");
}
void ERLocalizer::localize() {
    print_header();

    // => Sizing <= //

    int nso = C_->rowspi()[0];
    int nmo = C_->colspi()[0];

    // => Compute Tensor Hypercontraction form of Two-Electron Integrals (ERIs) <= //

    auto thc_computer = std::make_shared<LS_THC_Computer>(primary_->molecule(), primary_, auxiliary_, Process::environment.options);
    thc_computer->compute_thc_factorization();

    auto Z_IJ = thc_computer->get_Z(); // Z^{IJ}
    auto xI = thc_computer->get_x1();  // x^{I}_{\mu}
    auto x_mo = linalg::doublet(xI, C_, false, false); // x^{I}_{p} = x^{I}_{\mu} C_{\mu p}

    size_t nthc = x_mo->rowspi()[0];

    // Compute two electron integrals of the form (pp|pq) from THC factors

    // (pp|pq) = \sum_{I,J} x^{I}_{p} x^{I}_{p} Z^{IJ} x^{J}_{p} x^{J}_{q}
    auto pppq = std::make_shared<Matrix>("(pp|pq)", nmo, nmo);

    // Intermediates for computing (pp|pq)

    // S^{I}_{p} = x^{I}_{p} x^{I}_{p} => O(N^{2})
    auto SI_p = std::make_shared<Matrix>("SI_p", nthc, nmo);

#pragma omp parallel for collapse(2)
    for (size_t I = 0; I < nthc; ++I) {
        for (size_t p = 0; p < nmo; ++p) {
            double val = x_mo->get(I, p);
            SI_p->set(I, p, val * val);
        } // end p
    } // end I

    // A^{J}_{p} = Z^{JI} S^{I}_{p} => O(N^{3})
    auto AJ_p = linalg::doublet(Z_IJ, SI_p, false, false);

    // (pp|pq) = \sum_{J} A^{J}_{p} x^{J}_{p} x^{J}_{q} => O(N^{3})
#pragma omp parallel for collapse(2)
    for (size_t p = 0; p < nmo; ++p) {
        for (size_t q = 0; q < nmo; ++q) {
            double val = 0.0;
            for (size_t J = 0; J < nthc; ++J) {
                val += AJ_p->get(J, p) * x_mo->get(J, p) * x_mo->get(J, q);
            } // end J
            pppq->set(p, q, val);
        } // end q
    } // end p

    // => Targets <= //

    L_ = std::make_shared<Matrix>("L", nso, nmo);
    U_ = std::make_shared<Matrix>("U", nmo, nmo);
    L_->copy(C_);
    U_->identity();
    converged_ = false;

    if (nmo < 1) return;

    // => (Initialize) Metric and Error Vector <= //

    double metric = 0.0;
    SharedMatrix delta_pppp = std::make_shared<Matrix>("delta_pppp", nmo, 1);
    for (size_t p = 0; p < nmo; p++) {
        delta_pppp->set(p, 0, pppq->get(p, p));
        metric += pppq->get(p, p);
    }
    double old_metric = metric;

    // => Iteration Print <= //

    outfile->Printf("    Iteration %24s %14s\n", "Metric", "Residual");
    outfile->Printf("    @ER   %4d %24.16E %14s\n", 0, metric, "-");

    // => DIIS Setup <= //
    
    size_t max_vecs = Process::environment.options.get_int("DIIS_MAX_VECS");
    DIISManager diis(max_vecs, "ER DIIS", DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);

    // Total rotation angle
    auto Kappa = std::make_shared<Matrix>("Kappa", nmo, nmo);
    Kappa->zero();

    // ==> Master Loop <== //
    
    for (int iter = 1; iter <= maxiter_; iter++) {
        // > Compute gradient for rotations (gradient ascent) < //

        // Source: https://gqcg-res.github.io/knowdes/edmiston-ruedenberg-localization.html
        auto dKappa = pppq->transpose();
        dKappa->subtract(pppq);
        dKappa->scale(-4.0);

        auto Kappa_old = Kappa->clone();
        Kappa->add(dKappa);

        // DIIS extrapolation (of dKappa, past iteration 100)
        if (iter > 100) {
            diis.set_error_vector_size(dKappa.get());
            diis.set_vector_size(Kappa.get());

            diis.add_entry(dKappa.get(), Kappa.get());
            diis.extrapolate(Kappa.get());

            // Compute dKappa from Kappa - Kappa_old
            dKappa = Kappa->clone();
            dKappa->subtract(Kappa_old);
        }

        // Directly compute unitary transformation matrix
        auto dU = dKappa->clone();
        dU->scale(-0.25); // negative for gradient "ascent"
        dU->expm();

        // Form new U and L matrices
        U_ = linalg::doublet(U_, dU, false, false);
        L_ = linalg::doublet(L_, dU, false, false);

        // Recompute two electron integrals from rotated orbitals

        // Recomputation of x_mo
        x_mo = linalg::doublet(x_mo, dU, false, false);

        // Recomputation of S^{I}_{p}
#pragma omp parallel for collapse(2)
        for (size_t I = 0; I < nthc; ++I) {
            for (size_t p = 0; p < nmo; ++p) {
                double val = x_mo->get(I, p);
                SI_p->set(I, p, val * val);
            } // end p
        } // end I

        // Recomputation of A^{J}_{p}
        AJ_p = linalg::doublet(Z_IJ, SI_p, false, false);

        // Recomputation of (pp|pq)
#pragma omp parallel for collapse(2)
        for (size_t p = 0; p < nmo; ++p) {
            for (size_t q = 0; q < nmo; ++q) {
                double val = 0.0;
                for (size_t J = 0; J < nthc; ++J) {
                    val += AJ_p->get(J, p) * x_mo->get(J, p) * x_mo->get(J, q);
                } // end J
                pppq->set(p, q, val);
            } // end q
        } // end p

        // => Metric and Convergence Checks <= //
        
        metric = 0.0;
        for (size_t p = 0; p < nmo; p++) {
            delta_pppp->set(p, 0, pppq->get(p, p) - delta_pppp->get(p, 0));
            metric += pppq->get(p, p);
        }

        // Check for convergence
        double conv = std::fabs(metric - old_metric) / std::fabs(old_metric);
        old_metric = metric;

        // => Iteration Print <= //
        
        outfile->Printf("    @ER   %4d %24.16E %14.6E\n", iter, metric, conv);

        // => Convergence Check <= //

        if (conv < convergence_) {
            converged_ = true;
            break;
        }
    } // end iter

    outfile->Printf("\n");
    if (converged_) {
        outfile->Printf("    ER   Localizer converged.\n\n");
    } else {
        outfile->Printf("    ER   Localizer failed.\n\n");
    }
}

}  // Namespace psi
