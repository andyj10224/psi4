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

#include "native_response.h"
#include <set>
#include <cmath>

namespace psi {
namespace dlpno {
namespace response {
bool Fitting::keep_panel(const Mixed& x) const {
    const size_t bytes = 4 * sizeof(double) * x[0].size();
    const size_t limit = workspace.cache->limit() / 4;
    if (panel_bytes > limit || bytes > limit - panel_bytes) return false;
    panel_bytes += bytes;
    return true;
}
Mixed Fitting::overlap(Pair p, Pair q) const {
    p = key(p);
    q = key(q);
    const auto id = std::make_pair(p, q);
    auto found = overlap_panels.find(id);
    if (found != overlap_panels.end()) {
        ++panel_counts[5];
        return found->second;
    }
    ++panel_counts[4];
    auto s = product("ma,mn,nb->ab", {frames.at(p), molecular.s, frames.at(q)});
    s.context->retain(s);
    if (keep_panel(s)) overlap_panels.emplace(id, s);
    return s;
}
Mixed Fitting::pao(const std::vector<int>& ids) const {
    auto it = pao_cache.find(ids);
    if (it != pao_cache.end()) return it->second;
    return pao_cache.emplace(ids, paos(molecular, ids)).first->second;
}
Mixed Fitting::refit(int atom, const std::vector<int>& occ) const {
    const auto id = std::make_pair(atom, occ);
    auto found = refit_panels.find(id);
    if (found != refit_panels.end()) {
        ++panel_counts[3];
        return found->second;
    }
    ++panel_counts[2];
    auto& rows = workspace.domains.at("AO_OCC").at(atom);
    int n = molecular.s[0].shape[0];
    auto rhs = mat(molecular.s.select({rows, range(n)}), molecular.l.select({range(n), occ}));
    auto out = metric_solve(molecular.s.select({rows, rows}), rhs);
    Tensor present({int(occ.size())});
    auto& allowed = workspace.domains.at("OCC").at(atom);
    for (size_t i = 0; i < occ.size(); ++i)
        present.mutable_ptr()[i] = std::find(allowed.begin(), allowed.end(), occ[i]) != allowed.end();
    out = product("mi,i->mi", {out, out.context->constant("occupied_mask", present)});
    out.context->retain(out);
    if (keep_panel(out)) refit_panels.emplace(id, out);
    return out;
}
Mixed Fitting::coefficients(const std::vector<int>& ids, const Tensor& parent, const Mixed& frame) const {
    auto X = frame.context->constant("parent", parent), B = mat(pao(ids), X);
    return mat(X, metric_solve(product("ma,mn,nb->ab", {B, molecular.s, B}),
                               product("ma,mn,nb->ab", {B, molecular.s, frame})));
}
Mixed Fitting::raw(const std::string& kind, const std::vector<int>& aux, const std::vector<int>& left,
                   const std::vector<int>& right) const {
    const RawKey id{kind, aux, left, right};
    auto found = raw_panels.find(id);
    if (found != raw_panels.end()) {
        ++panel_counts[1];
        return found->second;
    }
    ++panel_counts[0];
    const auto& c = *molecular.s.context;
    std::set<int> atoms;
    for (int q : aux) atoms.insert(workspace.auxiliary_centers.at(q));
    std::vector<Mixed> blocks;
    std::vector<std::vector<int>> destinations;
    for (int atom : atoms) {
        std::vector<int> rows, actual;
        for (size_t q = 0; q < aux.size(); ++q)
            if (workspace.auxiliary_centers.at(aux[q]) == atom) {
                rows.push_back(q);
                actual.push_back(aux[q]);
            }
        const auto &A = workspace.domains.at("AO_OCC").at(atom), &B = workspace.domains.at("AO_VIR").at(atom);
        Mixed block;
        Tensor mask({int(left.size()), int(right.size())}, 1.);
        if (kind == "oo")
            block =
                product("ma,Qmn,nb->Qab", {refit(atom, left), molecular.q.select({actual, A, A}), refit(atom, right)});
        else if (kind == "ov") {
            block = product("ma,Qmn,nb->Qab", {refit(atom, left), molecular.q.select({actual, A, B}),
                                               pao(right).select({B, range(right.size())})});
            const auto& allowed = workspace.domains.at("PAO").at(atom);
            for (size_t a = 0; a < left.size(); ++a)
                for (size_t b = 0; b < right.size(); ++b)
                    mask.mutable_ptr()[a * right.size() + b] =
                        std::find(allowed.begin(), allowed.end(), right[b]) != allowed.end();
        } else if (kind == "vv") {
            block = product("ma,Qmn,nb->Qab",
                            {pao(left).select({B, range(left.size())}), molecular.q.select({actual, B, B}),
                             pao(right).select({B, range(right.size())})});
            std::set<Pair> allowed;
            const auto& labels = workspace.domains.at("PAO_PAIRS").at(atom);
            if (labels.size() % 2) throw std::invalid_argument("Invalid PAO-pair mask.");
            for (size_t i = 0; i < labels.size(); i += 2) {
                allowed.insert({labels[i], labels[i + 1]});
                allowed.insert({labels[i + 1], labels[i]});
            }
            for (size_t a = 0; a < left.size(); ++a)
                for (size_t b = 0; b < right.size(); ++b)
                    mask.mutable_ptr()[a * right.size() + b] = allowed.count({left[a], right[b]});
        } else
            throw std::invalid_argument("Unknown native fitting block.");
        blocks.push_back(product("Qab,ab->Qab", {block, c.constant("fitting_mask", mask)}));
        destinations.push_back(rows);
    }
    std::vector<size_t> deps;
    for (auto& b : blocks) deps.push_back(b.id);
    std::string label = "assemble_raw:" + kind;
    for (int q : aux) label += ":" + std::to_string(q);
    label += "/left";
    for (int i : left) label += ":" + std::to_string(i);
    label += "/right";
    for (int i : right) label += ":" + std::to_string(i);
    auto out = c.evaluate(label, deps, [&](int k) {
        Tensor result({int(aux.size()), int(left.size()), int(right.size())});
        size_t width = left.size() * right.size();
        for (size_t a = 0; a < blocks.size(); ++a)
            for (size_t q = 0; q < destinations[a].size(); ++q)
                std::copy_n(blocks[a][k].ptr() + q * width, width, result.mutable_ptr() + destinations[a][q] * width);
        return result;
    });
    out.context->retain(out);
    if (keep_panel(out)) raw_panels.emplace(id, out);
    return out;
}
Mixed Fitting::metric(const std::vector<int>& ids) const { return molecular.j.select({ids, ids}); }
const std::array<Mixed, 5>& Fitting::projected(Pair p) const {
    p = key(p);
    auto it = projected_cache.find(p);
    if (it != projected_cache.end()) return it->second;
    const auto& data = workspace.pairs.at(p);
    auto X = coefficients(data.paos, data.first_parent, frames.at(p));
    coefficient_cache[p] = X;
    auto oo = raw("oo", data.auxiliary, data.occupied, data.occupied);
    auto ov = product("Qma,ab->Qmb", {raw("ov", data.auxiliary, data.occupied, data.paos), X});
    auto vv = product("ma,Qmn,nb->Qab", {X, raw("vv", data.auxiliary, data.paos, data.paos), X});
    auto m = metric(data.auxiliary), g = symmetric_inverse(m);
    auto filtered = filtered_inverse(m, g);
    for (const auto& block : {X, oo, ov, vv, g, filtered}) block.context->retain(block);
    return projected_cache.emplace(p, std::array<Mixed, 5>{oo, ov, vv, g, filtered}).first->second;
}
std::pair<Mixed, Mixed> Fitting::mixed_integrals(int i, int j, int k) const {
    Pair p = key({i, j}), q = key({k, j});
    projected(p);
    projected(q);
    const auto& a = workspace.pairs.at(p);
    const auto& b = workspace.pairs.at(q);
    auto X = coefficient_cache.at(p), Y = coefficient_cache.at(q);
    auto oo = auxat(raw("oo", a.auxiliary, {i}, {k}), 0, 0);
    // Contract the occupied/auxiliary leg before projecting the two virtual
    // legs. Materializing Qab here repeated two O(Q*v^3) transformations for
    // every ordered mixed vertex, although J only needs an ab matrix.
    auto vv = raw("vv", a.auxiliary, a.paos, b.paos);
    auto ov = mat(auxrow(raw("ov", a.auxiliary, {i}, a.paos), 0), X),
         other = mat(auxrow(raw("ov", a.auxiliary, {k}, b.paos), 0), Y), g = projected(p)[3];
    auto weight = product("Q,QR->R", {oo, g});
    auto contracted = product("R,Rmn->mn", {weight, vv});
    return {product("ma,mn,nb->ab", {X, contracted, Y}), product("Qa,QR,Rb->ab", {ov, g, other})};
}
namespace {
PairMixed residual(const PNOResult& state, const Mixed& foo, const Mixed& s, const std::map<Pair, PairData>& pairs) {
    PairMixed result;
    for (auto& [p, in] : state.final) {
        auto t = in.amplitudes, F = in.fock;
        auto occ = at(foo, p.first, p.first) + at(foo, p.second, p.second);
        auto r = in.exchange + mat(F, t) + mat(t, F) - product(",ab->ab", {occ, t});
        for (auto edge : pairs.at(p).edges) {
            Pair q = key({edge[0], edge[1]});
            auto tq = state.final.at(q).amplitudes;
            if (edge[0] > edge[1]) tq = tq.transpose();
            auto S = product("ma,mn,nb->ab", {state.first.at(p).selection.frame, s, state.first.at(q).selection.frame});
            r = r - product(",ac,cd,bd->ab", {at(foo, edge[2], edge[3]), S, tq, S});
        }
        if (p.first == p.second) r = .5 * (r + r.transpose());
        result[p] = r;
    }
    return result;
}
double maximum(const Tensor& x) {
    double e = 0.;
    for (double a : *x.data) {
        if (!std::isfinite(a)) throw std::runtime_error("Nonfinite LMP2 residual.");
        e = std::max(e, std::abs(a));
    }
    return e;
}
}  // namespace
PNOResult Workspace::pno(const Molecular& m, const FirstSolver& first_solver, const MixedSolver& mixed_solver) {
    PNOResult result;
    auto foo = product("mi,mn,nj->ij", {m.l, m.f, m.l});
    // The first OV fit uses its own captured domain lists, rather than assuming
    // that the subsequent local CC fit used identical occupied/PAO masks.
    Workspace first_fit(0);
    first_fit.domains = first_domains;
    first_fit.auxiliary_centers = auxiliary_centers;
    std::vector<Pair> keys;
    for (auto& [p, data] : pairs) keys.push_back(p);
    std::vector<PNOResult> rows(keys.size());
    parallel_rows(keys.size(), [&](size_t index) {
        const auto p = keys[index];
        const auto& data = pairs.at(p);
        auto& result = rows[index];
        Fitting fit(first_fit, m, {});
        auto N = fit.pao(data.paos);
        result.normalized[p] = N;
        auto qi = auxrow(fit.raw("ov", data.first_auxiliary, {p.first}, data.paos), 0),
             qj = auxrow(fit.raw("ov", data.first_auxiliary, {p.second}, data.paos), 0),
             J = fit.metric(data.first_auxiliary);
        close(qi[0], data.qi, 2.e-10, "Native first-PNO QI");
        close(qj[0], data.qj, 2.e-10, "Native first-PNO QJ");
        auto K = product("Qa,Qb->ab", {qi, metric_solve(J, qj)});
        auto [parent, coeff] = moving_parent(N, m.s, data.first_parent);
        auto F = product("ma,mn,nb->ab", {parent, m.f, parent}), exchange = product("ma,mn,nb->ab", {coeff, K, coeff});
        auto t = semicanonical(exchange, F, at(foo, p.first, p.first) + at(foo, p.second, p.second));
        auto selection = pno_selection(parent, m.s, t, data.first_frame, p.first == p.second);
        close(t[0], data.first_amplitude, 2.e-10, "Native first-PNO amplitude");
        close(selection.occupations, data.occupations, 2.e-10, "Native first-PNO occupations");
        result.first[p] = {parent, exchange, F, t, selection};
        auto Y = selection.frame, C = product("ma,mn,nb->ab", {parent, m.s, Y});
        result.initial[p] = {
            exchange,     t, F, product("m,mn,na->a", {col(m.l, p.first), m.f, parent}), at(foo, p.first, p.first),
            Y[0].shape[1]};
        result.final[p] = {product("ma,mn,nb->ab", {C, exchange, C}),
                           {},
                           product("ma,mn,nb->ab", {Y, m.f, Y}),
                           product("m,mn,na->a", {col(m.l, p.first), m.f, Y}),
                           at(foo, p.first, p.first),
                           data.final_frame.shape[1]};
    });
    for (auto& row : rows) {
        result.first.merge(row.first);
        result.normalized.merge(row.normalized);
        result.initial.merge(row.initial);
        result.final.merge(row.final);
    }

    std::array<PairTensor, 2> tangents;
    for (int k = 1; k <= 2; ++k) {
        int coordinate = k == 1 ? m.s.context->u : m.s.context->v;
        auto found = lmp2_first.find(coordinate);
        if (found != lmp2_first.end()) {
            tangents[k - 1] = found->second;
            ++lmp2_hits;
            continue;
        }
        PairTensor dy, dk;
        for (auto& [p, out] : result.first) {
            dy[p] = out.selection.frame[k];
            dk[p] = result.final.at(p).exchange[k];
        }
        auto solved = first_solver(dy, dk, m.s[k], m.f[k], foo[k]);
        if (solved.size() != pairs.size()) throw std::runtime_error("First LMP2 pair graph mismatch.");
        for (auto& [p, data] : pairs)
            if (solved.at(p).shape != data.lmp2_amplitude.shape)
                throw std::runtime_error("First LMP2 dimensions mismatch.");
        tangents[k - 1] = solved;
        lmp2_first.emplace(coordinate, std::move(solved));
        ++lmp2_solves;
    }
    for (auto& [p, data] : pairs)
        result.final.at(p).amplitudes = m.s.context->input(
            "LMP2:" + std::to_string(p.first) + ":" + std::to_string(p.second),
            {data.lmp2_amplitude, tangents[0].at(p), tangents[1].at(p), Tensor(data.lmp2_amplitude.shape)});
    auto affine = residual(result, foo, m.s, pairs);
    for (auto& [p, r] : affine) {
        for (int k = 0; k < 3; ++k) result.residual_errors[k] = std::max(result.residual_errors[k], maximum(r[k]));
        result.source[p] = r[3];
    }
    for (int k = 0; k < 3; ++k)
        if (result.residual_errors[k] > 2.e-9)
            throw std::runtime_error("Screened LMP2 value/first moving-frame residual failed.");
    auto mixed = mixed_solver(result.source);
    if (mixed.size() != pairs.size()) throw std::runtime_error("Mixed LMP2 pair graph mismatch.");
    Energies affine_final = result.final;
    for (auto& [p, data] : pairs) {
        auto t = result.final.at(p).amplitudes;
        auto values = t.x;
        values[3] = mixed.at(p);
        result.final.at(p).amplitudes =
            m.s.context->input("LMP2_complete:" + std::to_string(p.first) + ":" + std::to_string(p.second), values);
    }
    auto complete = residual(result, foo, m.s, pairs);
    for (auto& [p, r] : complete) result.residual_errors[3] = std::max(result.residual_errors[3], maximum(r[3]));
    if (result.residual_errors[3] > 2.e-9) throw std::runtime_error("Mixed LMP2 moving-frame residual failed.");
    for (auto& [p, data] : pairs)
        result.selection[p] = pno_selection(result.first.at(p).selection.frame, m.s, result.final.at(p).amplitudes,
                                            data.final_frame, p.first == p.second);
    result.initial_loss = pno_energy(result.initial, {});
    result.final_loss = pno_energy({}, result.final);
    result.loss = result.initial_loss + result.final_loss;
    auto affine_loss = pno_energy(result.initial, affine_final);
    result.stationary_loss = affine_loss[3].scalar();
    if (std::abs(affine_loss[0].scalar() - pno_reference) > 1.e-11 + 1.e-9 * std::abs(pno_reference))
        throw std::runtime_error("Native PNO-loss reference energy mismatch.");
    if (pno_adjoint.size() != pairs.size()) throw std::runtime_error("PNO loss adjoint graph mismatch.");
    for (auto& [p, z] : pno_adjoint) result.stationary_loss += dot(z, result.source.at(p));
    if (std::abs(result.loss[3].scalar() - result.stationary_loss) > 2.e-10)
        throw std::runtime_error("Direct and stationary PNO-loss curvatures disagree.");
    return result;
}
}  // namespace response
}  // namespace dlpno
}  // namespace psi
