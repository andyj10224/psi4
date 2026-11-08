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
    auto s = congruence(frames.at(p), molecular.s, frames.at(q));
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
    return mat(X, metric_solve(congruence(B, molecular.s, B), congruence(B, molecular.s, frame)));
}
void Fitting::prepare_raw(const std::vector<std::string>& kinds) const {
    for (const auto& kind : kinds) {
        if (!prepared_raw.insert(kind).second) continue;
        const int category = kind == "oo" ? 0 : kind == "ov" ? 1 : 2;
        for (size_t atom = 0; atom < workspace.domains.at("OCC").size(); ++atom) {
            std::vector<int> aux, left, right;
            for (size_t q = 0; q < workspace.auxiliary_centers.size(); ++q)
                if (workspace.auxiliary_centers[q] == int(atom)) aux.push_back(q);
            if (aux.empty()) continue;
            if (kind == "vv") {
                const auto& labels = workspace.domains.at("PAO_PAIRS").at(atom);
                std::set<int> unique(labels.begin(), labels.end());
                left.assign(unique.begin(), unique.end());
                right = left;
            } else {
                left = workspace.domains.at("OCC").at(atom);
                right = workspace.domains.at(kind == "oo" ? "OCC" : "PAO").at(atom);
            }
            const size_t bytes = 4 * sizeof(double) * aux.size() * left.size() * right.size();
            const size_t limit = workspace.cache->limit() / 4;
            if (panel_bytes > limit || bytes > limit - panel_bytes) continue;
            // Reserve the panel before local refits consume their allowance.
            panel_bytes += bytes;
            auto block = raw_direct(kind, aux, left, right);
            block.context->retain(block);
            atomic_panels.emplace(std::make_pair(kind, int(atom)), RawPanel{block, aux, left, right});
            atomic_bytes[category] += bytes;
        }
    }
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
    auto out = raw_direct(kind, aux, left, right);
    out.context->retain(out);
    if (keep_panel(out)) raw_panels.emplace(id, out);
    return out;
}
Mixed Fitting::raw_direct(const std::string& kind, const std::vector<int>& aux, const std::vector<int>& left,
                          const std::vector<int>& right) const {
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
        auto resident = atomic_panels.find({kind, atom});
        if (resident != atomic_panels.end()) {
            const auto& panel = resident->second;
            auto indices = [](const auto& all, const auto& requested) {
                std::vector<int> ids;
                for (int x : requested) {
                    auto found = std::find(all.begin(), all.end(), x);
                    ids.push_back(found == all.end() ? -1 : int(found - all.begin()));
                }
                return ids;
            };
            const auto qs = indices(panel.auxiliary, actual), ls = indices(panel.left, left),
                       rs = indices(panel.right, right);
            std::string label = "raw_slice:" + kind;
            for (const auto& ids : {qs, ls, rs}) {
                label += "/";
                for (int i : ids) label += std::to_string(i) + ",";
            }
            blocks.push_back(c.evaluate(label, {panel.values.id}, [&](int component) {
                Tensor out({int(qs.size()), int(ls.size()), int(rs.size())});
                const size_t nl = panel.left.size(), nr = panel.right.size();
                for (size_t q = 0; q < qs.size(); ++q)
                    for (size_t a = 0; a < ls.size(); ++a)
                        for (size_t b = 0; b < rs.size(); ++b)
                            if (qs[q] >= 0 && ls[a] >= 0 && rs[b] >= 0)
                                out.mutable_ptr()[(q * ls.size() + a) * rs.size() + b] =
                                    panel.values[component].ptr()[(qs[q] * nl + ls[a]) * nr + rs[b]];
                return out;
            }));
            destinations.push_back(rows);
            continue;
        }
        const auto &A = workspace.domains.at("AO_OCC").at(atom), &B = workspace.domains.at("AO_VIR").at(atom);
        Mixed block;
        Tensor mask({int(left.size()), int(right.size())}, 1.);
        if (kind == "oo")
            block = virtual_transform(molecular.q.select({actual, A, A}), refit(atom, left), refit(atom, right));
        else if (kind == "ov") {
            block = virtual_transform(molecular.q.select({actual, A, B}), refit(atom, left),
                                      pao(right).select({B, range(right.size())}));
            const auto& allowed = workspace.domains.at("PAO").at(atom);
            for (size_t a = 0; a < left.size(); ++a)
                for (size_t b = 0; b < right.size(); ++b)
                    mask.mutable_ptr()[a * right.size() + b] =
                        std::find(allowed.begin(), allowed.end(), right[b]) != allowed.end();
        } else if (kind == "vv") {
            block = virtual_transform(molecular.q.select({actual, B, B}), pao(left).select({B, range(left.size())}),
                                      pao(right).select({B, range(right.size())}));
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
    return out;
}
Mixed Fitting::metric(const std::vector<int>& ids) const { return molecular.j.select({ids, ids}); }
Mixed Fitting::weighted(Pair p, int panel, bool filtered) const {
    p = key(p);
    if (panel < 0 || panel > 2) throw std::invalid_argument("Invalid local DF panel.");
    const auto id = std::make_tuple(p, panel, filtered);
    auto found = weighted_panels.find(id);
    if (found != weighted_panels.end()) return found->second;
    const auto& blocks = projected(p);
    auto out = auxiliary_transform(blocks[filtered ? 4 : 3], blocks[panel]);
    out.context->retain(out);
    if (keep_panel(out)) {
        weighted_panels.emplace(id, out);
        weighted_bytes += 4 * sizeof(double) * out[0].size();
    }
    return out;
}
const std::array<Mixed, 5>& Fitting::projected(Pair p) const {
    p = key(p);
    auto it = projected_cache.find(p);
    if (it != projected_cache.end()) return it->second;
    const auto& data = workspace.pairs.at(p);
    auto X = coefficients(data.paos, data.first_parent, frames.at(p));
    coefficient_cache[p] = X;
    auto oo = raw("oo", data.auxiliary, data.occupied, data.occupied);
    const int nq = data.auxiliary.size(), no = data.occupied.size(), nv = data.paos.size(), np = X[0].shape[1];
    auto ov =
        doublet(raw("ov", data.auxiliary, data.occupied, data.paos).reshape({nq * no, nv}), X).reshape({nq, no, np});
    auto vv = virtual_transform(raw("vv", data.auxiliary, data.paos, data.paos), X, X);
    auto m = metric(data.auxiliary), g = symmetric_inverse(m);
    auto filtered = filtered_inverse(m, g);
    for (const auto& block : {X, oo, ov, vv, g, filtered}) block.context->retain(block);
    return projected_cache.emplace(p, std::array<Mixed, 5>{oo, ov, vv, g, filtered}).first->second;
}
std::pair<Mixed, Mixed> Fitting::mixed_integrals(int i, int j, int k) const {
    const auto request = std::make_tuple(i, j, k);
    auto found = mixed_vertices.find(request);
    if (found != mixed_vertices.end()) return found->second;
    const Pair p = key({i, j});
    const auto& data = workspace.pairs.at(p);
    const auto& panels = projected(p);
    const auto X = coefficient_cache.at(p);
    const int nq = data.auxiliary.size(), no = data.occupied.size(), na = data.paos.size(), np = X[0].shape[1];
    std::set<int> union_paos;
    for (int l : data.occupied) {
        const auto& ids = workspace.pairs.at(key({l, j})).paos;
        union_paos.insert(ids.begin(), ids.end());
    }
    const std::vector<int> extended(union_paos.begin(), union_paos.end());
    const int ne = extended.size();
    // Match compute_pno_integrals(): one target auxiliary fit, an extended
    // PAO domain, and only the actual final source-pair transformation. No
    // S_pq S_qr replacement of either nonprojected vertex is permitted.
    auto q_io = auxrow(panels[0], pos(data.occupied, i));
    auto q_iv = auxrow(panels[1], pos(data.occupied, i));
    auto fitted_io = doublet(panels[3], q_io);
    auto fitted_iv = doublet(panels[3], q_iv);
    auto q_vv = raw("vv", data.auxiliary, data.paos, extended);
    // Count multiplications for both exact orders, including each actual
    // source PAO list/rank and the cheaper order of its final two GEMMs.
    double aux_cost = double(nq) * no * na * ne;
    double virtual_cost = double(nq) * na * np * ne + double(nq) * no * np * ne;
    for (int l : data.occupied) {
        const Pair q = key({l, j});
        const int nb = workspace.pairs.at(q).paos.size(), nc = frames.at(q)[0].shape[1];
        aux_cost += std::min(double(na) * np * nb + double(np) * nb * nc, double(na) * nb * nc + double(na) * np * nc);
        virtual_cost += double(np) * nb * nc;
    }
    const bool auxiliary_first = aux_cost <= virtual_cost;
    Mixed coulomb;
    if (auxiliary_first) {
        coulomb = doublet(fitted_io, q_vv.reshape({nq, na * ne}), true);
    } else {
        auto partial =
            doublet(X, q_vv.permute({1, 0, 2}).reshape({na, nq * ne}), true).reshape({np, nq, ne}).permute({1, 0, 2});
        coulomb = doublet(fitted_io, partial.reshape({nq, np * ne}), true);
    }
    auto q_ov = raw("ov", data.auxiliary, data.occupied, extended);
    auto exchange = doublet(fitted_iv, q_ov.reshape({nq, no * ne}), true).reshape({np, no, ne});
    for (int l : data.occupied) {
        const Pair q = key({l, j});
        projected(q);
        const auto& ids = workspace.pairs.at(q).paos;
        std::vector<int> columns;
        for (int label : ids) columns.push_back(pos(extended, label));
        const int index = pos(data.occupied, l);
        auto J = row(coulomb, index).reshape({auxiliary_first ? na : np, ne});
        J = J.select({range(J[0].shape[0]), columns});
        const auto& Y = coefficient_cache.at(q);
        J = auxiliary_first ? congruence(X, J, Y) : doublet(J, Y);
        auto K = exchange.select({range(np), {index}, columns}, {1});
        K = doublet(K, Y);
        J.context->retain(J);
        K.context->retain(K);
        mixed_vertices.emplace(std::make_tuple(i, j, l), std::make_pair(J, K));
    }
    return mixed_vertices.at(request);
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
    for (double a : x) {
        if (!std::isfinite(a)) throw std::runtime_error("Nonfinite LMP2 residual.");
        e = std::max(e, std::abs(a));
    }
    return e;
}
}  // namespace
PNOResult Workspace::pno(const Molecular& m, const FirstSolver& first_solver, const MixedSolver& mixed_solver,
                         const Fitting* common) {
    PNOResult result;
    auto foo = product("mi,mn,nj->ij", {m.l, m.f, m.l});
    // The first OV fit uses its own captured domain lists, rather than assuming
    // that the subsequent local CC fit used identical occupied/PAO masks.
    Workspace first_fit(0);
    first_fit.domains = first_domains;
    first_fit.auxiliary_centers = auxiliary_centers;
    first_fit.cache = cache;
    Fitting own_first(first_fit, m, {});
    bool same_domains = common != nullptr;
    // Only the OV panel is shared here. Its dependencies are these four
    // maps; the VV-only PAO_PAIRS mask is not part of a first-PNO snapshot.
    for (const auto& name : {"AO_OCC", "AO_VIR", "OCC", "PAO"})
        same_domains &= first_domains.at(name) == domains.at(name);
    const Fitting& first = same_domains ? *common : own_first;
    first.prepare_raw({"ov"});
    std::vector<Pair> keys;
    for (auto& [p, data] : pairs) keys.push_back(p);
    std::vector<PNOResult> rows(keys.size());
    parallel_rows(keys.size(), [&](size_t index) {
        const auto p = keys[index];
        const auto& data = pairs.at(p);
        auto& result = rows[index];
        Fitting fit(first);
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
