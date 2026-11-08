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
#include <cmath>
#include <set>

namespace psi {
namespace dlpno {
namespace response {
namespace {
std::vector<int> labels(const Tensor& a) {
    std::vector<int> ids;
    for (double v : a) {
        if (v < 0. || v != std::floor(v)) throw std::invalid_argument("Invalid native domain label.");
        ids.push_back(int(v));
    }
    return ids;
}
std::pair<Mixed, Mixed> local_parent(const Mixed& N, const Mixed& S, const Tensor& parent) {
    auto X = N.context->constant("crude_parent", parent), Y = mat(N, X);
    auto norm = polar(product("ma,mn,nb->ab", {Y, S, Y}));
    auto coefficients = mat(X, norm);
    return {mat(N, coefficients), coefficients};
}
Mixed rectangular(const Mixed& K, const Mixed& F, const Mixed& G, const Mixed& o) {
    auto [a, U] = eigh(F[0]);
    auto [b, V] = eigh(G[0]);
    int n = a.size(), m = b.size();
    std::array<Tensor, 4> ts;
    std::array<bool, 4> ready{};
    auto apply = [&](int k, const Tensor& t) { return o[k].scalar() * t - mat(F[k], t) - mat(t, G[k]); };
    auto solve = [&](const Tensor& rhs) {
        auto t = mat(tr(U), mat(rhs, V));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                double gap = o[0].scalar() - a.ptr()[i] - b.ptr()[j];
                if (std::abs(gap) <= 1.e-12) throw std::runtime_error("Singular dipole pair denominator.");
                t.mutable_ptr()[i * m + j] /= gap;
            }
        return mat(U, mat(t, tr(V)));
    };
    std::function<Tensor(int)> component = [&](int k) {
        if (ready[k]) return ts[k];
        auto rhs = K[k];
        for (int i = 1; i < 4; ++i)
            if ((i & k) == i) rhs = rhs - apply(i, component(i ^ k));
        ready[k] = true;
        return ts[k] = solve(rhs);
    };
    return K.context->evaluate("rectangular_mp2", {K.id, F.id, G.id, o.id}, component);
}
Mixed power(const Mixed& a, double exponent) {
    return a.context->evaluate("power:" + std::to_string(exponent), {a.id}, [&](int k) {
        double x = a[0].scalar();
        if (x <= 0.) throw std::runtime_error("Expected positive squared centroid distance.");
        double out = std::pow(x, exponent);
        if (k) out = exponent * std::pow(x, exponent - 1.) * a[k].scalar();
        if (k == 3) out += exponent * (exponent - 1.) * std::pow(x, exponent - 2.) * a[1].scalar() * a[2].scalar();
        return Tensor({}, out);
    });
}
}  // namespace
std::pair<Mixed, Mixed> screened_energy(const Workspace& w, const Molecular& m, const std::vector<Mixed>& dipoles) {
    auto C = [](const std::string& eq, std::initializer_list<Mixed> xs) { return product(eq, xs); };
    auto foo = C("mi,mn,nj->ij", {m.l, m.f, m.l});
    auto sc = m.s.context->constant("zero_energy", Tensor({}, 0.)), dip = sc;
    auto occ = [&](int i, int j) { return at(foo, i, i) + at(foo, j, j); };
    std::map<std::vector<int>, Mixed> pao_cache;
    auto pao = [&](const std::vector<int>& ids) {
        auto it = pao_cache.find(ids);
        if (it != pao_cache.end()) return it->second;
        return pao_cache.emplace(ids, paos(m, ids)).first->second;
    };
    for (const auto& pair : w.screened_domains.at("SCMP2_PAIRS")) {
        int i = pair.at(0), j = pair.at(1);
        if (i >= j) throw std::runtime_error("Expected unique excluded SCMP2 pairs.");
        const auto& data = w.screened_state.at({i, j});
        auto ids = labels(data.at("PAOS")), aux = labels(data.at("AUX"));
        auto N = pao(ids);
        std::vector<Mixed> panels;
        for (int label : {i, j}) {
            std::set<int> atoms;
            for (int q : aux) atoms.insert(w.auxiliary_centers.at(q));
            std::vector<Mixed> blocks;
            std::vector<std::vector<int>> positions;
            std::vector<size_t> deps;
            for (int atom : atoms) {
                std::vector<int> qs, rows;
                for (size_t q = 0; q < aux.size(); ++q)
                    if (w.auxiliary_centers.at(aux[q]) == atom) {
                        rows.push_back(q);
                        qs.push_back(aux[q]);
                    }
                auto A = w.screened_domains.at("AO_OCC").at(atom), B = w.screened_domains.at("AO_VIR").at(atom);
                auto rhs = C("mn,n->m", {m.s.select({A, range(m.s[0].shape[0])}), col(m.l, label)});
                auto fitted = metric_solve(m.s.select({A, A}), rhs);
                blocks.push_back(C("m,Qmn,na->Qa", {fitted, m.q.select({qs, A, B}), N.select({B, range(ids.size())})}));
                positions.push_back(rows);
                deps.push_back(blocks.back().id);
            }
            auto panel = m.s.context->evaluate(
                "screened_OV:" + std::to_string(i) + ":" + std::to_string(j) + ":" + std::to_string(label), deps,
                [&](int k) {
                    Tensor out({int(aux.size()), int(ids.size())});
                    for (size_t a = 0; a < blocks.size(); ++a)
                        for (size_t q = 0; q < positions[a].size(); ++q)
                            std::copy_n(blocks[a][k].ptr() + q * ids.size(), ids.size(),
                                        out.mutable_ptr() + positions[a][q] * ids.size());
                    return out;
                });
            panels.push_back(panel);
        }
        close(panels[0][0], data.at("QI"), 2.e-10, "Screened SCMP2 QI");
        close(panels[1][0], data.at("QJ"), 2.e-10, "Screened SCMP2 QJ");
        auto K0 = C("Qa,Qb->ab", {panels[0], metric_solve(m.j.select({aux, aux}), panels[1])});
        auto [Y, X] = local_parent(N, m.s, data.at("PARENT"));
        auto F = C("ma,mn,nb->ab", {Y, m.f, Y}), K = C("ma,mn,nb->ab", {X, K0, X});
        auto t = semicanonical(K, F, occ(i, j));
        sc = sc + 2. * C("ab,ab->", {K, spin(t)});
    }
    std::map<int, std::array<Mixed, 3>> orbital;
    auto dipole_orbital = [&](int i) {
        auto it = orbital.find(i);
        if (it != orbital.end()) return it->second;
        auto N = pao(w.screened_domains.at("DIPOLE_PAO").at(i));
        auto Y = local_parent(N, m.s, w.screened_state.at({i, i}).at("PARENT")).first, L = col(m.l, i);
        std::vector<Mixed> center, moment;
        for (auto& op : dipoles) {
            center.push_back(C("m,mn,n->", {L, op, L}));
            moment.push_back(C("m,mn,na->a", {L, op, Y}));
        }
        return orbital.emplace(i, std::array<Mixed, 3>{stack(center), stack(moment), C("ma,mn,nb->ab", {Y, m.f, Y})})
            .first->second;
    };
    for (const auto& pair : w.screened_domains.at("DIPOLE_PAIRS")) {
        int i = pair.at(0), j = pair.at(1);
        if (i >= j) throw std::runtime_error("Expected unique excluded dipole pairs.");
        auto a = dipole_orbital(i), b = dipole_orbital(j);
        auto R = a[0] - b[0], r2 = C("k,k->", {R, R});
        auto A = C(",kl->kl", {power(r2, -1.5), m.s.context->constant("three_identity", eye(3))}) -
                 3. * C(",k,l->kl", {power(r2, -2.5), R, R});
        auto K = C("ka,kl,lb->ab", {a[1], A, b[1]});
        dip = dip + 4. * C("ab,ab->", {K, rectangular(K, a[2], b[2], occ(i, j))});
    }
    return {sc, dip};
}
}  // namespace response
}  // namespace dlpno
}  // namespace psi
