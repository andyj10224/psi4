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
#include <limits>

namespace psi {
namespace dlpno {
namespace response {
namespace {
Mixed derived(const std::string& name, const std::vector<Mixed>& inputs,
              const std::function<std::array<Tensor, 4>()>& build) {
    std::vector<size_t> ids;
    for (auto& x : inputs) ids.push_back(x.id);
    std::array<Tensor, 4> values;
    bool ready = false;
    return inputs[0].context->evaluate(name, ids, [&](int k) {
        if (!ready) {
            values = build();
            ready = true;
        }
        return values[k];
    });
}
Tensor rotate(const Tensor& u, const Tensor& x) { return mat(tr(u), mat(x, u)); }
Tensor back(const Tensor& u, const Tensor& x) { return mat(u, mat(x, tr(u))); }
Tensor diagonal(const Tensor& x) {
    Tensor out({int(x.size()), int(x.size())});
    for (size_t i = 0; i < x.size(); ++i) out.mutable_ptr()[i * x.size() + i] = x.ptr()[i];
    return out;
}
Tensor had(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) throw std::invalid_argument("Native Hadamard shapes disagree.");
    Tensor c(a.shape);
    for (size_t i = 0; i < a.size(); ++i) c.mutable_ptr()[i] = a.ptr()[i] * b.ptr()[i];
    return c;
}
}  // namespace
Mixed symmetric_inverse(const Mixed& m) {
    return derived("symmetric_inverse", {m}, [&]() {
        for (auto& x : m.x) close(x, tr(x), 1.e-12, "Symmetric fitting metric", 1.e-12);
        auto [eig, u] = eigh(sym(m[0]));
        int n = eig.size();
        Tensor g({n}), scale({n, n});
        for (int i = 0; i < n; ++i) {
            if (eig.ptr()[i] == 0.) throw std::runtime_error("Singular fitting metric.");
            g.mutable_ptr()[i] = 1. / eig.ptr()[i];
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) scale.mutable_ptr()[i * n + j] = g.ptr()[i] * g.ptr()[j];
        auto a = sym(rotate(u, sym(m[1]))), b = sym(rotate(u, sym(m[2]))), c = sym(rotate(u, sym(m[3])));
        auto cross = mat(mat(a, diagonal(g)), b);
        return std::array<Tensor, 4>{sym(back(u, diagonal(g))), sym(back(u, -had(scale, a))),
                                     sym(back(u, -had(scale, b))), sym(back(u, had(scale, cross + tr(cross) - c)))};
    });
}
Mixed filtered_inverse(const Mixed& m, const Mixed& full) {
    return derived("filtered_inverse", {m, full}, [&]() {
        auto [e, u] = eigh(m[0]);
        int n = e.size();
        if (!n) return full.x;
        std::vector<bool> active(n);
        Tensor g({n}), dd({n, n});
        bool all = true;
        double cutoff = 1.e-14 * std::max(std::abs(e.ptr()[0]), std::abs(e.ptr()[n - 1]));
        for (int i = 0; i < n; ++i) {
            active[i] = e.ptr()[i] > 0. && e.ptr()[i] >= cutoff;
            all &= active[i];
            g.mutable_ptr()[i] = active[i] ? 1. / e.ptr()[i] : 0.;
        }
        if (all) return full.x;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                dd.mutable_ptr()[i * n + j] = active[i] != active[j]
                                                  ? (g.ptr()[i] - g.ptr()[j]) / (e.ptr()[i] - e.ptr()[j])
                                                  : -g.ptr()[i] * g.ptr()[j];
        auto a = rotate(u, m[1]), b = rotate(u, m[2]), mixed = had(dd, rotate(u, m[3]));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) {
                    double d = 0.;
                    if (active[i] && active[j] && active[k]) d = g.ptr()[i] * g.ptr()[j] * g.ptr()[k];
                    if (active[i] != active[j])
                        d = (dd.ptr()[i * n + k] - dd.ptr()[k * n + j]) / (e.ptr()[i] - e.ptr()[j]);
                    else if (active[i] != active[k])
                        d = (dd.ptr()[j * n + i] - dd.ptr()[j * n + k]) / (e.ptr()[i] - e.ptr()[k]);
                    mixed.mutable_ptr()[i * n + j] +=
                        d * (a.ptr()[i * n + k] * b.ptr()[k * n + j] + b.ptr()[i * n + k] * a.ptr()[k * n + j]);
                }
        return std::array<Tensor, 4>{back(u, diagonal(g)), back(u, had(dd, a)), back(u, had(dd, b)), back(u, mixed)};
    });
}
Mixed polar(const Mixed& g) {
    close(g[0], eye(g[0].shape[0]), 2.e-8, "Polar reference metric");
    auto root = derived("polar_root", {g}, [&]() {
        auto [eig, u] = eigh(g[0]);
        for (double& e : *eig.data) {
            if (e <= 0.) throw std::runtime_error("Nonpositive polar metric.");
            e = std::sqrt(e);
        }
        auto sylvester = [&](const Tensor& rhs) {
            auto x = rotate(u, rhs);
            int n = eig.size();
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) x.mutable_ptr()[i * n + j] /= eig.ptr()[i] + eig.ptr()[j];
            return back(u, x);
        };
        auto a = sylvester(g[1]), b = sylvester(g[2]);
        return std::array<Tensor, 4>{back(u, diagonal(eig)), a, b, sylvester(g[3] - mat(a, b) - mat(b, a))};
    });
    return metric_solve(root, root.context->constant("identity", eye(g[0].shape[0])));
}
Mixed projector(const Mixed& d, int kept) {
    return derived("projector:" + std::to_string(kept), {d}, [&]() {
        auto [e, u] = eigh(d[0]);
        int n = e.size(), cut = n - kept;
        if (kept <= 0 || kept > n) throw std::invalid_argument("Invalid natural-orbital selection rank.");
        if (kept == n) return std::array<Tensor, 4>{eye(n), Tensor({n, n}), Tensor({n, n}), Tensor({n, n})};
        double scale = 0.;
        for (double x : *e.data) scale = std::max(scale, std::abs(x));
        if (e.ptr()[cut] - e.ptr()[cut - 1] <= 64 * std::numeric_limits<double>::epsilon() * scale)
            throw std::runtime_error("Unresolved retained/discarded natural-orbital boundary.");
        auto first = [&](const Tensor& x) {
            Tensor p({n, n});
            for (int i = 0; i < cut; ++i)
                for (int j = cut; j < n; ++j)
                    p.mutable_ptr()[i * n + j] = p.mutable_ptr()[j * n + i] =
                        x.ptr()[i * n + j] / (e.ptr()[j] - e.ptr()[i]);
            return p;
        };
        auto a = rotate(u, d[1]), b = rotate(u, d[2]), c = rotate(u, d[3]), pu = first(a), pv = first(b);
        auto puv = first(c + mat(a, pv) - mat(pv, a) + mat(b, pu) - mat(pu, b));
        auto diag = mat(pu, pv) + mat(pv, pu);
        Tensor p({n, n});
        for (int i = 0; i < n; ++i) {
            if (i >= cut) p.mutable_ptr()[i * n + i] = 1.;
            for (int j = 0; j < n; ++j)
                if ((i < cut) == (j < cut)) puv.mutable_ptr()[i * n + j] = (i < cut ? 1. : -1.) * diag.ptr()[i * n + j];
        }
        return std::array<Tensor, 4>{back(u, p), back(u, pu), back(u, pv), back(u, puv)};
    });
}
Mixed paos(const Molecular& m, const std::vector<int>& ids) {
    int n = m.s[0].shape[0];
    Tensor basis({n, int(ids.size())});
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= n) throw std::invalid_argument("Invalid PAO domain.");
        basis.mutable_ptr()[size_t(ids[i]) * ids.size() + i] = 1.;
    }
    auto raw = m.s.context->constant("AO_domain", basis) - mat(m.d, m.s.select({range(n), ids}));
    auto norms = product("ma,mn,na->a", {raw, m.s, raw});
    auto factors = derived("PAO_normalization", {norms}, [&]() {
        std::array<Tensor, 4> x{Tensor(norms[0].shape), Tensor(norms[0].shape), Tensor(norms[0].shape),
                                Tensor(norms[0].shape)};
        for (size_t i = 0; i < norms[0].size(); ++i) {
            double q = norms[0].ptr()[i], a = norms[1].ptr()[i], b = norms[2].ptr()[i], c = norms[3].ptr()[i];
            if (q <= 1.e-20) continue;
            x[0].mutable_ptr()[i] = std::pow(q, -.5);
            x[1].mutable_ptr()[i] = -.5 * std::pow(q, -1.5) * a;
            x[2].mutable_ptr()[i] = -.5 * std::pow(q, -1.5) * b;
            x[3].mutable_ptr()[i] = .75 * std::pow(q, -2.5) * a * b - .5 * std::pow(q, -1.5) * c;
        }
        return x;
    });
    return product("ma,a->ma", {raw, factors});
}
std::pair<Mixed, Mixed> moving_parent(const Mixed& pao, const Mixed& s, const Tensor& transform) {
    if (transform.shape.size() != 2 || transform.shape[0] != pao[0].shape[1] || !transform.shape[1])
        throw std::invalid_argument("Invalid native PAO parent transform.");
    int pivots = 0;
    for (int i = 0; i < transform.shape[0]; ++i) {
        bool nonzero = false;
        for (int j = 0; j < transform.shape[1]; ++j)
            nonzero |= transform.ptr()[size_t(i) * transform.shape[1] + j] != 0.;
        pivots += nonzero;
    }
    if (pivots != transform.shape[1]) throw std::invalid_argument("Expected a full-rank selected PAO pivot span.");
    auto fixed = pao.context->constant("parent", transform);
    auto b0 = mat(pao[0], transform), metric = mat(tr(b0), mat(s[0], b0));
    close(metric, eye(metric.shape[0]), 2.e-8, "First-PNO reference parent");
    auto [e, u] = eigh(metric);
    for (double& v : *e.data) v = std::pow(v, -.5);
    auto X = mat(transform, back(u, diagonal(e)));
    auto b = mat(pao[0], X), bu = mat(pao[1], X), bv = mat(pao[2], X), buv = mat(pao[3], X);
    metric = mat(tr(b), mat(s[0], b));
    auto cu = -solve(metric, mat(tr(b), mat(s[0], bu)) + .5 * mat(tr(b), mat(s[1], b)));
    auto cv = -solve(metric, mat(tr(b), mat(s[0], bv)) + .5 * mat(tr(b), mat(s[2], b)));
    auto yu = bu + mat(b, cu), yv = bv + mat(b, cv), bare = buv + mat(bu, cv) + mat(bv, cu);
    auto residual = mat(tr(bare), mat(s[0], b)) + mat(tr(b), mat(s[0], bare)) + mat(tr(yu), mat(s[0], yv)) +
                    mat(tr(yv), mat(s[0], yu)) + mat(tr(yu), mat(s[2], b)) + mat(tr(b), mat(s[2], yu)) +
                    mat(tr(yv), mat(s[1], b)) + mat(tr(b), mat(s[1], yv)) + mat(tr(b), mat(s[3], b));
    auto cuv = -.5 * solve(metric, residual);
    auto parent = pao.context->evaluate("moving_parent", {pao.id, s.id, fixed.id},
                                        [&](int k) { return std::array<Tensor, 4>{b, yu, yv, bare + mat(b, cuv)}[k]; });
    auto coefficients = pao.context->evaluate("parent_coefficients", {pao.id, s.id, fixed.id}, [&](int k) {
        return std::array<Tensor, 4>{X, mat(X, cu), mat(X, cv), mat(X, cuv)}[k];
    });
    return {parent, coefficients};
}
Mixed semicanonical(const Mixed& K, const Mixed& F, const Mixed& occupied) {
    return derived("semicanonical", {K, F, occupied}, [&]() {
        auto [e, u] = eigh(F[0]);
        int n = e.size();
        auto apply = [&](int k, const Tensor& t) { return occupied[k].scalar() * t - mat(F[k], t) - mat(t, F[k]); };
        auto inverse = [&](const Tensor& rhs) {
            auto x = rotate(u, rhs);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    double gap = occupied[0].scalar() - e.ptr()[i] - e.ptr()[j];
                    if (std::abs(gap) <= 1.e-12) throw std::runtime_error("Singular semicanonical MP2 denominator.");
                    x.mutable_ptr()[i * n + j] /= gap;
                }
            return back(u, x);
        };
        auto t = inverse(K[0]), a = inverse(K[1] - apply(1, t)), b = inverse(K[2] - apply(2, t));
        return std::array<Tensor, 4>{t, a, b, inverse(K[3] - apply(3, t) - apply(1, b) - apply(2, a))};
    });
}
Selection pno_selection(const Mixed& parent, const Mixed& s, const Mixed& t, const Tensor& reference,
                        bool diagonal_pair) {
    int n = parent[0].shape[1], rank = reference.shape[1];
    auto gram = product("ma,mn,nb->ab", {parent, s, parent});
    close(gram[0], eye(n), 2.e-8, "PNO parent metric");
    for (int k = 1; k < 4; ++k) close(gram[k], Tensor({n, n}), 2.e-8, "PNO parent metric derivative");
    auto d = (1. / (1 + diagonal_pair)) * (product("ac,bc->ab", {spin(t), t}) + product("ca,cb->ab", {spin(t), t}));
    auto p = projector(d, rank);
    auto [e, u] = eigh(d[0]);
    std::reverse(e.data->begin(), e.data->end());
    double gap = rank < n ? e.ptr()[rank - 1] - e.ptr()[rank] : std::numeric_limits<double>::infinity();
    auto anchor = s.context->constant("PNO_anchor", mat(s[0], reference));
    auto projected = product("ma,ab,bc->mc", {parent, p, mat(parent.transpose(), anchor)});
    close(projected[0], reference, 2.e-8, "Selected PNO reference frame");
    auto frame = mat(projected, polar(product("ma,mn,nb->ab", {projected, s, projected})));
    return {frame, p, e, gap};
}
Mixed pno_loss(const PairEnergy& in, bool diagonal_pair) {
    auto t = in.amplitudes, F = in.fock;
    int n = t[0].shape[0], rank = in.rank;
    for (auto& x : F.x) close(x, tr(x), 1.e-10, "PNO Fock symmetry");
    auto d = (1. / (1 + diagonal_pair)) * (product("ac,bc->ab", {spin(t), t}) + product("ca,cb->ab", {spin(t), t}));
    auto p = projector(d, rank);
    auto loss = product("ab,ab->", {in.exchange, spin(t)}) - product("ab,ac,cd,db->", {in.exchange, p, spin(t), p});
    if (diagonal_pair) {
        auto [e, u] = eigh(d[0]);
        auto selected = mat(p, p.context->constant("selected_density_vectors",
                                                   u.select({range(n), [&]() {
                                                                 auto ids = range(n);
                                                                 ids.erase(ids.begin(), ids.begin() + n - rank);
                                                                 return ids;
                                                             }()})));
        auto gram = product("ma,mb->ab", {selected, selected});
        auto norm = derived("loss_selected_normalization", {gram}, [&]() {
            return std::array<Tensor, 4>{eye(rank), -.5 * gram[1], -.5 * gram[2],
                                         -.5 * gram[3] + .375 * (mat(gram[1], gram[2]) + mat(gram[2], gram[1]))};
        });
        selected = mat(selected, norm);
        auto singles = [&](const Mixed& f, const Mixed& v) {
            auto gap = product(",ab->ab", {in.occupied, f.context->constant("identity", eye(f[0].shape[0]))}) - f;
            auto eig = eigh(gap[0]).first;
            for (double x : *eig.data)
                if (std::abs(x) <= 1.e-12)
                    throw std::runtime_error("Screened singles denominator requires spectral-mask response.");
            return product("a,ab,b->",
                           {v, metric_solve(gap, gap.context->constant("identity", eye(f[0].shape[0]))), v});
        };
        loss = loss + singles(F, in.fov) -
               singles(product("ma,mn,nb->ab", {selected, F, selected}), product("ma,m->a", {selected, in.fov}));
    }
    return loss;
}
Mixed pno_energy(const Energies& initial, const Energies& final) {
    std::vector<Mixed> terms;
    for (auto* stage : {&initial, &final})
        for (auto& [p, x] : *stage) terms.push_back((p.first == p.second ? 1. : 2.) * pno_loss(x, p.first == p.second));
    if (terms.empty()) throw std::invalid_argument("Empty PNO energy stages.");
    return sum(terms);
}
Mixed tno_frame(const Triplet& r, const Mixed& parent, const Mixed& s, const PairMixed& frames,
                const PairMixed& doubles, const Tensor& reference) {
    std::vector<Mixed> terms;
    for (Pair p : std::vector<Pair>{{r[0], r[1]}, {r[1], r[2]}, {r[0], r[2]}}) {
        auto t = doubles.at(p);
        auto d = (1. / (3 * (1 + (p.first == p.second)))) *
                 (product("ac,bc->ab", {spin(t), t}) + product("ca,cb->ab", {spin(t), t}));
        auto transfer = product("ma,mn,nb->ab", {frames.at(p), s, parent});
        terms.push_back(product("ia,ij,jb->ab", {transfer, d, transfer}));
    }
    auto p = projector(sum(terms), reference.shape[1]);
    auto ao = product("ma,ab,nb->mn", {parent, p, parent});
    auto y = mat(ao, s.context->constant("TNO_anchor", mat(s[0], reference)));
    auto gram = product("ma,mn,nb->ab", {y, s, y});
    close(y[0], reference, 2.e-8, "Selected TNO reference frame");
    close(gram[0], eye(reference.shape[1]), 2.e-8, "TNO reference metric");
    return mat(y, polar(gram));
}
}  // namespace response
}  // namespace dlpno
}  // namespace psi
