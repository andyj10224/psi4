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
#include <numeric>
#include <limits>
#include <exception>
#include "psi4/libpsi4util/process.h"
#ifdef USING_LAPACK_MKL
#include <mkl.h>
#endif

namespace psi {
namespace dlpno {
namespace response {
namespace {
std::vector<int> inverse(const std::vector<int>& axes) {
    std::vector<int> out(axes.size());
    for (size_t i = 0; i < axes.size(); ++i) out[axes[i]] = i;
    return out;
}
std::vector<int> ranks(const Triplet& r) {
    auto order = range(3);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return r[a] < r[b]; });
    return inverse(order);
}
Triplet sorted(Triplet r) {
    std::sort(r.begin(), r.end());
    return r;
}
Tensor energy_moment(const Triplet& r, const Tensor& v) {
    auto [i, j, k] = r;
    auto out = 8. * v;
    for (auto term : std::vector<std::pair<Triplet, double>>{
             {{k, j, i}, -4.}, {{i, k, j}, -4.}, {{j, i, k}, -4.}, {{j, k, i}, 2.}, {{k, i, j}, 2.}})
        out = out + term.second * v.permute(ranks(term.first));
    return (1. / (i == k ? 6 : i == j || j == k ? 2 : 1)) * out;
}
Tensor action(const Tensor& f, const Tensor& t, double occ) {
    return contract("ad,dbc->abc", {f, t}) + contract("bd,adc->abc", {f, t}) + contract("cd,abd->abc", {f, t}) -
           occ * t;
}
Tensor transfer(const Tensor& t, const Tensor& a, const Tensor& b, const Tensor& c) {
    return contract("ia,jb,kc,abc->ijk", {a, b, c, t});
}
Tensor tangent_transfer(const Tensor& t, const Tensor& s, const Tensor& ds) {
    return transfer(t, ds, s, s) + transfer(t, s, ds, s) + transfer(t, s, s, ds);
}
void check(const Tensor& x, double tolerance, const std::string& message) {
    close(x, Tensor(x.shape), tolerance, message);
}
}  // namespace
std::array<Mixed, 3> triples_moments(const Triplet& r, const std::vector<int>& domain, const Mixed& metric,
                                     const Mixed& oo, const Mixed& ov, const Mixed& vv,
                                     const std::function<Mixed(int, int)>& transfer,
                                     const std::function<Mixed(int)>& single,
                                     const std::function<Mixed(int, int)>& doubles,
                                     const std::function<Mixed(int)>& fov, const Mixed& vf) {
    auto g = symmetric_inverse(metric), gf = filtered_inverse(metric, g);
    std::map<Pair, Mixed> pairs, ooovs;
    std::map<int, Mixed> ovvvs;
    auto pair = [&](int a, int b) {
        Pair p{a, b};
        auto it = pairs.find(p);
        if (it != pairs.end()) return it->second;
        auto s = transfer(a, b);
        return pairs.emplace(p, product("ia,ij,jb->ab", {s, doubles(a, b), s})).first->second;
    };
    auto ovvv = [&](int a) {
        auto it = ovvvs.find(a);
        if (it != ovvvs.end()) return it->second;
        return ovvvs.emplace(a, product("Qa,QR,Rbd->abd", {auxrow(ov, pos(domain, a)), g, vv})).first->second;
    };
    auto ooov = [&](int b, int c) {
        Pair p{b, c};
        auto it = ooovs.find(p);
        if (it != ooovs.end()) return it->second;
        return ooovs.emplace(p, product("Ql,QR,Rc->lc", {auxrow(oo, pos(domain, b)), gf, auxrow(ov, pos(domain, c))}))
            .first->second;
    };
    int n = vv[0].shape[2];
    auto w = metric.context->constant("triples_zero", Tensor({n, n, n}));
    auto perm = range(3);
    do {
        int a = r[perm[0]], b = r[perm[1]], c = r[perm[2]];
        auto term = product("abd,cd->abc", {ovvv(a), pair(c, b)}), integral = ooov(b, c);
        for (size_t m = 0; m < domain.size(); ++m)
            term = term - product("ab,c->abc", {pair(a, domain[m]), row(integral, m)});
        w = w + term.permute(inverse(perm));
    } while (std::next_permutation(perm.begin(), perm.end()));
    auto v = w;
    for (int slot = 0; slot < 3; ++slot) {
        int a = r[slot];
        std::vector<int> other;
        for (int k = 0; k < 3; ++k)
            if (k != slot) other.push_back(r[k]);
        int b = other[0], c = other[1];
        auto s = product("ia,i->a", {transfer(a, a), single(a)}),
             exchange = product("Qb,QR,Rc->bc", {auxrow(ov, pos(domain, b)), gf, auxrow(ov, pos(domain, c))});
        auto term = product("a,bc->abc", {s, exchange}) + product("a,bc->abc", {fov(a), pair(b, c)});
        std::vector<int> axes{1, 2};
        axes.insert(axes.begin() + slot, 0);
        v = v + term.permute(axes);
    }
    return {w, v, vf};
}
Triples::Triples(std::shared_ptr<Fitting> f, std::vector<Mixed> s, PairMixed d)
    : fit(std::move(f)), singles(std::move(s)), doubles(std::move(d)) {
    for (const auto& t : singles)
        for (double value : *t[3].data)
            if (value != 0.) throw std::invalid_argument("Triples mixed inputs require affine CCSD amplitudes.");
    for (const auto& item : doubles)
        for (double value : *item.second[3].data)
            if (value != 0.) throw std::invalid_argument("Triples mixed inputs require affine CCSD amplitudes.");
    foo = product("mi,mn,nj->ij", {fit->molecular.l, fit->molecular.f, fit->molecular.l});
}
void Triples::build_all() {
    std::vector<Triplet> pending;
    for (auto& [r, data] : fit->workspace.triplets)
        if (!moments.count(r)) pending.push_back(r);
    struct Result {
        Mixed parent, frame;
        std::array<Mixed, 3> moments;
    };
    std::vector<Result> rows(pending.size());
    // Each row owns fitting caches and all temporary tensors. Reference
    // intermediates use the synchronized, reference-owned numerical cache.
    parallel_rows(pending.size(), [&](size_t k) {
        auto local_fit = std::make_shared<Fitting>(*fit);
        Triples task(local_fit, singles, doubles);
        const auto r = pending[k];
        rows[k].moments = task.blocks(r);
        rows[k].parent = task.parents.at(r);
        rows[k].frame = task.frames.at(r);
        // Retain results only. Fitting panels must be released at task exit,
        // so their peak storage scales with workers, not the triplet count.
    });
    for (size_t k = 0; k < pending.size(); ++k) {
        const auto r = pending[k];
        parents[r] = std::move(rows[k].parent);
        frames[r] = std::move(rows[k].frame);
        moments[r] = std::move(rows[k].moments);
    }
}
Mixed Triples::frame(const Triplet& r) {
    auto it = frames.find(r);
    if (it != frames.end()) return it->second;
    const auto& data = fit->workspace.triplets.at(r);
    auto parent = moving_parent(fit->pao(data.paos), fit->molecular.s, data.parent).first;
    parents[r] = parent;
    return frames.emplace(r, tno_frame(r, parent, fit->molecular.s, fit->frames, doubles, data.frame)).first->second;
}
std::array<Mixed, 3> Triples::blocks(const Triplet& r) {
    auto it = moments.find(r);
    if (it != moments.end()) return it->second;
    const auto& data = fit->workspace.triplets.at(r);
    const auto& m = fit->molecular;
    auto y = frame(r), X = fit->coefficients(data.paos, data.parent, y);
    auto oo = fit->raw("oo", data.auxiliary, data.occupied, data.occupied),
         ov = product("Qma,ab->Qmb", {fit->raw("ov", data.auxiliary, data.occupied, data.paos), X}),
         vv = product("ma,Qmn,nb->Qab", {X, fit->raw("vv", data.auxiliary, data.paos, data.paos), X});
    auto transfer = [&](int i, int j) { return product("ma,mn,nb->ab", {fit->frames.at(key({i, j})), m.s, y}); };
    auto amp = [&](int i, int j) {
        auto t = doubles.at(key({i, j}));
        return i <= j ? t : t.transpose();
    };
    auto fov = [&](int i) { return product("m,mn,nb->b", {col(m.l, i), m.f, y}); };
    return moments
        .emplace(r, triples_moments(
                        r, data.occupied, fit->metric(data.auxiliary), oo, ov, vv, transfer,
                        [&](int i) { return singles.at(i); }, amp, fov, product("ma,mn,nb->ab", {y, m.f, y})))
        .first->second;
}
Mixed Triples::edge(const Triplet& r, const Triplet& q) {
    auto key = std::make_pair(r, q);
    auto it = overlaps.find(key);
    if (it != overlaps.end()) return it->second;
    return overlaps.emplace(key, product("ma,mn,nb->ab", {frame(r), fit->molecular.s, frame(q)})).first->second;
}
std::shared_ptr<TripleDirection> Triples::direction(int component, double threshold, double reference_energy) {
    if (component != 1 && component != 2) throw std::invalid_argument("Expected a first triples component.");
    build_all();
    auto out = std::make_shared<TripleDirection>();
    std::map<Triplet, Tensor> left_residual;
    double energy = 0.;
    for (auto& [r, data] : fit->workspace.triplets) {
        auto block = blocks(r);
        auto w = block[0], v = block[1], f = block[2];
        int n = f[0].shape[0];
        close(w[0], data.w, 2.e-10, "Native triples W replay", 1.e-8);
        close(v[0], data.v, 2.e-10, "Native triples V replay", 1.e-8);
        Tensor diag({n, n});
        for (int i = 0; i < n; ++i) diag.mutable_ptr()[i * n + i] = f[0].ptr()[i * n + i];
        close(f[0], diag, 1.e-8, "Semicanonical TNO Fock");
        out->frames[r] = frame(r)[component];
        out->t[r] = data.t;
        out->z[r] = data.z;
        out->seeds[r] = energy_moment(r, v[0]);
        out->dseeds[r] = energy_moment(r, v[component]);
        double occ = 0., docc = 0.;
        for (int a : r) {
            occ += foo[0].ptr()[a * foo[0].shape[0] + a];
            docc += foo[component].ptr()[a * foo[0].shape[0] + a];
        }
        Tensor gap({n, n, n});
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                for (int c = 0; c < n; ++c)
                    gap.mutable_ptr()[(a * n + b) * n + c] =
                        f[0].ptr()[a * n + a] + f[0].ptr()[b * n + b] + f[0].ptr()[c * n + c] - occ;
        out->gaps[r] = gap;
        out->right[r] = w[component] + action(f[component], data.t, docc);
        out->left[r] = out->dseeds[r] + action(f[component], data.z, docc);
        left_residual[r] = out->seeds[r] + contract("abc,abc->abc", {gap, data.z});
        energy += dot(out->seeds[r], data.t);
        out->edges[r] = {};
    }
    for (auto& [r, data] : fit->workspace.triplets) {
        auto residual = data.w + contract("abc,abc->abc", {out->gaps.at(r), data.t});
        for (auto e : data.edges) {
            Triplet order{e[2], e[3], e[4]}, q = sorted(order);
            auto axes = ranks(order);
            auto s = edge(r, q);
            auto f = at(foo, e[0], e[1]);
            auto tq = out->t.at(q).permute(axes), moved = transfer(tq, s[0], s[0], s[0]);
            residual = residual - f[0].scalar() * moved;
            out->right.at(r) = out->right.at(r) - f[component].scalar() * moved -
                               f[0].scalar() * tangent_transfer(tq, s[0], s[component]);
            auto back = inverse(axes);
            auto st = tr(s[0]), dst = tr(s[component]);
            auto zq = transfer(data.z, st, st, st).permute(back);
            left_residual.at(q) = left_residual.at(q) - f[0].scalar() * zq;
            out->left.at(q) = out->left.at(q) - f[component].scalar() * zq -
                              f[0].scalar() * tangent_transfer(data.z, st, dst).permute(back);
            out->edges.at(r).push_back(
                {q, axes, {e[0], e[1]}, f[0].scalar(), f[component].scalar(), s[0], s[component]});
        }
        check(residual, threshold, "Native iterative triples residual");
    }
    for (auto& [r, x] : left_residual) check(x, threshold, "Native triples adjoint residual");
    if (std::abs(energy - reference_energy) > 1.e-11 + 1.e-9 * std::abs(reference_energy))
        throw std::runtime_error("Native triples energy replay mismatch.");
    return out;
}
Tensor TripleDirection::pack(const std::map<Triplet, Tensor>& x) const {
    size_t n = 0;
    for (auto& [r, t] : this->t) n += t.size();
    if (n > size_t(std::numeric_limits<int>::max())) throw std::invalid_argument("Triples vector too large.");
    Tensor flat({int(n)});
    size_t p = 0;
    for (auto& [r, t] : this->t) {
        auto& v = x.at(r);
        if (v.shape != t.shape) throw std::invalid_argument("Triples packed tensor dimensions disagree.");
        std::copy_n(v.ptr(), v.size(), flat.mutable_ptr() + p);
        p += v.size();
    }
    return flat;
}
std::map<Triplet, Tensor> TripleDirection::unpack(const Tensor& flat) const {
    std::map<Triplet, Tensor> x;
    size_t p = 0;
    for (auto& [r, t] : t) {
        if (p + t.size() > flat.size()) throw std::invalid_argument("Short triples response vector.");
        x[r] = Tensor(t.shape, flat.ptr() + p);
        p += t.size();
    }
    if (p != flat.size()) throw std::invalid_argument("Long triples response vector.");
    return x;
}
Tensor TripleDirection::apply(const Tensor& flat, bool left) const {
    auto x = unpack(flat);
    std::map<Triplet, Tensor> result;
    for (auto& [r, t] : t) result[r] = contract("abc,abc->abc", {gaps.at(r), x.at(r)});
    // The transpose has intertriplet scatter. Keep that reduction ordered;
    // permutations/packing and independent output cells use native OpenMP.
    for (auto& [r, es] : edges)
        for (auto& e : es) {
            if (left) {
                auto s = tr(e.overlap);
                result.at(e.source) =
                    result.at(e.source) - e.fock * transfer(x.at(r), s, s, s).permute(inverse(e.axes));
            } else
                result.at(r) =
                    result.at(r) - e.fock * transfer(x.at(e.source).permute(e.axes), e.overlap, e.overlap, e.overlap);
        }
    return pack(result);
}
double TripleDirection::energy_tangent() const {
    double e = 0.;
    for (auto& [r, t] : t) e += dot(dseeds.at(r), t) + dot(z.at(r), right.at(r));
    return e;
}
double TripleColumn::element(const TripleDirection& row, Triples& inputs) {
    inputs.build_all();
    double result = 0.;
    for (auto& [r, t] : row.t) {
        auto z = row.z.at(r);
        result += dot(row.left.at(r), dt.at(r)) + dot(dz.at(r), row.right.at(r));
        auto block = inputs.blocks(r);
        auto w = block[0][3], v = block[1][3], f = block[2][3];
        double occ = 0.;
        for (int a : r) occ += at(inputs.foo, a, a)[3].scalar();
        result += dot(energy_moment(r, v), t) + dot(z, w + action(f, t, occ));
        const auto &es = row.edges.at(r), &cs = direction->edges.at(r);
        if (es.size() != cs.size()) throw std::runtime_error("Triples edge lists disagree.");
        for (size_t index = 0; index < es.size(); ++index) {
            const auto& eu = es[index];
            const auto& ev = cs[index];
            if (eu.source != ev.source || eu.axes != ev.axes || eu.occupied != ev.occupied)
                throw std::runtime_error("Triples ordered edge metadata disagree.");
            auto key = std::make_pair(r, int(index));
            auto found = covectors.find(key);
            if (found == covectors.end()) {
                auto tq = direction->t.at(ev.source).permute(ev.axes), zz = direction->z.at(r);
                Tensor bar(ev.overlap.shape), column_bar(ev.overlap.shape);
                double value = 0.;
                int leg = 0;
                for (std::vector<int> axes : std::vector<std::vector<int>>{{0, 1, 2}, {1, 0, 2}, {2, 1, 0}}) {
                    auto a = zz.permute(axes), b = tq.permute(axes);
                    auto cov = contract("ijk,abc,jb,kc->ia", {a, b, ev.overlap, ev.overlap});
                    if (leg++ == 0) value = dot(ev.overlap, cov);
                    bar = bar + cov;
                    column_bar = column_bar + contract("ijk,abc,jb,kc->ia", {a, b, ev.doverlap, ev.overlap}) +
                                 contract("ijk,abc,jb,kc->ia", {a, b, ev.overlap, ev.doverlap});
                }
                found = covectors.emplace(key, Covector{bar, column_bar, value, dot(bar, ev.doverlap)}).first;
            }
            const auto& cov = found->second;
            auto suv = inputs.edge(r, eu.source)[3];
            double fuv = at(inputs.foo, eu.occupied.first, eu.occupied.second)[3].scalar();
            result -= fuv * cov.value + eu.dfock * cov.column + ev.dfock * dot(cov.bar, eu.doverlap) +
                      eu.fock * (dot(cov.bar, suv) + dot(cov.column_bar, eu.doverlap));
        }
    }
    if (!std::isfinite(result)) throw std::runtime_error("Nonfinite native triples curvature.");
    return result;
}
}  // namespace response
}  // namespace dlpno
}  // namespace psi
