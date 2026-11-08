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
#include <chrono>

namespace psi {
namespace dlpno {
namespace response {
namespace {
Mixed vector_product(const Mixed& a, const Mixed& b, bool transpose = false) {
    return doublet(a, b.reshape({int(b[0].size()), 1}), transpose).reshape({a[0].shape[transpose ? 1 : 0]});
}
Mixed outer(const Mixed& a, const Mixed& b) {
    return doublet(a.reshape({int(a[0].size()), 1}), b.reshape({int(b[0].size()), 1}), false, true);
}
Mixed project(const Mixed& s, const Mixed& amplitude) { return doublet(doublet(s, amplitude), s, false, true); }
Mixed chain(const Mixed& a, const Mixed& b, const Mixed& c) {
    const auto& x = a[0].shape;
    const auto& y = b[0].shape;
    const int n = c[0].shape[1];
    if (double(x[0]) * x[1] * y[1] + double(x[0]) * y[1] * n <= double(x[1]) * y[1] * n + double(x[0]) * x[1] * n)
        return doublet(doublet(a, b), c);
    return doublet(a, doublet(b, c));
}
// Sum_Q A_Q T B_Q^T, as two binary GEMMs. SharedMatrix reshapes
// retain the same allocation; no four-virtual-index ERI is formed.
Mixed ladder(const Mixed& a, const Mixed& t, const Mixed& b) {
    const int q = a[0].shape[0], n = a[0].shape[1], k = a[0].shape[2], m = b[0].shape[1];
    const int d = t[0].shape[1];
    auto left = doublet(a.reshape({q * n, k}), t).reshape({q, n, d}).permute({1, 0, 2});
    return doublet(left.reshape({n, q * d}), b.permute({1, 0, 2}).reshape({m, q * d}), false, true);
}
struct Dressed {
    Mixed occupied, virtuals, fitted_occupied, fitted_virtuals;
};
struct FockBlocks {
    Mixed gram, fitted_gram, vv, ov;
};
// These maps are owned by one residual evaluation. Only amplitude-independent
// factors live in Fitting across the fixed/affine replays and Hessian stages.
class CCSDIntermediates {
   public:
    const Fitting& fit;
    const std::vector<Mixed>& singles;
    const PairMixed& doubles;
    std::map<Pair, Mixed> amplitudes, spin_amplitudes, exchanges, spin_exchanges, rows;
    std::map<std::pair<Pair, int>, Mixed> projected_singles;
    std::map<std::pair<Pair, Pair>, Mixed> overlaps;
    std::map<std::pair<Pair, int>, Dressed> dressed_blocks;
    std::map<Pair, FockBlocks> fock_blocks;
    std::map<Pair, std::array<Mixed, 3>> intermediates;
    std::map<Pair, std::array<Mixed, 3>> fitted;
    CCSDIntermediates(const Fitting& f, const std::vector<Mixed>& s, const PairMixed& d)
        : fit(f), singles(s), doubles(d) {}
    const std::vector<int>& domain(Pair p) const { return fit.workspace.pairs.at(key(p)).occupied; }
    Mixed frame(Pair p) const { return fit.frames.at(key(p)); }
    Mixed amp(Pair p) {
        auto found = amplitudes.find(p);
        if (found != amplitudes.end()) return found->second;
        const auto& t = doubles.at(key(p));
        return amplitudes.emplace(p, p.first <= p.second ? t : t.transpose()).first->second;
    }
    Mixed u(Pair p) {
        auto found = spin_amplitudes.find(p);
        if (found != spin_amplitudes.end()) return found->second;
        return spin_amplitudes.emplace(p, spin(amp(p))).first->second;
    }
    Mixed overlap(Pair p, Pair q) {
        const auto id = std::make_pair(key(p), key(q));
        auto found = overlaps.find(id);
        if (found != overlaps.end()) return found->second;
        return overlaps.emplace(id, fit.overlap(p, q)).first->second;
    }
    Mixed single(Pair p, int i) {
        const auto id = std::make_pair(key(p), i);
        auto found = projected_singles.find(id);
        if (found != projected_singles.end()) return found->second;
        return projected_singles.emplace(id, vector_product(overlap(p, {i, i}), singles.at(i))).first->second;
    }
    Mixed tn(Pair p) {
        p = key(p);
        auto found = rows.find(p);
        if (found != rows.end()) return found->second;
        std::vector<Mixed> values;
        for (int i : domain(p)) values.push_back(single(p, i));
        return rows.emplace(p, stack(values)).first->second;
    }
    const std::array<Mixed, 3>& weighted(Pair p) {
        p = key(p);
        auto found = fitted.find(p);
        if (found != fitted.end()) return found->second;
        return fitted.emplace(p, std::array<Mixed, 3>{fit.weighted(p, 0), fit.weighted(p, 1), fit.weighted(p, 2)})
            .first->second;
    }
    Mixed exchange(Pair p) {
        const bool reverse = p.first > p.second;
        const auto canonical = key(p);
        auto found = exchanges.find(canonical);
        if (found == exchanges.end()) {
            const auto& a = fit.projected(canonical);
            const auto& dom = domain(p);
            auto qi = auxrow(a[1], pos(dom, canonical.first));
            auto qj = auxrow(a[1], pos(dom, canonical.second));
            // K_iajb uses the full inverse, as in the forward MP2/CCSD fit.
            found = exchanges.emplace(canonical, doublet(qi, doublet(a[3], qj), true)).first;
        }
        return reverse ? found->second.transpose() : found->second;
    }
    Mixed L(Pair p) {
        auto found = spin_exchanges.find(p);
        if (found != spin_exchanges.end()) return found->second;
        return spin_exchanges.emplace(p, spin(exchange(p))).first->second;
    }
    const Dressed& dressed(Pair p, int i) {
        const auto id = std::make_pair(key(p), i);
        auto found = dressed_blocks.find(id);
        if (found != dressed_blocks.end()) return found->second;
        const auto& a = fit.projected(p);
        const auto& w = weighted(p);
        const int qi = a[1][0].shape[0], nv = a[1][0].shape[2], r = pos(domain(p), i), no = domain(p).size();
        const auto t = tn(p), ti = row(t, r).reshape({nv, 1});
        auto build = [&](const Mixed& oo, const Mixed& ov, const Mixed& vv) {
            auto o = auxrow(oo, r) + doublet(ov.reshape({qi * no, nv}), ti).reshape({qi, no});
            auto v = auxrow(ov, r) - doublet(o, t) + doublet(vv.reshape({qi * nv, nv}), ti).reshape({qi, nv});
            return std::make_pair(o, v);
        };
        const auto raw = build(a[0], a[1], a[2]), fitted = build(w[0], w[1], w[2]);
        return dressed_blocks.emplace(id, Dressed{raw.first, raw.second, fitted.first, fitted.second}).first->second;
    }
    const FockBlocks& partial_fock(Pair p) {
        p = key(p);
        auto found = fock_blocks.find(p);
        if (found != fock_blocks.end()) return found->second;
        const auto& a = fit.projected(p);
        const auto& w = weighted(p);
        const auto t = tn(p);
        const int nq = a[1][0].shape[0], no = domain(p).size(), nv = t[0].shape[1];
        auto gram = doublet(a[1].reshape({nq, no * nv}), t.reshape({no * nv, 1}));
        auto fitted_gram = doublet(a[4], gram);
        // First contract the occupied index. This is the energy code's
        // exchange Fock construction, without a Qrme intermediate.
        auto ht = doublet(w[1].permute({0, 2, 1}).reshape({nq * nv, no}), t)
                      .reshape({nq, nv, nv})
                      .permute({1, 0, 2})
                      .reshape({nv, nq * nv});
        auto bar = [&](const Mixed& block, const Mixed& left) {
            const int nr = block[0].shape[1];
            auto coulomb = doublet(fitted_gram, block.reshape({nq, nr * nv}), true).reshape({nr, nv});
            auto exchange = doublet(block.permute({1, 0, 2}).reshape({nr, nq * nv}), ht, false, true);
            return congruence(left, fit.molecular.f, frame(p)) + 2. * coulomb - exchange;
        };
        auto left = fit.molecular.l.select({range(fit.molecular.l[0].shape[0]), domain(p)});
        return fock_blocks.emplace(p, FockBlocks{gram, fitted_gram, bar(a[2], frame(p)), bar(a[1], left)})
            .first->second;
    }
    const std::array<Mixed, 3>& pair_intermediates(Pair p) {
        auto found = intermediates.find(p);
        if (found != intermediates.end()) return found->second;
        const auto [i, j] = p;
        const auto& dom = domain(p);
        const auto& a = fit.projected(p);
        const auto& w = weighted(p);
        const int nq = a[1][0].shape[0], nv = a[1][0].shape[2], no = dom.size();
        const auto t = tn(p), f = a[2], wi = auxrow(w[1], pos(dom, i)), wj = auxrow(w[1], pos(dom, j));
        const auto& di = dressed(p, i);
        const auto& dj = dressed(p, j);
        // Jiang 82: dressed OO dot, followed by sum_Q h_Q T h_Q^T.
        auto beta = doublet(di.occupied, dj.fitted_occupied, true) + ladder(a[1], amp(p), w[1]);
        auto ooij = auxat(a[0], pos(dom, i), pos(dom, j)).reshape({nq, 1});
        auto coulomb = doublet(ooij, w[1].reshape({nq, no * nv}), true).reshape({no, nv});
        auto exchange = doublet(auxrow(a[0], pos(dom, i)), wj, true);
        // Jiang 83b/84b: contract t into the same local W factors before
        // building a three-virtual tensor. This inserts no PNO projector.
        auto gamma_w =
            doublet(doublet(wi, single(p, j).reshape({nv, 1})), f.reshape({nq, nv * nv}), true).reshape({nv, nv});
        auto ft = doublet(f.reshape({nq * nv, nv}), single(p, i).reshape({nv, 1})).reshape({nq, nv});
        auto delta_w = 2. * doublet(ft, wj, true) -
                       doublet(doublet(wj, single(p, i).reshape({nv, 1})), f.reshape({nq, nv * nv}), true)
                           .reshape({nv, nv})
                           .transpose();
        auto gamma = -doublet(t, coulomb, true) + gamma_w;
        auto delta = -doublet(t, 2. * exchange - coulomb, true) + delta_w;
        for (int l : dom) {
            const Pair il{i, l}, lj{l, j};
            auto kil = this->exchange(il), llj = L(lj);
            auto column = vector_product(overlap(p, il), vector_product(kil, single(il, j), true));
            gamma = gamma - outer(single(p, l), column);
            auto middle = chain(amp(lj), overlap(lj, il), kil);
            gamma = gamma - .5 * chain(overlap(p, lj), middle, overlap(il, p));
            column = vector_product(overlap(lj, p), vector_product(llj, single(lj, i), true), true);
            delta = delta - outer(single(p, l), column);
            middle = chain(u(il), overlap(il, lj), llj);
            delta = delta + .5 * chain(overlap(p, il), middle, overlap(lj, p));
        }
        return intermediates.emplace(p, std::array<Mixed, 3>{beta, gamma, delta}).first->second;
    }
};
}  // namespace
Mixed retained_energy(const Fitting& fit, const std::vector<Mixed>& singles, const PairMixed& doubles) {
    const auto& m = fit.molecular;
    CCSDIntermediates state(fit, singles, doubles);
    std::vector<Mixed> terms;
    for (size_t i = 0; i < singles.size(); ++i) {
        auto f = congruence(col(m.l, i).reshape({m.l[0].shape[0], 1}), m.f, fit.frames.at({i, i}));
        terms.push_back(2. * inner(f.reshape(singles[i][0].shape), singles[i]));
    }
    for (const auto& [p, t] : doubles) {
        auto tau = t + outer(state.single(p, p.first), state.single(p, p.second));
        terms.push_back((p.first == p.second ? 1. : 2.) * inner(tau, state.L(p)));
    }
    return terms.empty() ? m.s.context->constant("zero_energy", Tensor({}, 0.)) : sum(terms);
}
CCSDResult ccsd_residual(const Fitting& fit, const std::vector<Mixed>& singles, const PairMixed& doubles) {
    auto checkpoint = std::chrono::steady_clock::now();
    auto phase = [&](const std::string& name) {
        auto now = std::chrono::steady_clock::now();
        fit.ccsd_seconds[name] += std::chrono::duration<double>(now - checkpoint).count();
        checkpoint = now;
    };
    const auto& m = fit.molecular;
    const int no = m.l[0].shape[1];
    if (singles.size() != size_t(no)) throw std::invalid_argument("CCSD singles graph mismatch.");
    for (const auto& t : singles)
        for (double x : t[3])
            if (x != 0.) throw std::invalid_argument("CCSD mixed source requires affine amplitudes.");
    for (const auto& [p, t] : doubles)
        for (double x : t[3])
            if (x != 0.) throw std::invalid_argument("CCSD mixed source requires affine amplitudes.");
    fit.prepare_raw();
    CCSDIntermediates s(fit, singles, doubles);
    auto foo = congruence(m.l, m.f, m.l);
    std::vector<std::vector<Mixed>> fij(no, std::vector<Mixed>(no));
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j) fij[i][j] = at(foo, i, j);
    std::vector<Pair> orders;
    for (const auto& [p, t] : doubles) {
        orders.push_back(p);
        if (p.first != p.second) orders.push_back({p.second, p.first});
    }
    for (Pair p : orders) {
        const auto [i, j] = p;
        const auto& a = fit.projected(p);
        const auto& w = s.weighted(p);
        const int x = pos(s.domain(p), i), y = pos(s.domain(p), j), nq = a[0][0].shape[0];
        auto coulomb = inner(auxat(a[0], x, y), s.partial_fock(p).fitted_gram.reshape({nq}));
        auto exchange = doublet(auxrow(a[0], y), auxrow(w[1], x), true);
        fij[i][j] = fij[i][j] + 2. * coulomb - inner(s.tn(p), exchange);
    }
    std::vector<Mixed> one;
    for (const auto& t : singles) one.push_back(0. * t);
    PairMixed two;
    phase("preparation");
    // Jiang 87--90: A/C updates and the B scatter use the same pair spaces
    // as compute_R_ia(). Scatter destinations are reduced in fixed order.
    for (Pair p : orders) {
        const auto [i, k] = p;
        const auto& dom = s.domain(p);
        const auto& a = fit.projected(p);
        const auto& w = s.weighted(p);
        const int mi = pos(dom, i), mk = pos(dom, k), nq = a[1][0].shape[0], nv = a[1][0].shape[2];
        const auto t = s.tn(p), u = s.u(p), f = a[2];
        if (i == k) {
            const auto& part = s.partial_fock(p);
            std::vector<Mixed> fi;
            for (int l : dom) fi.push_back(fij[l][i]);
            auto bare = congruence(col(m.l, i).reshape({m.l[0].shape[0], 1}), m.f, s.frame(p)).reshape({nv});
            auto tmp = doublet(auxrow(w[0], mi), t);
            auto exchange = vector_product(f.permute({1, 0, 2}).reshape({nv, nq * nv}), tmp.reshape({nq * nv}));
            one[i] = one[i] + bare + 2. * vector_product(auxrow(a[1], mi), part.fitted_gram.reshape({nq}), true) -
                     exchange + vector_product(part.vv, singles[i]) -
                     vector_product(t, vector_product(part.ov, singles[i]), true) - vector_product(t, stack(fi), true);
        }
        // A: sum_Qcd h_k(Q,c) U(d,c) f(Q,d,a); never build W(c,d,a).
        auto tmp = doublet(auxrow(w[1], mk), u, false, true);
        auto av = doublet(tmp.reshape({1, nq * nv}), f.reshape({nq * nv, nv})).reshape({nv});
        one[i] = one[i] + vector_product(s.overlap(p, {i, i}), av, true);
        auto fk = congruence(col(m.l, k).reshape({m.l[0].shape[0], 1}), m.f, s.frame(p)).reshape({nv});
        for (int l : dom) {
            const Pair kl{k, l};
            auto coefficient = inner(project(s.overlap(kl, p), u.transpose()), s.exchange(kl));
            one[i] = one[i] - product(",a->a", {coefficient, s.single({i, i}, l)});
            fk = fk + vector_product(s.overlap(p, kl), vector_product(s.L(kl), s.single(kl, l)));
        }
        one[i] = one[i] + vector_product(s.overlap(p, {i, i}), vector_product(u, fk), true);
        auto M = doublet(auxrow(a[0], mi), auxrow(w[1], mk), true) + doublet(t, s.exchange(p));
        auto B = doublet(M, u, false, true);
        for (size_t l = 0; l < dom.size(); ++l)
            one[dom[l]] = one[dom[l]] - vector_product(s.overlap(p, {dom[l], dom[l]}), row(B, l), true);
    }
    phase("singles residual");
    for (int j = 0; j < no; ++j) {
        const Pair p{j, j};
        auto correction = vector_product(s.partial_fock(p).ov, singles[j]);
        const auto& dom = s.domain(p);
        for (size_t k = 0; k < dom.size(); ++k) fij[dom[k]][j] = fij[dom[k]][j] + row(correction, k);
    }
    for (auto [i, j] : orders)
        for (int l : s.domain({i, j}))
            fij[i][j] = fij[i][j] + inner(s.exchange({l, i}), project(s.overlap({l, i}, {l, j}), s.u({l, j})));
    phase("occupied Fock");
    std::map<Pair, Mixed> ordered_mixed;
    auto ordered_cd = [&](int i, int j) {
        const Pair p{i, j};
        auto found = ordered_mixed.find(p);
        if (found != ordered_mixed.end()) return found->second;
        auto c = 0. * s.amp(p), d = c;
        for (int k : s.domain(p)) {
            const Pair ki{k, i}, kj{k, j}, ik{i, k}, jk{j, k};
            const auto [J, K] = fit.mixed_integrals(i, j, k);
            auto gamma = chain(s.overlap(p, ki), s.pair_intermediates(ki)[1], s.overlap(ki, kj));
            c = c - doublet(doublet(J + gamma, s.amp(kj), false, true), s.overlap(kj, p));
            auto delta = chain(s.overlap(p, ik), s.pair_intermediates(ik)[2], s.overlap(ik, jk));
            d = d + .5 * doublet(doublet(2. * K - J + delta, s.u(jk), false, true), s.overlap(jk, p));
        }
        return ordered_mixed.emplace(p, .5 * c + c.transpose() + d).first->second;
    };
    // Jiang 75--86. The only cross-pair projections below are those in the
    // energy equations. Rectangular J/K vertices retain their own local fits.
    for (const auto& [p, t] : doubles) {
        const auto [i, j] = p;
        const auto& dom = s.domain(p);
        const auto& a = fit.projected(p);
        const auto& w = s.weighted(p);
        const int nq = a[1][0].shape[0], nv = a[1][0].shape[2], nl = dom.size();
        auto dress_vv = [&](const Mixed& h, const Mixed& f) {
            return f - doublet(s.tn(p), h.permute({1, 0, 2}).reshape({nl, nq * nv}), true)
                           .reshape({nv, nq, nv})
                           .permute({1, 0, 2});
        };
        auto D = dress_vv(a[1], a[2]), fitted_D = dress_vv(w[1], w[2]);
        auto result = doublet(s.dressed(p, i).virtuals, s.dressed(p, j).fitted_virtuals, true) + ladder(D, t, fitted_D);
        const auto& part = s.partial_fock(p);
        auto virt = part.vv - doublet(s.tn(p), part.ov, true);
        const auto beta = s.pair_intermediates(p)[0];
        for (size_t k = 0; k < dom.size(); ++k)
            for (size_t l = 0; l < dom.size(); ++l) {
                const Pair q{dom[k], dom[l]};
                if (!doubles.count(key(q))) continue;
                result = result + product(",ab->ab", {at(beta, k, l), project(s.overlap(p, q), s.amp(q))});
                const auto ov = s.overlap(q, p);
                virt = virt - congruence(ov, doublet(s.u(q), s.exchange(q), false, true), ov);
            }
        result =
            result + doublet(t, virt, false, true) + doublet(virt, t) + ordered_cd(i, j) + ordered_cd(j, i).transpose();
        for (int perm = 0; perm < 2; ++perm) {
            const int a = perm ? j : i, b = perm ? i : j;
            auto g = 0. * t;
            for (int k : s.domain({a, a}))
                g = g - product(",ab->ab", {fij[k][b], project(s.overlap(p, {a, k}), s.amp({a, k}))});
            result = result + (perm ? g.transpose() : g);
        }
        two[p] = result;
    }
    phase("doubles residual");
    return {one, two};
}
}  // namespace response
}  // namespace dlpno
}  // namespace psi
