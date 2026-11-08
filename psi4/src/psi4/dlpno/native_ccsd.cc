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
#include <tuple>
#include <chrono>

namespace psi {
namespace dlpno {
namespace response {
Mixed retained_energy(const Fitting& fit, const std::vector<Mixed>& singles, const PairMixed& doubles) {
    const auto& m = fit.molecular;
    std::vector<Mixed> terms;
    for (size_t i = 0; i < singles.size(); ++i)
        terms.push_back(
            2. * product("a,a->", {product("m,mn,na->a", {col(m.l, i), m.f, fit.frames.at({i, i})}), singles[i]}));
    for (auto& [p, t] : doubles) {
        auto Y = fit.frames.at(p);
        auto transfer = [&](int i) { return product("ma,mn,nb->ab", {Y, m.s, fit.frames.at({i, i})}); };
        auto ti = product("ab,b->a", {transfer(p.first), singles.at(p.first)}),
             tj = product("ab,b->a", {transfer(p.second), singles.at(p.second)});
        auto tau = t + product("a,b->ab", {ti, tj});
        const auto& panels = fit.projected(p);
        auto& dom = fit.workspace.pairs.at(p).occupied;
        auto K = product("Qa,QR,Rb->ab",
                         {auxrow(panels[1], pos(dom, p.first)), panels[3], auxrow(panels[1], pos(dom, p.second))});
        terms.push_back((p.first == p.second ? 1. : 2.) * product("ab,ab->", {tau, spin(K)}));
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
    int no = m.l[0].shape[1];
    if (singles.size() != size_t(no)) throw std::invalid_argument("CCSD singles graph mismatch.");
    for (auto& t : singles)
        for (double x : *t[3].data)
            if (x != 0.) throw std::invalid_argument("CCSD mixed source requires affine amplitudes.");
    for (auto& [p, t] : doubles)
        for (double x : *t[3].data)
            if (x != 0.) throw std::invalid_argument("CCSD mixed source requires affine amplitudes.");
    auto C = [](const std::string& eq, std::initializer_list<Mixed> x) { return product(eq, x); };
    auto domain = [&](Pair p) -> const std::vector<int>& { return fit.workspace.pairs.at(key(p)).occupied; };
    auto frame = [&](Pair p) { return fit.frames.at(key(p)); };
    auto amp = [&](Pair p) {
        auto t = doubles.at(key(p));
        return p.first <= p.second ? t : t.transpose();
    };
    std::map<std::pair<Pair, Pair>, Mixed> overlaps;
    auto ov = [&](Pair p, Pair q) {
        auto id = std::make_pair(key(p), key(q));
        auto found = overlaps.find(id);
        if (found != overlaps.end()) return found->second;
        return overlaps.emplace(id, fit.overlap(p, q)).first->second;
    };
    std::map<std::pair<Pair, int>, Mixed> projected_singles;
    auto sing = [&](Pair p, int i) {
        auto id = std::make_pair(p, i);
        auto it = projected_singles.find(id);
        if (it != projected_singles.end()) return it->second;
        return projected_singles.emplace(id, C("ab,b->a", {ov(p, {i, i}), singles.at(i)})).first->second;
    };
    std::map<Pair, Mixed> rows;
    auto tn = [&](Pair p) {
        auto it = rows.find(p);
        if (it != rows.end()) return it->second;
        std::vector<Mixed> x;
        for (int i : domain(p)) x.push_back(sing(p, i));
        return rows.emplace(p, stack(x)).first->second;
    };
    auto df = [&](Pair p, std::string eq, std::vector<Mixed> args) {
        auto arrow = eq.find("->");
        int seen = 0;
        for (size_t i = 0; i < arrow; ++i)
            if (eq[i] == 'Q') {
                if (++seen == 2) eq[i] = 'R';
            }
        if (seen != 2) throw std::invalid_argument("Expected two fitted factors.");
        eq.insert(arrow, ",QR");
        args.push_back(fit.projected(p)[4]);
        return product(eq, args);
    };
    std::map<Pair, Mixed> exchanges;
    auto exchange = [&](int i, int j) {
        Pair p{i, j};
        auto it = exchanges.find(p);
        if (it != exchanges.end()) return it->second;
        auto& a = fit.projected(p);
        auto& dom = domain(p);
        return exchanges.emplace(p, C("Qa,QR,Rb->ab", {auxrow(a[1], pos(dom, i)), a[3], auxrow(a[1], pos(dom, j))}))
            .first->second;
    };
    std::map<std::pair<Pair, int>, std::pair<Mixed, Mixed>> dressed_cache;
    auto dressed = [&](Pair p, int i) {
        auto id = std::make_pair(p, i);
        auto it = dressed_cache.find(id);
        if (it != dressed_cache.end()) return it->second;
        auto a = fit.projected(p);
        int r = pos(domain(p), i);
        auto t = tn(p);
        auto o = auxrow(a[0], r) + C("Qkc,c->Qk", {a[1], row(t, r)});
        auto v = auxrow(a[1], r) - C("Qk,ka->Qa", {o, t}) + C("Qac,c->Qa", {a[2], row(t, r)});
        return dressed_cache.emplace(id, std::make_pair(o, v)).first->second;
    };
    std::map<Pair, std::array<Mixed, 3>> intermediates;
    auto intermediate = [&](int i, int j) {
        Pair p{i, j};
        auto it = intermediates.find(p);
        if (it != intermediates.end()) return it->second;
        const auto& dom = domain(p);
        int a = pos(dom, i), b = pos(dom, j);
        auto proj = fit.projected(p);
        auto oo = proj[0], h = proj[1], f = proj[2], t = tn(p);
        auto [oi, vi] = dressed(p, i);
        auto [oj, vj] = dressed(p, j);
        auto beta = df(p, "Qk,Ql->kl", {oi, oj}) + df(p, "Qkc,cd,Qld->kl", {h, amp(p), h});
        auto coul = df(p, "Q,Qlc->lc", {auxat(oo, a, b), h}), exch = df(p, "Ql,Qc->lc", {auxrow(oo, a), auxrow(h, b)});
        auto ivvv = df(p, "Qb,Qac->bac", {auxrow(h, a), f}), jvvv = df(p, "Qb,Qac->bac", {auxrow(h, b), f});
        auto gamma = -C("la,lc->ac", {t, coul}) + C("b,bac->ac", {sing(p, j), ivvv});
        auto delta = C("la,lc->ac", {t, coul}) - 2. * C("la,lc->ac", {t, exch}) +
                     2. * C("b,cab->ac", {sing(p, i), jvvv}) - C("b,bca->ac", {sing(p, i), jvvv});
        for (int l : dom) {
            Pair il{i, l}, lj{l, j};
            auto kil = exchange(i, l), llj = spin(exchange(l, j));
            auto column = C("ca,ba,b->c", {ov(p, il), kil, sing(il, j)});
            auto middle = C("ad,de,ec->ac", {amp(lj), ov(lj, il), kil});
            gamma = gamma - C("a,c->ac", {sing(p, l), column}) - .5 * C("xa,ac,cy->xy", {ov(p, lj), middle, ov(il, p)});
            column = C("b,ba,ac->c", {sing(lj, i), llj, ov(lj, p)});
            middle = C("ad,de,ec->ac", {spin(amp(il)), ov(il, lj), llj});
            delta = delta - C("a,c->ac", {sing(p, l), column}) + .5 * C("xa,ac,cy->xy", {ov(p, il), middle, ov(lj, p)});
        }
        return intermediates.emplace(p, std::array<Mixed, 3>{beta, gamma, delta}).first->second;
    };
    std::map<Pair, std::array<Mixed, 3>> partial;
    auto partial_fock = [&](Pair p) {
        auto it = partial.find(p);
        if (it != partial.end()) return it->second;
        auto a = fit.projected(p);
        auto h = a[1], f = a[2], t = tn(p), gram = C("Qmc,mc->Q", {h, t});
        auto bar = [&](const Mixed& block, const Mixed& left) {
            return C("mr,mn,nc->rc", {left, m.f, frame(p)}) + 2. * df(p, "Q,Qrc->rc", {gram, block}) -
                   df(p, "Qre,me,Qmc->rc", {block, t, h});
        };
        return partial
            .emplace(p, std::array<Mixed, 3>{gram, bar(f, frame(p)),
                                             bar(h, m.l.select({range(m.l[0].shape[0]), domain(p)}))})
            .first->second;
    };
    auto foo = C("mi,mn,nj->ij", {m.l, m.f, m.l});
    std::vector<std::vector<Mixed>> fij(no, std::vector<Mixed>(no));
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j) fij[i][j] = at(foo, i, j);
    std::vector<Pair> orders;
    for (auto& [p, t] : doubles) {
        orders.push_back(p);
        if (p.first != p.second) orders.push_back({p.second, p.first});
    }
    for (Pair p : orders) {
        auto [i, j] = p;
        auto a = fit.projected(p);
        auto oo = a[0], h = a[1];
        int x = pos(domain(p), i), y = pos(domain(p), j);
        auto coul = df(p, "Q,Qlc->lc", {auxat(oo, x, y), h}), ex = df(p, "Ql,Qc->lc", {auxrow(oo, y), auxrow(h, x)});
        fij[i][j] = fij[i][j] + C("lc,lc->", {tn(p), 2. * coul - ex});
    }
    std::vector<Mixed> one;
    for (auto& t : singles) one.push_back(0. * t);
    PairMixed two;
    phase("preparation");
    // Jiang 87--90. Scatter destinations are accumulated in a fixed order;
    // they must not be written by an OpenMP loop over source pairs.
    for (Pair p : orders) {
        auto [i, k] = p;
        const auto& dom = domain(p);
        int mi = pos(dom, i), mk = pos(dom, k);
        auto a = fit.projected(p);
        auto oo = a[0], h = a[1], f = a[2], t = tn(p), u = spin(amp(p));
        if (i == k) {
            auto part = partial_fock(p);
            auto gram = part[0], fvv = part[1], fov = part[2], ti = singles[i];
            std::vector<Mixed> fi;
            for (int l : dom) fi.push_back(fij[l][i]);
            auto ai = C("m,mn,na->a", {col(m.l, i), m.f, frame(p)}) + 2. * df(p, "Q,Qa->a", {gram, auxrow(h, mi)}) -
                      df(p, "Qae,me,Qm->a", {f, t, auxrow(oo, mi)}) + C("ab,b->a", {fvv, ti}) -
                      C("ma,mb,b->a", {t, fov, ti}) - C("ma,m->a", {t, stack(fi)});
            one[i] = one[i] + ai;
        }
        auto ovvv = df(p, "Qc,Qda->cda", {auxrow(h, mk), f}), dest = ov(p, {i, i});
        one[i] = one[i] + C("ax,cda,cd->x", {dest, ovvv, u.transpose()});
        auto fk = C("m,mn,na->a", {col(m.l, k), m.f, frame(p)});
        for (int l : dom) {
            Pair kl{k, l};
            auto K = exchange(k, l), s = ov(kl, p);
            auto coefficient = C("ac,cd,bd,ab->", {s, u.transpose(), s, K});
            one[i] = one[i] - C(",a->a", {coefficient, sing({i, i}, l)});
            fk = fk + C("ac,cb,b->a", {ov(p, kl), spin(K), sing(kl, l)});
        }
        one[i] = one[i] + C("ax,ac,c->x", {dest, u, fk});
        auto M = df(p, "Ql,Qc->lc", {auxrow(oo, mi), auxrow(h, mk)});
        auto rr = C("lb,cb->lc", {M + C("lb,bc->lc", {t, exchange(i, k)}), u});
        for (size_t l = 0; l < dom.size(); ++l)
            one[dom[l]] = one[dom[l]] - C("a,ax->x", {row(rr, l), ov(p, {dom[l], dom[l]})});
    }
    phase("singles residual");
    for (int j = 0; j < no; ++j) {
        Pair p{j, j};
        auto corr = C("ma,a->m", {partial_fock(p)[2], singles[j]});
        const auto& dom = domain(p);
        for (size_t k = 0; k < dom.size(); ++k) fij[dom[k]][j] = fij[dom[k]][j] + row(corr, k);
    }
    for (auto [i, j] : orders)
        for (int l : domain({i, j})) {
            auto s = ov({l, i}, {l, j});
            fij[i][j] = fij[i][j] + C("cd,ca,ab,db->", {exchange(l, i), s, spin(amp({l, j})), s});
        }
    phase("occupied Fock");
    std::map<std::tuple<int, int, int>, std::pair<Mixed, Mixed>> integrals;
    auto mixed_integrals = [&](int i, int j, int k) {
        auto key = std::make_tuple(i, j, k);
        auto it = integrals.find(key);
        if (it != integrals.end()) return it->second;
        return integrals.emplace(key, fit.mixed_integrals(i, j, k)).first->second;
    };
    auto ordered_cd = [&](int i, int j) {
        Pair p{i, j};
        auto c = 0. * amp(p), d = c;
        for (int k : domain(p)) {
            Pair ki{k, i}, kj{k, j}, ik{i, k}, jk{j, k};
            auto [J, K] = mixed_integrals(i, j, k);
            auto g = C("xa,ac,cy->xy", {ov(p, ki), intermediate(k, i)[1], ov(ki, kj)});
            c = c - C("ac,bc,by->ay", {J + g, amp(kj), ov(kj, p)});
            g = C("xa,ac,cy->xy", {ov(p, ik), intermediate(i, k)[2], ov(ik, jk)});
            d = d + .5 * C("ac,bc,by->ay", {2. * K - J + g, spin(amp(jk)), ov(jk, p)});
        }
        return .5 * c + c.transpose() + d;
    };
    for (auto& [p, t] : doubles) {
        auto [i, j] = p;
        const auto& dom = domain(p);
        auto a = fit.projected(p);
        auto h = a[1], f = a[2], vv = f - C("ka,Qkc->Qac", {tn(p), h});
        auto result =
            df(p, "Qa,Qb->ab", {dressed(p, i).second, dressed(p, j).second}) + df(p, "Qac,cd,Qbd->ab", {vv, t, vv});
        auto part = partial_fock(p);
        auto virt = part[1] - C("ma,mb->ab", {tn(p), part[2]}), beta = intermediate(i, j)[0];
        for (size_t a = 0; a < dom.size(); ++a)
            for (size_t b = 0; b < dom.size(); ++b) {
                Pair q{dom[a], dom[b]};
                if (!doubles.count(key(q))) continue;
                auto s = ov(p, q);
                result = result + C(",ac,cd,bd->ab", {at(beta, a, b), s, amp(q), s});
                s = ov(q, p);
                virt = virt - C("bx,bd,cd,cy->xy", {s, spin(amp(q)), exchange(q.first, q.second), s});
            }
        result = result + C("ac,bc->ab", {t, virt}) + C("ac,cb->ab", {virt, t}) + ordered_cd(i, j) +
                 ordered_cd(j, i).transpose();
        for (int perm = 0; perm < 2; ++perm) {
            int a = perm ? j : i, b = perm ? i : j;
            auto g = 0. * t;
            for (int k : domain({a, a})) {
                auto s = ov(p, {a, k});
                g = g - C(",uc,cd,vd->uv", {fij[k][b], s, amp({a, k}), s});
            }
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
