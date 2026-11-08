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

#pragma once

#include "psi4/libmints/matrix.h"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <list>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <set>

namespace psi {
namespace dlpno {
namespace response {

// Dense local tensors own immutable, contiguous row-major storage. Reshapes
// share storage; packing, slicing and contractions are performed natively.
struct Tensor {
    std::vector<int> shape;
    SharedMatrix data;
    Tensor();
    explicit Tensor(std::vector<int> dimensions, double value = 0.);
    Tensor(std::vector<int> dimensions, const double* values);
    size_t size() const;
    const double* ptr() const { return data->pointer()[0]; }
    double* mutable_ptr() { return data->pointer()[0]; }
    const double* begin() const { return ptr(); }
    const double* end() const { return ptr() + size(); }
    double* begin() { return mutable_ptr(); }
    double* end() { return mutable_ptr() + size(); }
    double scalar() const;
    Tensor reshape(std::vector<int> dimensions) const;
    Tensor permute(const std::vector<int>& axes) const;
    Tensor select(const std::vector<std::vector<int>>& indices, const std::vector<int>& drop = {}) const;
};
Tensor add(const Tensor&, const Tensor&, double factor = 1.);
Tensor scale(const Tensor&, double);
Tensor contract(const std::string&, const std::vector<Tensor>&);
Tensor doublet(const Tensor&, const Tensor&, bool trans_a = false, bool trans_b = false);
Tensor solve(const Tensor&, const Tensor&);
std::pair<Tensor, Tensor> eigh(const Tensor&);
Tensor eye(int n);
double dot(const Tensor&, const Tensor&);
void close(const Tensor&, const Tensor&, double tolerance, const std::string& label, double relative = 0.);

struct Mixed;
class Cache {
    struct Entry {
        Tensor tensor;
        std::list<std::pair<size_t, int>>::iterator order;
    };
    std::map<std::string, size_t> nodes_;
    std::map<std::pair<size_t, int>, uint64_t> input_hashes_;
    std::map<std::pair<size_t, int>, Entry> values_;
    std::map<std::pair<size_t, int>, Tensor> persistent_;
    std::list<std::pair<size_t, int>> order_;
    mutable std::mutex mutex_;
    size_t bytes_ = 0, limit_ = 0, hits_ = 0, evaluations_ = 0, next_node_ = 0;
    size_t persistent_bytes_ = 0, evictions_ = 0, invalidations_ = 0, graph_resets_ = 0;

   public:
    explicit Cache(size_t bytes) : limit_(bytes) {}
    size_t node(const std::string& key);
    Tensor evaluate(size_t node, int coordinate, const std::function<Tensor()>& build);
    void check_input(size_t node, int coordinate, const Tensor& value);
    void retain(size_t node, int coordinate, const Tensor& value);
    std::array<size_t, 4> statistics() const;
    std::array<size_t, 5> storage_statistics() const;
    size_t limit() const { return limit_; }
};

// Each stage/row has a stable namespace; reference and first derivatives of
// intermediates may then be shared across all ordered Hessian entries. Mixed
// derivatives are ephemeral. Every cache is bounded and reference-owned.
struct Context {
    std::shared_ptr<Cache> cache;
    std::string scope;
    int u = -1, v = -1;
    Context(std::shared_ptr<Cache> c, std::string name, int row, int column)
        : cache(std::move(c)), scope(std::move(name)), u(row), v(column) {}
    Mixed input(const std::string&, const std::array<Tensor, 4>&) const;
    Mixed constant(const std::string&, const Tensor&) const;
    Mixed evaluate(const std::string&, const std::vector<size_t>&, const std::function<Tensor(int)>&) const;
    void retain(const Mixed&) const;
};
struct Mixed {
    std::array<Tensor, 4> x;
    size_t id = 0;
    const Context* context = nullptr;
    const Tensor& operator[](int k) const { return x.at(k); }
    Mixed permute(const std::vector<int>& axes) const;
    Mixed reshape(std::vector<int> dimensions) const;
    Mixed transpose() const { return permute({1, 0}); }
    Mixed select(const std::vector<std::vector<int>>& indices, const std::vector<int>& drop = {}) const;
};
Mixed sum(const std::vector<Mixed>&);
Mixed scaled(const Mixed&, double);
Mixed product(const std::string&, const std::vector<Mixed>&);
Mixed doublet(const Mixed&, const Mixed&, bool trans_a = false, bool trans_b = false);
// Explicit two-operand matrix stages, matching the energy-code layouts.
Mixed auxiliary_transform(const Mixed& metric, const Mixed& factors);
Mixed virtual_transform(const Mixed& factors, const Mixed& left, const Mixed& right);
Mixed congruence(const Mixed& left, const Mixed& matrix, const Mixed& right);
Mixed inner(const Mixed&, const Mixed&);
Mixed stack(const std::vector<Mixed>&);
inline Mixed operator+(const Mixed& a, const Mixed& b) { return sum({a, b}); }
inline Mixed operator-(const Mixed& a, const Mixed& b) { return sum({a, scaled(b, -1.)}); }
inline Mixed operator-(const Mixed& a) { return scaled(a, -1.); }
inline Mixed operator*(double a, const Mixed& b) { return scaled(b, a); }
inline Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
inline Tensor operator-(const Tensor& a, const Tensor& b) { return add(a, b, -1.); }
inline Tensor operator-(const Tensor& a) { return scale(a, -1.); }
inline Tensor operator*(double a, const Tensor& b) { return scale(b, a); }
inline Tensor mat(const Tensor& a, const Tensor& b) { return doublet(a, b); }
inline Tensor tr(const Tensor& a) { return a.permute({1, 0}); }
inline Tensor sym(const Tensor& a) { return .5 * (a + tr(a)); }
inline Mixed mat(const Mixed& a, const Mixed& b) { return doublet(a, b); }
inline Mixed spin(const Mixed& a) { return 2. * a - a.transpose(); }
Mixed metric_solve(const Mixed&, const Mixed&);
Mixed symmetric_inverse(const Mixed&);
Mixed filtered_inverse(const Mixed&, const Mixed&);
Mixed polar(const Mixed&);
std::vector<int> range(int n);
void parallel_rows(size_t count, const std::function<void(size_t)>&);

using Pair = std::pair<int, int>;
using Triplet = std::array<int, 3>;
using PairMixed = std::map<Pair, Mixed>;
using PairTensor = std::map<Pair, Tensor>;
using Domains = std::map<std::string, std::vector<std::vector<int>>>;
struct PairData {
    std::vector<int> occupied, paos, auxiliary, first_auxiliary;
    Tensor first_parent, first_frame, final_frame, first_amplitude, lmp2_amplitude;
    Tensor qi, qj, occupations;
    std::vector<std::array<int, 4>> edges;
};
struct TripletData {
    std::vector<int> occupied, paos, auxiliary;
    Tensor parent, frame, t, z, w, v;
    std::vector<std::array<int, 6>> edges;
};
struct Molecular {
    Mixed s, f, l, d, j, q;
};
struct Selection {
    Mixed frame, projector;
    Tensor occupations;
    double gap;
};
struct FirstPNO {
    Mixed parent, exchange, fock, amplitudes;
    Selection selection;
};
struct PairEnergy {
    Mixed exchange, amplitudes, fock, fov, occupied;
    int rank = 0;
};
using Energies = std::map<Pair, PairEnergy>;
Mixed projector(const Mixed&, int);
Mixed semicanonical(const Mixed&, const Mixed&, const Mixed&);
Selection pno_selection(const Mixed&, const Mixed&, const Mixed&, const Tensor&, bool);
std::pair<Mixed, Mixed> moving_parent(const Mixed&, const Mixed&, const Tensor&);
Mixed paos(const Molecular&, const std::vector<int>&);
Mixed pno_loss(const PairEnergy&, bool);
Mixed pno_energy(const Energies&, const Energies&);
Mixed tno_frame(const Triplet&, const Mixed&, const Mixed&, const PairMixed&, const PairMixed&, const Tensor&);

struct PNOResult {
    std::map<Pair, FirstPNO> first;
    PairMixed normalized;
    Energies initial, final;
    std::map<Pair, Selection> selection;
    PairTensor source;
    std::array<double, 4> residual_errors{};
    Mixed loss, initial_loss, final_loss;
    double stationary_loss = 0.;
};
using FirstSolver =
    std::function<PairTensor(const PairTensor&, const PairTensor&, const Tensor&, const Tensor&, const Tensor&)>;
using MixedSolver = std::function<PairTensor(const PairTensor&)>;

class Fitting;
class Workspace {
   public:
    Domains domains, first_domains, screened_domains;
    std::map<Pair, std::map<std::string, Tensor>> screened_state;
    std::map<Pair, PairData> pairs;
    std::map<Triplet, TripletData> triplets;
    std::vector<int> auxiliary_centers;
    std::shared_ptr<Cache> cache;
    PairTensor pno_adjoint;
    double pno_reference = 0.;
    size_t lmp2_solves = 0, lmp2_hits = 0;
    // First amplitudes are small relative to the integral cache; callers
    // reserve these in the dense Hessian memory preflight.
    std::map<int, PairTensor> lmp2_first;
    Workspace(size_t bytes) : cache(std::make_shared<Cache>(bytes)) {}
    PNOResult pno(const Molecular&, const FirstSolver&, const MixedSolver&, const Fitting* common = nullptr);
};

class Fitting {
   public:
    const Workspace& workspace;
    Molecular molecular;
    PairMixed frames;
    std::shared_ptr<Context> owner;
    mutable std::map<std::vector<int>, Mixed> pao_cache;
    mutable std::map<Pair, std::array<Mixed, 5>> projected_cache;
    mutable std::map<Pair, Mixed> coefficient_cache;
    struct RawPanel {
        Mixed values;
        std::vector<int> auxiliary, left, right;
    };
    // Atom-local qij/qia/qab jets, prepared before pair/triplet work. Their
    // AO row lists and screens are the captured forward domains, never unions
    // of independently fitted PNO spaces.
    mutable std::map<std::pair<std::string, int>, RawPanel> atomic_panels;
    mutable std::set<std::string> prepared_raw;
    mutable std::array<size_t, 3> atomic_bytes{};
    mutable std::map<std::tuple<Pair, int, bool>, Mixed> weighted_panels;
    mutable size_t weighted_bytes = 0;
    mutable std::map<std::tuple<int, int, int>, std::pair<Mixed, Mixed>> mixed_vertices;
    // These are stage-owned panels, including the mixed component. They do
    // not compete with transient DAG nodes in the cross-entry LRU. The fixed
    // per-workspace allowance also bounds task-private copies in triples.
    using RawKey = std::tuple<std::string, std::vector<int>, std::vector<int>, std::vector<int>>;
    mutable std::map<RawKey, Mixed> raw_panels;
    mutable std::map<std::pair<int, std::vector<int>>, Mixed> refit_panels;
    mutable std::map<std::pair<Pair, Pair>, Mixed> overlap_panels;
    mutable size_t panel_bytes = 0;
    mutable std::array<size_t, 6> panel_counts{};  // raw/refit/overlap builds,hits
    mutable std::map<std::string, double> ccsd_seconds;
    bool keep_panel(const Mixed&) const;
    Fitting(const Workspace& w, Molecular m, PairMixed y)
        : workspace(w), molecular(std::move(m)), frames(std::move(y)) {}
    Mixed pao(const std::vector<int>&) const;
    Mixed refit(int atom, const std::vector<int>&) const;
    Mixed coefficients(const std::vector<int>&, const Tensor&, const Mixed&) const;
    Mixed raw(const std::string&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&) const;
    Mixed raw_direct(const std::string&, const std::vector<int>&, const std::vector<int>&,
                     const std::vector<int>&) const;
    void prepare_raw(const std::vector<std::string>& kinds = {"oo", "ov", "vv"}) const;
    Mixed metric(const std::vector<int>&) const;
    Mixed overlap(Pair, Pair) const;
    const std::array<Mixed, 5>& projected(Pair) const;
    Mixed weighted(Pair, int panel, bool filtered = true) const;
    std::pair<Mixed, Mixed> mixed_integrals(int, int, int) const;
};
Mixed retained_energy(const Fitting&, const std::vector<Mixed>&, const PairMixed&);
std::pair<Mixed, Mixed> screened_energy(const Workspace&, const Molecular&, const std::vector<Mixed>&);
using CCSDResult = std::pair<std::vector<Mixed>, PairMixed>;
CCSDResult ccsd_residual(const Fitting&, const std::vector<Mixed>&, const PairMixed&);
inline Pair key(Pair p) { return {std::min(p.first, p.second), std::max(p.first, p.second)}; }
inline int pos(const std::vector<int>& a, int i) {
    auto p = std::find(a.begin(), a.end(), i);
    if (p == a.end()) throw std::invalid_argument("Missing occupied label in native response domain.");
    return p - a.begin();
}
inline Mixed row(const Mixed& a, int i) {
    std::vector<std::vector<int>> s;
    for (int n : a[0].shape) s.push_back(range(n));
    s[0] = {i};
    return a.select(s, {0});
}
inline Mixed col(const Mixed& a, int i) { return a.select({range(a[0].shape[0]), {i}}, {1}); }
inline Mixed at(const Mixed& a, int i, int j) { return a.select({{i}, {j}}, {0, 1}); }
inline Mixed auxrow(const Mixed& a, int i) { return a.select({range(a[0].shape[0]), {i}, range(a[0].shape[2])}, {1}); }
inline Mixed auxat(const Mixed& a, int i, int j) { return a.select({range(a[0].shape[0]), {i}, {j}}, {1, 2}); }
std::array<Mixed, 3> triples_moments(const Triplet&, const std::vector<int>&, const Mixed&, const Mixed&, const Mixed&,
                                     const Mixed&, const std::function<Mixed(int, int)>&,
                                     const std::function<Mixed(int)>&, const std::function<Mixed(int, int)>&,
                                     const std::function<Mixed(int)>&, const Mixed&);

struct TripleEdge {
    Triplet source;
    std::vector<int> axes;
    Pair occupied;
    double fock = 0., dfock = 0.;
    Tensor overlap, doverlap;
};
struct TripleDirection {
    std::map<Triplet, Tensor> frames, t, z, gaps, seeds, dseeds, right, left;
    std::map<Triplet, std::vector<TripleEdge>> edges;
    Tensor apply(const Tensor&, bool) const;
    Tensor pack(const std::map<Triplet, Tensor>&) const;
    std::map<Triplet, Tensor> unpack(const Tensor&) const;
    double energy_tangent() const;
};
class Triples {
   public:
    std::shared_ptr<Fitting> fit;
    std::vector<Mixed> singles;
    PairMixed doubles;
    std::map<Triplet, Mixed> parents, frames;
    std::map<Triplet, std::array<Mixed, 3>> moments;
    std::map<std::pair<Triplet, Triplet>, Mixed> overlaps;
    Mixed foo;
    Triples(std::shared_ptr<Fitting>, std::vector<Mixed>, PairMixed);
    void build_all();
    Mixed frame(const Triplet&);
    std::array<Mixed, 3> blocks(const Triplet&);
    Mixed edge(const Triplet&, const Triplet&);
    std::shared_ptr<TripleDirection> direction(int, double, double);
};
struct TripleColumn {
    std::shared_ptr<TripleDirection> direction;
    std::map<Triplet, Tensor> dt, dz;
    struct Covector {
        Tensor bar, column_bar;
        double value, column;
    };
    std::map<std::pair<Triplet, int>, Covector> covectors;
    TripleColumn(std::shared_ptr<TripleDirection> d, const Tensor& t, const Tensor& z)
        : direction(d), dt(d->unpack(t)), dz(d->unpack(z)) {}
    double element(const TripleDirection&, Triples&);
};
}  // namespace response
}  // namespace dlpno
}  // namespace psi
