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
#include "native_response.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <climits>

namespace psi {
namespace dlpno {
namespace response {
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
inline Tensor from_array(const Array& a) {
    std::vector<int> shape;
    for (int k = 0; k < a.ndim(); ++k) {
        if (a.shape(k) > INT_MAX) throw std::invalid_argument("Native tensor dimension exceeds integer range.");
        shape.push_back(a.shape(k));
    }
    return Tensor(shape, a.data());
}
inline py::array to_array(const Tensor& a) {
    auto* owner = new std::shared_ptr<std::vector<double>>(a.data);
    py::capsule capsule(owner, [](void* p) { delete static_cast<std::shared_ptr<std::vector<double>>*>(p); });
    std::vector<py::ssize_t> shape(a.shape.begin(), a.shape.end());
    py::array out(py::dtype::of<double>(), shape, a.ptr(), capsule);
    out.attr("setflags")(false);
    return out;
}
inline std::array<Tensor, 4> from_blocks(const std::vector<Array>& x) {
    if (x.size() != 4) throw std::invalid_argument("Expected four native mixed blocks.");
    return {from_array(x[0]), from_array(x[1]), from_array(x[2]), from_array(x[3])};
}
inline py::tuple to_blocks(const Mixed& x) {
    py::tuple out(4);
    for (int k = 0; k < 4; ++k) out[k] = to_array(x[k]);
    return out;
}
inline py::tuple python_key(const Pair& p) { return py::make_tuple(p.first, p.second); }
inline py::tuple python_key(const Triplet& p) { return py::make_tuple(p[0], p[1], p[2]); }
template <class Key>
py::dict to_arrays(const std::map<Key, Tensor>& xs) {
    py::dict out;
    for (auto& [p, x] : xs) out[python_key(p)] = to_array(x);
    return out;
}
template <class Key>
std::map<Key, Tensor> from_arrays(const py::dict& xs) {
    std::map<Key, Tensor> out;
    for (auto x : xs) out[py::cast<Key>(x.first)] = from_array(py::cast<Array>(x.second));
    return out;
}
inline py::dict selection(const Selection& s) {
    py::dict out;
    out["frame"] = to_blocks(s.frame);
    out["projector"] = to_blocks(s.projector);
    out["occupations"] = to_array(s.occupations);
    out["boundary_gap"] = s.gap;
    return out;
}
inline py::dict energies(const Energies& stage) {
    py::dict out;
    for (auto& [p, x] : stage) {
        py::dict item;
        item["exchange"] = to_blocks(x.exchange);
        item["amplitudes"] = to_blocks(x.amplitudes);
        item["fock"] = to_blocks(x.fock);
        item["fov"] = to_blocks(x.fov);
        item["occupied_energy"] = to_blocks(x.occupied);
        item["rank"] = x.rank;
        out[python_key(p)] = item;
    }
    return out;
}
struct Entry {
    std::shared_ptr<Workspace> workspace;
    std::shared_ptr<Context> context;
    Molecular molecular;
    std::shared_ptr<Fitting> fitting;
    Entry(std::shared_ptr<Workspace> w, int u, int v, const std::vector<std::vector<Array>>& inputs)
        : workspace(w), context(std::make_shared<Context>(w->cache, "Hessian", u, v)) {
        if (u < 0 || v < 0 || inputs.size() != 6) throw std::invalid_argument("Invalid native Hessian entry inputs.");
        molecular = {context->input("S", from_blocks(inputs[0])), context->input("F", from_blocks(inputs[1])),
                     context->input("L", from_blocks(inputs[2])), context->input("D", from_blocks(inputs[3])),
                     context->input("J", from_blocks(inputs[4])), context->input("Q", from_blocks(inputs[5]))};
    }
    void frames(const std::map<Pair, std::vector<Array>>& frames) {
        PairMixed ys;
        for (auto& [p, x] : frames)
            ys[p] = context->input("PNO:" + std::to_string(p.first) + ":" + std::to_string(p.second), from_blocks(x));
        fitting = std::make_shared<Fitting>(*workspace, molecular, ys);
        fitting->owner = context;
    }
    std::pair<std::vector<Mixed>, PairMixed> amplitudes(const std::vector<std::vector<Array>>& t1,
                                                        const std::map<Pair, std::vector<Array>>& t2,
                                                        const std::string& path) {
        std::vector<Mixed> one;
        PairMixed two;
        for (size_t i = 0; i < t1.size(); ++i)
            one.push_back(context->input("CCSD:" + path + ":t1:" + std::to_string(i), from_blocks(t1[i])));
        for (auto& [p, x] : t2)
            two[p] = context->input("CCSD:" + path + ":t2:" + std::to_string(p.first) + ":" + std::to_string(p.second),
                                    from_blocks(x));
        return {one, two};
    }
};
inline py::dict pno_result(const PNOResult& r) {
    py::dict out, first, normalized, selected;
    for (auto& [p, x] : r.first) {
        py::dict one;
        one["parent"] = to_blocks(x.parent);
        one["exchange"] = to_blocks(x.exchange);
        one["fock"] = to_blocks(x.fock);
        one["amplitudes"] = to_blocks(x.amplitudes);
        one["selection"] = selection(x.selection);
        first[python_key(p)] = one;
    }
    for (auto& [p, x] : r.normalized) normalized[python_key(p)] = to_blocks(x);
    for (auto& [p, x] : r.selection) selected[python_key(p)] = selection(x);
    out["first"] = first;
    out["normalized"] = normalized;
    out["initial"] = energies(r.initial);
    out["final"] = energies(r.final);
    out["selection"] = selected;
    out["source"] = to_arrays(r.source);
    out["residual_errors"] = r.residual_errors;
    out["loss"] = to_blocks(r.loss);
    out["initial_loss"] = to_blocks(r.initial_loss);
    out["final_loss"] = to_blocks(r.final_loss);
    out["stationary_loss"] = r.stationary_loss;
    return out;
}
inline void export_native_response(py::module_& module) {
    py::class_<Workspace, std::shared_ptr<Workspace>>(module, "DLPNOResponseWorkspace")
        .def(py::init([](Domains domains, Domains first, py::dict pairs, py::dict triples, std::vector<int> centers,
                         py::dict adjoint, double energy, size_t bytes) {
            auto w = std::make_shared<Workspace>(bytes);
            w->domains = std::move(domains);
            w->first_domains = std::move(first);
            w->auxiliary_centers = std::move(centers);
            w->pno_adjoint = from_arrays<Pair>(adjoint);
            w->pno_reference = energy;
            for (auto item : pairs) {
                Pair p = py::cast<Pair>(item.first);
                auto x = py::cast<py::dict>(item.second);
                PairData d;
                d.occupied = py::cast<std::vector<int>>(x["occupied"]);
                d.paos = py::cast<std::vector<int>>(x["paos"]);
                d.auxiliary = py::cast<std::vector<int>>(x["auxiliary"]);
                d.first_auxiliary =
                    x.contains("first_auxiliary") ? py::cast<std::vector<int>>(x["first_auxiliary"]) : d.auxiliary;
                d.edges = py::cast<std::vector<std::array<int, 4>>>(x["edges"]);
                d.first_parent = from_array(py::cast<Array>(x["parent"]));
                d.first_frame = from_array(py::cast<Array>(x["first_frame"]));
                d.final_frame = from_array(py::cast<Array>(x["final_frame"]));
                d.first_amplitude = from_array(py::cast<Array>(x["first_amplitude"]));
                d.lmp2_amplitude = from_array(py::cast<Array>(x["lmp2_amplitude"]));
                d.qi = from_array(py::cast<Array>(x["qi"]));
                d.qj = from_array(py::cast<Array>(x["qj"]));
                d.occupations = from_array(py::cast<Array>(x["occupations"]));
                w->pairs.emplace(p, std::move(d));
            }
            for (auto item : triples) {
                auto r = py::cast<Triplet>(item.first);
                auto x = py::cast<py::dict>(item.second);
                TripletData d;
                d.occupied = py::cast<std::vector<int>>(x["occupied"]);
                d.paos = py::cast<std::vector<int>>(x["paos"]);
                d.auxiliary = py::cast<std::vector<int>>(x["auxiliary"]);
                d.edges = py::cast<std::vector<std::array<int, 6>>>(x["edges"]);
                d.parent = from_array(py::cast<Array>(x["parent"]));
                d.frame = from_array(py::cast<Array>(x["frame"]));
                d.t = from_array(py::cast<Array>(x["t"]));
                d.z = from_array(py::cast<Array>(x["z"]));
                d.w = from_array(py::cast<Array>(x["w"]));
                d.v = from_array(py::cast<Array>(x["v"]));
                w->triplets.emplace(r, std::move(d));
            }
            return w;
        }))
        .def("entry", [](std::shared_ptr<Workspace> w, int u, int v,
                         std::vector<std::vector<Array>> input) { return std::make_shared<Entry>(w, u, v, input); })
        .def("screened_snapshot",
             [](Workspace& w, Domains domains, py::dict state) {
                 w.screened_domains = std::move(domains);
                 for (auto item : state) {
                     auto p = py::cast<Pair>(item.first);
                     auto blocks = py::cast<py::dict>(item.second);
                     for (auto x : blocks)
                         w.screened_state[p][py::cast<std::string>(x.first)] = from_array(py::cast<Array>(x.second));
                 }
             })
        .def_property_readonly("statistics", [](const Workspace& w) { return w.cache->statistics(); })
        .def_property_readonly("storage_statistics", [](const Workspace& w) { return w.cache->storage_statistics(); })
        .def_property_readonly("lmp2_counts",
                               [](const Workspace& w) { return std::array<size_t, 2>{w.lmp2_solves, w.lmp2_hits}; });
    py::class_<Entry, std::shared_ptr<Entry>>(module, "DLPNOResponseEntry")
        .def("set_frames", &Entry::frames)
        .def_property_readonly("ccsd_profile",
                               [](const Entry& e) {
                                   if (!e.fitting) throw std::runtime_error("Missing native fitting state.");
                                   return e.fitting->ccsd_seconds;
                               })
        .def_property_readonly("panel_statistics",
                               [](const Entry& e) {
                                   if (!e.fitting) throw std::runtime_error("Missing native fitting state.");
                                   auto c = e.fitting->panel_counts;
                                   return std::array<size_t, 7>{
                                       c[0], c[1], c[2], c[3], c[4], c[5], e.fitting->panel_bytes};
                               })
        .def("pno",
             [](Entry& e, py::function first, py::function mixed) {
                 PNOResult r;
                 auto first_solver = [&](const PairTensor& dy, const PairTensor& dk, const Tensor& s, const Tensor& f,
                                         const Tensor& foo) {
                     py::gil_scoped_acquire acquire;
                     return from_arrays<Pair>(
                         first(to_arrays(dy), to_arrays(dk), to_array(s), to_array(f), to_array(foo)).cast<py::dict>());
                 };
                 auto mixed_solver = [&](const PairTensor& source) {
                     py::gil_scoped_acquire acquire;
                     return from_arrays<Pair>(mixed(to_arrays(source)).cast<py::dict>());
                 };
                 {
                     py::gil_scoped_release release;
                     r = e.workspace->pno(e.molecular, first_solver, mixed_solver);
                     PairMixed frames;
                     for (auto& [p, x] : r.selection) frames[p] = x.frame;
                     e.fitting = std::make_shared<Fitting>(*e.workspace, e.molecular, frames);
                     e.fitting->owner = e.context;
                 }
                 return pno_result(r);
             })
        .def("ccsd",
             [](Entry& e, const std::vector<std::vector<Array>>& one, const std::map<Pair, std::vector<Array>>& two,
                const std::string& path) {
                 auto amps = e.amplitudes(one, two, path);
                 if (!e.fitting) throw std::runtime_error("PNO frames have not been supplied.");
                 CCSDResult r;
                 {
                     py::gil_scoped_release release;
                     r = ccsd_residual(*e.fitting, amps.first, amps.second);
                 }
                 py::list a;
                 py::dict b;
                 for (auto& x : r.first) a.append(to_blocks(x));
                 for (auto& [p, x] : r.second) b[python_key(p)] = to_blocks(x);
                 return py::make_tuple(a, b);
             })
        .def("retained",
             [](Entry& e, const std::vector<std::vector<Array>>& one, const std::map<Pair, std::vector<Array>>& two) {
                 auto amps = e.amplitudes(one, two, "affine");
                 if (!e.fitting) throw std::runtime_error("Missing native fitting state.");
                 Mixed value;
                 {
                     py::gil_scoped_release release;
                     value = retained_energy(*e.fitting, amps.first, amps.second);
                 }
                 return to_blocks(value);
             })
        .def("screened",
             [](Entry& e, const std::vector<std::vector<Array>>& operators) {
                 if (operators.size() != 3) throw std::invalid_argument("Expected three AO dipole components.");
                 std::vector<Mixed> ops;
                 for (size_t i = 0; i < operators.size(); ++i)
                     ops.push_back(e.context->input("dipole:" + std::to_string(i), from_blocks(operators[i])));
                 std::pair<Mixed, Mixed> value;
                 {
                     py::gil_scoped_release release;
                     value = screened_energy(*e.workspace, e.molecular, ops);
                 }
                 return py::make_tuple(to_blocks(value.first), to_blocks(value.second));
             })
        .def("projected",
             [](Entry& e, Pair p) {
                 if (!e.fitting) throw std::runtime_error("Missing native fitting state.");
                 std::array<Mixed, 5> r;
                 {
                     py::gil_scoped_release release;
                     r = e.fitting->projected(p);
                 }
                 py::tuple out(6);
                 for (int k = 0; k < 5; ++k) out[k] = to_blocks(r[k]);
                 out[5] = py::none();
                 return out;
             })
        .def(
            "triples",
            [](Entry& e, const std::vector<std::vector<Array>>& one, const std::map<Pair, std::vector<Array>>& two) {
                if (!e.fitting) throw std::runtime_error("Missing native fitting state.");
                auto amps = e.amplitudes(one, two, "affine");
                py::gil_scoped_release release;
                return std::make_shared<Triples>(e.fitting, amps.first, amps.second);
            },
            py::keep_alive<0, 1>());
    py::class_<Triples, std::shared_ptr<Triples>>(module, "DLPNOTriplesInputs")
        .def("frame",
             [](Triples& t, Triplet r) {
                 Mixed x;
                 {
                     py::gil_scoped_release release;
                     x = t.frame(r);
                 }
                 return to_blocks(x);
             })
        .def("blocks",
             [](Triples& t, Triplet r) {
                 std::array<Mixed, 3> x;
                 {
                     py::gil_scoped_release release;
                     x = t.blocks(r);
                 }
                 return py::make_tuple(to_blocks(x[0]), to_blocks(x[1]), to_blocks(x[2]));
             })
        .def("edge",
             [](Triples& t, Triplet r, Triplet q) {
                 Mixed x;
                 {
                     py::gil_scoped_release release;
                     x = t.edge(r, q);
                 }
                 return to_blocks(x);
             })
        .def("occupied", [](Triples& t, int i, int j) { return to_blocks(at(t.foo, i, j)); })
        .def("check_frames",
             [](Triples& t, const TripleDirection& d, int k) {
                 py::gil_scoped_release release;
                 for (auto& [r, x] : d.frames) close(t.frame(r)[k], x, 2.e-8, "First/mixed TNO chart consistency");
             })
        .def("direction", [](Triples& t, int k, double tolerance, double energy) {
            py::gil_scoped_release release;
            return t.direction(k, tolerance, energy);
        });
    py::class_<TripleDirection, std::shared_ptr<TripleDirection>>(module, "DLPNOTriplesDirection")
        .def("apply",
             [](const TripleDirection& d, const Array& x, bool left) {
                 Tensor input = from_array(x), out;
                 {
                     py::gil_scoped_release release;
                     out = d.apply(input, left);
                 }
                 return to_array(out);
             })
        .def("unpack", [](const TripleDirection& d, const Array& x) { return to_arrays(d.unpack(from_array(x))); })
        .def_property_readonly(
            "bytes",
            [](const TripleDirection& d) {
                size_t words = 0;
                for (auto* blocks : {&d.frames, &d.t, &d.z, &d.gaps, &d.seeds, &d.dseeds, &d.right, &d.left})
                    for (auto& [r, x] : *blocks) words += x.size();
                for (auto& [r, es] : d.edges)
                    for (auto& e : es) words += e.overlap.size() + e.doverlap.size();
                return words * sizeof(double);
            })
        .def_property_readonly("data",
                               [](const TripleDirection& d) {
                                   py::dict x;
                                   x["frame_tangents"] = to_arrays(d.frames);
                                   x["amplitudes"] = to_arrays(d.t);
                                   x["adjoints"] = to_arrays(d.z);
                                   x["gaps"] = to_arrays(d.gaps);
                                   x["energy_seeds"] = to_arrays(d.seeds);
                                   x["energy_seed_tangents"] = to_arrays(d.dseeds);
                                   x["right_source"] = to_arrays(d.right);
                                   x["left_source"] = to_arrays(d.left);
                                   py::dict es;
                                   for (auto& [r, edges] : d.edges) {
                                       py::list row;
                                       for (auto& e : edges)
                                           row.append(py::make_tuple(e.source, e.axes, e.occupied, e.fock, e.dfock,
                                                                     to_array(e.overlap), to_array(e.doverlap)));
                                       es[python_key(r)] = row;
                                   }
                                   x["edges"] = es;
                                   return x;
                               })
        .def_property_readonly("diagonal", [](const TripleDirection& d) { return to_array(d.pack(d.gaps)); })
        .def_property_readonly("right", [](const TripleDirection& d) { return to_array(d.pack(d.right)); })
        .def_property_readonly("left", [](const TripleDirection& d) { return to_array(d.pack(d.left)); })
        .def_property_readonly("frames", [](const TripleDirection& d) { return to_arrays(d.frames); })
        .def_property_readonly("energy_tangent", &TripleDirection::energy_tangent)
        .def("relaxed_energy",
             [](const TripleDirection& d, const Array& dt) {
                 auto x = d.unpack(from_array(dt));
                 double e = 0.;
                 for (auto& [r, t] : d.t) e += dot(d.dseeds.at(r), t) + dot(d.seeds.at(r), x.at(r));
                 return e;
             })
        .def("column", [](std::shared_ptr<TripleDirection> d, const Array& t, const Array& z) {
            return std::make_shared<TripleColumn>(d, from_array(t), from_array(z));
        });
    py::class_<TripleColumn, std::shared_ptr<TripleColumn>>(module, "DLPNOTriplesColumn")
        .def("element", [](TripleColumn& c, const TripleDirection& row, Triples& t) {
            py::gil_scoped_release release;
            return c.element(row, t);
        });
    module.def("dlpno_response_weighted_dot", [](const std::vector<Array>& one, const py::dict& two,
                                                 const std::vector<Array>& other, const py::dict& other_two) {
        if (one.size() != other.size() || two.size() != other_two.size())
            throw std::invalid_argument("CCSD contraction graph mismatch.");
        auto a = from_arrays<Pair>(two), b = from_arrays<Pair>(other_two);
        double value = 0.;
        for (size_t i = 0; i < one.size(); ++i) value += dot(from_array(one[i]), from_array(other[i]));
        for (auto& [p, x] : a) value += (p.first == p.second ? .5 : 1.) * dot(x, b.at(p));
        return value;
    });
    // Public diagnostic boundary: exercises the same compiled tensor/subspace
    // code used by the production stages against independent Python oracles.
    module.def(
        "dlpno_native_response_test",
        [](const std::string& op, std::vector<std::vector<Array>> inputs, int rank) {
            auto cache = std::make_shared<Cache>(0);
            Context c(cache, "test", 0, 1);
            std::vector<Mixed> x;
            for (size_t i = 0; i < inputs.size(); ++i) x.push_back(c.input(std::to_string(i), from_blocks(inputs[i])));
            Mixed out;
            {
                py::gil_scoped_release release;
                if (op == "solve")
                    out = metric_solve(x.at(0), x.at(1));
                else if (op == "inverse")
                    out = symmetric_inverse(x.at(0));
                else if (op == "filtered")
                    out = filtered_inverse(x.at(0), symmetric_inverse(x.at(0)));
                else if (op == "projector")
                    out = projector(x.at(0), rank);
                else if (op == "polar")
                    out = polar(x.at(0));
                else
                    out = product(op, x);
            }
            return to_blocks(out);
        },
        py::arg("operation"), py::arg("inputs"), py::arg("rank") = 0);
}
}  // namespace response
}  // namespace dlpno
}  // namespace psi
