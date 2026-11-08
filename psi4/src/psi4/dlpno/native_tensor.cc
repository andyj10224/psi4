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
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/process.h"
#include <exception>
#ifdef USING_LAPACK_MKL
#include <mkl.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {
namespace dlpno {
namespace response {
namespace {
size_t extent(const std::vector<int>& s) {
    size_t n = 1;
    for (int x : s) {
        if (x < 0 || (x && n > std::numeric_limits<size_t>::max() / size_t(x)))
            throw std::invalid_argument("Invalid native response tensor dimensions.");
        n *= x;
    }
    return n;
}
std::vector<size_t> strides(const std::vector<int>& s) {
    std::vector<size_t> out(s.size(), 1);
    for (int k = int(s.size()) - 2; k >= 0; --k) out[k] = out[k + 1] * s[k + 1];
    return out;
}
std::string numbers(const std::vector<int>& x) {
    std::string s;
    for (int i : x) s += std::to_string(i) + ",";
    return s;
}
void finite(const Tensor& x) {
    for (double v : *x.data)
        if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite native response tensor.");
}
std::vector<std::string> split(const std::string& s, char separator) {
    std::vector<std::string> out(1);
    for (char c : s) {
        if (c == separator)
            out.emplace_back();
        else
            out.back() += c;
    }
    return out;
}
Tensor binary(const std::string& a, const std::string& b, const std::string& out, const Tensor& x, const Tensor& y) {
    if (a.size() != x.shape.size() || b.size() != y.shape.size())
        throw std::invalid_argument("Native contraction rank mismatch.");
    std::map<char, int> dims;
    auto dimensions = [&](const std::string& labels, const Tensor& t) {
        for (size_t k = 0; k < labels.size(); ++k) {
            auto found = dims.find(labels[k]);
            if (found != dims.end() && found->second != t.shape[k])
                throw std::invalid_argument("Native contraction dimension mismatch.");
            dims[labels[k]] = t.shape[k];
        }
    };
    dimensions(a, x);
    dimensions(b, y);
    std::vector<int> shape;
    std::set<char> output;
    for (char c : out) {
        if (!dims.count(c) || !output.insert(c).second)
            throw std::invalid_argument("Invalid native contraction output.");
        shape.push_back(dims[c]);
    }
    if ((a == b && a == out) || (a.empty() && b == out) || (b.empty() && a == out) ||
        (a == out && b.size() <= a.size() && a.substr(a.size() - b.size()) == b) ||
        (b == out && a.size() <= b.size() && b.substr(b.size() - a.size()) == a)) {
        Tensor result(shape);
        size_t count = result.size(), nx = x.size(), ny = y.size();
#pragma omp parallel for schedule(static) if (count >= 131072)
        for (size_t i = 0; i < count; ++i) result.mutable_ptr()[i] = x.ptr()[i % nx] * y.ptr()[i % ny];
        return result;
    }
    if (std::all_of(x.data->begin(), x.data->end(), [](double v) { return v == 0.; }) ||
        std::all_of(y.data->begin(), y.data->end(), [](double v) { return v == 0.; }))
        return Tensor(shape);
    std::string left, right, inner;
    bool gemm =
        std::set<char>(a.begin(), a.end()).size() == a.size() && std::set<char>(b.begin(), b.end()).size() == b.size();
    for (char c : a) {
        bool common = b.find(c) != std::string::npos;
        if (common && !output.count(c))
            inner += c;
        else if (!common && output.count(c))
            left += c;
        else
            gemm = false;
    }
    for (char c : b)
        if (a.find(c) == std::string::npos) {
            if (output.count(c))
                right += c;
            else
                gemm = false;
        }
    if (gemm) {
        auto axes = [](const std::string& from, const std::string& to) {
            std::vector<int> ids;
            for (char c : to) ids.push_back(int(from.find(c)));
            return ids;
        };
        auto count = [&](const std::string& labels) {
            size_t n = 1;
            for (char c : labels) n *= dims[c];
            if (n > size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Native response BLAS dimension overflow.");
            return int(n);
        };
        const int m = count(left), k = count(inner), n = count(right);
        auto ax = axes(a, left + inner), by = axes(b, inner + right);
        const bool ta = x.shape.size() == 2 && left.size() == 1 && inner.size() == 1 && ax == std::vector<int>{1, 0};
        const bool tb = y.shape.size() == 2 && right.size() == 1 && inner.size() == 1 && by == std::vector<int>{1, 0};
        auto aa = ta ? x : x.permute(ax), bb = tb ? y : y.permute(by);
        Tensor result({m, n});
        if (m && n && k)
            C_DGEMM(ta ? 'T' : 'N', tb ? 'T' : 'N', m, n, k, 1., const_cast<double*>(aa.ptr()), ta ? m : k,
                    const_cast<double*>(bb.ptr()), tb ? k : n, 0., result.mutable_ptr(), n);
        std::vector<int> packed_shape;
        for (char c : left + right) packed_shape.push_back(dims[c]);
        return result.reshape(packed_shape).permute(axes(left + right, out));
    }
    // Batched free indices, diagonals and elementwise products. Output cells
    // own their complete reduction; no shared reductions or BLAS in this loop.
    std::string reduced;
    for (const auto& [c, n] : dims)
        if (!output.count(c)) reduced += c;
    std::vector<int> red_shape;
    for (char c : reduced) red_shape.push_back(dims[c]);
    const auto os = strides(shape), rs = strides(red_shape), xs = strides(x.shape), ys = strides(y.shape);
    Tensor result(shape);
    const size_t nr = extent(red_shape), total = result.size();
#pragma omp parallel for schedule(static) if (total * nr >= 1048576)
    for (size_t q = 0; q < total; ++q) {
        std::array<size_t, 256> indices{};
        for (size_t i = 0; i < out.size(); ++i) indices[static_cast<unsigned char>(out[i])] = (q / os[i]) % shape[i];
        double value = 0.;
        for (size_t r = 0; r < nr; ++r) {
            for (size_t i = 0; i < reduced.size(); ++i)
                indices[static_cast<unsigned char>(reduced[i])] = (r / rs[i]) % red_shape[i];
            size_t ix = 0, iy = 0;
            for (size_t i = 0; i < a.size(); ++i) ix += indices[static_cast<unsigned char>(a[i])] * xs[i];
            for (size_t i = 0; i < b.size(); ++i) iy += indices[static_cast<unsigned char>(b[i])] * ys[i];
            value += x.ptr()[ix] * y.ptr()[iy];
        }
        result.mutable_ptr()[q] = value;
    }
    return result;
}
struct Step {
    size_t i, j;
    std::string a, b, out;
};
std::vector<Step> plan(const std::string& equation, const std::vector<std::vector<int>>& shapes) {
    auto arrow = equation.find("->");
    if (arrow == std::string::npos)
        throw std::invalid_argument("Native response requires explicit contraction outputs.");
    auto labels = split(equation.substr(0, arrow), ',');
    auto output = equation.substr(arrow + 2);
    if (labels.size() != shapes.size()) throw std::invalid_argument("Native contraction operand count mismatch.");
    std::map<char, int> dims;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i].size() != shapes[i].size())
            throw std::invalid_argument("Native contraction input rank mismatch.");
        for (size_t j = 0; j < labels[i].size(); ++j) {
            char c = labels[i][j];
            if (dims.count(c) && dims[c] != shapes[i][j])
                throw std::invalid_argument("Native contraction label mismatch.");
            dims[c] = shapes[i][j];
        }
    }
    auto size = [&](const std::string& s) {
        double n = 1.;
        for (char c : s) n *= dims[c];
        return n;
    };
    std::vector<Step> steps;
    while (labels.size() > 1) {
        double best = -std::numeric_limits<double>::infinity(), best_flops = 0.;
        Step step{};
        for (size_t i = 0; i < labels.size(); ++i)
            for (size_t j = i + 1; j < labels.size(); ++j) {
                std::string all = labels[i];
                for (char c : labels[j])
                    if (all.find(c) == std::string::npos) all += c;
                std::string needed = output;
                for (size_t k = 0; k < labels.size(); ++k)
                    if (k != i && k != j) needed += labels[k];
                std::string target;
                for (char c : all)
                    if (needed.find(c) != std::string::npos) target += c;
                if (labels.size() == 2) target = output;
                double removed = size(labels[i]) + size(labels[j]) - size(target), flops = size(all);
                bool shared = false;
                for (char c : labels[i])
                    if (labels[j].find(c) != std::string::npos) shared = true;
                if (!shared && labels.size() > 2) removed = -1.e100 - flops;
                if (removed > best || (removed == best && flops < best_flops)) {
                    best = removed;
                    best_flops = flops;
                    step = {i, j, labels[i], labels[j], target};
                }
            }
        steps.push_back(step);
        labels.erase(labels.begin() + step.j);
        labels.erase(labels.begin() + step.i);
        labels.push_back(step.out);
    }
    return steps;
}
}  // namespace

void parallel_rows(size_t count, const std::function<void(size_t)>& evaluate) {
    if (!count) return;
    [[maybe_unused]] const int threads = std::max(1, std::min(int(count), Process::environment.get_n_threads()));
    std::vector<std::exception_ptr> errors(count);
#pragma omp parallel for schedule(dynamic) num_threads(threads) if (threads > 1)
    for (size_t i = 0; i < count; ++i) {
#ifdef USING_LAPACK_MKL
        const int previous = mkl_set_num_threads_local(1);
#endif
        try {
            evaluate(i);
        } catch (...) {
            errors[i] = std::current_exception();
        }
#ifdef USING_LAPACK_MKL
        mkl_set_num_threads_local(previous);
#endif
    }
    for (auto& error : errors)
        if (error) std::rethrow_exception(error);
}
std::vector<int> range(int n) {
    std::vector<int> out(n);
    std::iota(out.begin(), out.end(), 0);
    return out;
}
Tensor::Tensor() : Tensor(std::vector<int>{}) {}
Tensor::Tensor(std::vector<int> s, double value)
    : shape(std::move(s)), data(std::make_shared<std::vector<double>>(extent(shape), value)) {}
Tensor::Tensor(std::vector<int> s, const double* p) : Tensor(std::move(s)) {
    if (size()) std::copy_n(p, size(), mutable_ptr());
    finite(*this);
}
size_t Tensor::size() const { return extent(shape); }
double Tensor::scalar() const {
    if (size() != 1) throw std::invalid_argument("Expected a scalar response.");
    return ptr()[0];
}
Tensor Tensor::reshape(std::vector<int> s) const {
    if (extent(s) != size()) throw std::invalid_argument("Native tensor reshape size mismatch.");
    Tensor result = *this;
    result.shape = std::move(s);
    return result;
}
Tensor Tensor::permute(const std::vector<int>& axes) const {
    if (axes.size() != shape.size()) throw std::invalid_argument("Native tensor permutation rank mismatch.");
    auto sorted = axes;
    std::sort(sorted.begin(), sorted.end());
    if (sorted != range(shape.size())) throw std::invalid_argument("Invalid tensor permutation.");
    if (axes == sorted) return *this;
    std::vector<int> output;
    for (int i : axes) output.push_back(shape[i]);
    Tensor result(output);
    auto ss = strides(shape), ds = strides(output);
    size_t n = size();
    if (!n) return result;
    // Dense two/three-index tensors dominate local response. Explicit strides
    // avoid integer division per element and copy contiguous inner runs.
    if (output.size() == 2) {
#pragma omp parallel for schedule(static) if (n >= 131072)
        for (int i = 0; i < output[0]; ++i)
            for (int j = 0; j < output[1]; ++j)
                result.mutable_ptr()[size_t(i) * output[1] + j] =
                    ptr()[size_t(i) * ss[axes[0]] + size_t(j) * ss[axes[1]]];
    } else if (output.size() == 3) {
#pragma omp parallel for schedule(static) if (n >= 131072)
        for (int i = 0; i < output[0]; ++i)
            for (int j = 0; j < output[1]; ++j) {
                const auto from = size_t(i) * ss[axes[0]] + size_t(j) * ss[axes[1]];
                auto* target = result.mutable_ptr() + (size_t(i) * output[1] + j) * output[2];
                if (ss[axes[2]] == 1)
                    std::copy_n(ptr() + from, output[2], target);
                else
                    for (int k = 0; k < output[2]; ++k) target[k] = ptr()[from + size_t(k) * ss[axes[2]]];
            }
    } else {
#pragma omp parallel for schedule(static) if (n >= 131072)
        for (size_t p = 0; p < n; ++p) {
            size_t from = 0;
            for (size_t i = 0; i < axes.size(); ++i) from += ((p / ds[i]) % output[i]) * ss[axes[i]];
            result.mutable_ptr()[p] = ptr()[from];
        }
    }
    return result;
}
Tensor Tensor::select(const std::vector<std::vector<int>>& ids, const std::vector<int>& drop) const {
    if (ids.size() != shape.size()) throw std::invalid_argument("Native tensor slice rank mismatch.");
    std::vector<int> output;
    for (size_t i = 0; i < ids.size(); ++i) {
        for (int j : ids[i])
            if (j < 0 || j >= shape[i]) throw std::out_of_range("Native response slice index.");
        output.push_back(ids[i].size());
    }
    auto ss = strides(shape), ds = strides(output);
    bool full = true;
    for (size_t i = 0; i < ids.size(); ++i) full = full && ids[i] == range(shape[i]);
    Tensor result = full ? *this : Tensor(output);
    size_t n = result.size();
    bool contiguous = !ids.empty();
    if (contiguous)
        for (size_t k = 1; k < ids.back().size(); ++k)
            contiguous = contiguous && ids.back()[k] == ids.back()[0] + int(k);
    if (!full && n && output.size() == 2) {
#pragma omp parallel for schedule(static) if (n >= 131072)
        for (int i = 0; i < output[0]; ++i) {
            auto* target = result.mutable_ptr() + size_t(i) * output[1];
            const auto from = size_t(ids[0][i]) * ss[0];
            if (contiguous && output[1])
                std::copy_n(ptr() + from + ids[1][0], output[1], target);
            else
                for (int j = 0; j < output[1]; ++j) target[j] = ptr()[from + ids[1][j]];
        }
    } else if (!full && n && output.size() == 3) {
#pragma omp parallel for schedule(static) if (n >= 131072)
        for (int i = 0; i < output[0]; ++i)
            for (int j = 0; j < output[1]; ++j) {
                auto* target = result.mutable_ptr() + (size_t(i) * output[1] + j) * output[2];
                const auto from = size_t(ids[0][i]) * ss[0] + size_t(ids[1][j]) * ss[1];
                if (contiguous && output[2])
                    std::copy_n(ptr() + from + ids[2][0], output[2], target);
                else
                    for (int k = 0; k < output[2]; ++k) target[k] = ptr()[from + ids[2][k]];
            }
    } else if (!full) {
#pragma omp parallel for schedule(static) if (n >= 131072)
        for (size_t p = 0; p < n; ++p) {
            size_t from = 0;
            for (size_t i = 0; i < ids.size(); ++i) from += ids[i][(p / ds[i]) % output[i]] * ss[i];
            result.mutable_ptr()[p] = ptr()[from];
        }
    }
    auto axes = drop;
    std::sort(axes.begin(), axes.end(), std::greater<int>());
    for (int a : axes) {
        if (a < 0 || a >= int(output.size()) || output[a] != 1)
            throw std::invalid_argument("Cannot drop a nonsingleton native slice axis.");
        output.erase(output.begin() + a);
    }
    return result.reshape(output);
}
Tensor add(const Tensor& a, const Tensor& b, double factor) {
    if (a.shape != b.shape) throw std::invalid_argument("Native response sum shape mismatch.");
    Tensor out(a.shape);
    size_t n = a.size();
#pragma omp parallel for schedule(static) if (n >= 131072)
    for (size_t i = 0; i < n; ++i) out.mutable_ptr()[i] = a.ptr()[i] + factor * b.ptr()[i];
    return out;
}
Tensor scale(const Tensor& a, double factor) {
    Tensor out(a.shape);
    size_t n = a.size();
#pragma omp parallel for schedule(static) if (n >= 131072)
    for (size_t i = 0; i < n; ++i) out.mutable_ptr()[i] = factor * a.ptr()[i];
    return out;
}
Tensor contract(const std::string& equation, const std::vector<Tensor>& input) {
    if (input.empty()) throw std::invalid_argument("Empty native contraction.");
    if (input.size() == 1) {
        auto arrow = equation.find("->");
        return binary(equation.substr(0, arrow), "", equation.substr(arrow + 2), input[0], Tensor({}, 1.));
    }
    std::vector<std::vector<int>> shapes;
    for (const auto& x : input) shapes.push_back(x.shape);
    auto work = input;
    for (const auto& step : plan(equation, shapes)) {
        auto out = binary(step.a, step.b, step.out, work[step.i], work[step.j]);
        work.erase(work.begin() + step.j);
        work.erase(work.begin() + step.i);
        work.push_back(std::move(out));
    }
    return work.at(0);
}
Tensor eye(int n) {
    Tensor x({n, n});
    for (int i = 0; i < n; ++i) x.mutable_ptr()[size_t(i) * n + i] = 1.;
    return x;
}
Tensor solve(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || a.shape[0] != a.shape[1] || b.shape.empty() || b.shape[0] != a.shape[0])
        throw std::invalid_argument("Native solve dimensions disagree.");
    int n = a.shape[0];
    if (!n) return b;
    const int nrhs = b.size() / n;
    auto at = a.permute({1, 0});
    Tensor ac(at.shape, at.ptr());
    auto bt = b.reshape({n, nrhs}).permute({1, 0});
    Tensor bc(bt.shape, bt.ptr());
    std::vector<int> piv(n);
    if (nrhs && C_DGESV(n, nrhs, ac.mutable_ptr(), n, piv.data(), bc.mutable_ptr(), n))
        throw std::runtime_error("Singular native response solve.");
    return bc.permute({1, 0}).reshape(b.shape);
}
std::pair<Tensor, Tensor> eigh(const Tensor& a) {
    if (a.shape.size() != 2 || a.shape[0] != a.shape[1])
        throw std::invalid_argument("Native eigensystem requires a square matrix.");
    int n = a.shape[0];
    Tensor values({n}), vectors(a.shape, a.ptr());
    if (!n) return {values, vectors};
    // Match the divide-and-conquer symmetric eigensolver used by the Python
    // oracle. Keep the same spectral branch on conditioned local metrics.
    double query = 0.;
    int iquery = 0;
    int info = C_DSYEVD('V', 'U', n, vectors.mutable_ptr(), n, values.mutable_ptr(), &query, -1, &iquery, -1);
    if (info || query > std::numeric_limits<int>::max() || iquery < 1)
        throw std::runtime_error("Native eigensystem workspace failure.");
    std::vector<double> work(std::max(1, int(query)));
    std::vector<int> iwork(iquery);
    if (C_DSYEVD('V', 'U', n, vectors.mutable_ptr(), n, values.mutable_ptr(), work.data(), work.size(), iwork.data(),
                 iwork.size()))
        throw std::runtime_error("Native eigensystem did not converge.");
    return {values, vectors.permute({1, 0})};
}
double dot(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) throw std::invalid_argument("Native response dot shape mismatch.");
    double value = 0.;
    for (size_t i = 0; i < a.size(); ++i) value += a.ptr()[i] * b.ptr()[i];
    return value;
}
void close(const Tensor& a, const Tensor& b, double atol, const std::string& label, double rtol) {
    if (a.shape != b.shape) throw std::invalid_argument(label + ": shapes disagree.");
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a.ptr()[i]) || !std::isfinite(b.ptr()[i]) ||
            std::abs(a.ptr()[i] - b.ptr()[i]) > atol + rtol * std::abs(b.ptr()[i]))
            throw std::runtime_error(label + ": response mismatch.");
}
size_t Cache::node(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(key);
    if (it != nodes_.end()) return it->second;
    // Bound graph metadata as well as numerical storage. Monotone identities
    // keep still-live tensors valid after clearing an overgrown graph.
    if (nodes_.size() >= 65536) {
        nodes_.clear();
        input_hashes_.clear();
        values_.clear();
        order_.clear();
        bytes_ = 0;
    }
    return nodes_.emplace(key, ++next_node_).first->second;
}
Tensor Cache::evaluate(size_t node, int coordinate, const std::function<Tensor()>& build) {
    const auto key = std::make_pair(node, coordinate);
    if (coordinate >= 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = values_.find(key);
        if (it != values_.end()) {
            ++hits_;
            order_.splice(order_.end(), order_, it->second.order);
            return it->second.tensor;
        }
    }
    Tensor value = build();
    finite(value);
    std::lock_guard<std::mutex> lock(mutex_);
    ++evaluations_;
    const size_t bytes = value.size() * sizeof(double);
    // Keep intermediates within a quarter of the budget; oversized panels are
    // streamed. LRU limits are shared by every stage and nuclear direction.
    if (coordinate >= 0 && bytes <= limit_ && bytes <= limit_ / 4) {
        auto it = values_.find(key);
        if (it != values_.end()) return it->second.tensor;
        while (!order_.empty() && bytes_ + bytes > limit_) {
            auto old = values_.find(order_.front());
            bytes_ -= old->second.tensor.size() * sizeof(double);
            values_.erase(old);
            order_.pop_front();
        }
        order_.push_back(key);
        values_.emplace(key, Entry{value, std::prev(order_.end())});
        bytes_ += bytes;
    }
    return value;
}
void Cache::check_input(size_t node, int coordinate, const Tensor& value) {
    finite(value);
    if (coordinate < 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = values_.find({node, coordinate});
    if (it != values_.end()) close(value, it->second.tensor, 2.e-9, "Native cached input");
    // Input blocks may be larger than the retained-panel cap or may have been
    // evicted. Never reuse descendants after an unverified input change. Exact
    // fingerprints invalidate the numerical cache; tolerances remain solely
    // acceptance checks, not permission to substitute a different input.
    uint64_t hash = 1469598103934665603ULL;
    for (int n : value.shape) hash = (hash ^ uint64_t(n)) * 1099511628211ULL;
    for (double x : *value.data) {
        uint64_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        hash = (hash ^ bits) * 1099511628211ULL;
    }
    auto key = std::make_pair(node, coordinate);
    auto old = input_hashes_.find(key);
    if (old != input_hashes_.end() && old->second != hash) {
        values_.clear();
        order_.clear();
        bytes_ = 0;
    }
    input_hashes_[key] = hash;
}
std::array<size_t, 4> Cache::statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {bytes_, limit_, hits_, evaluations_};
}
Mixed Context::evaluate(const std::string& operation, const std::vector<size_t>& inputs,
                        const std::function<Tensor(int)>& build) const {
    std::string key = scope + "/" + operation;
    for (size_t id : inputs) key += "/" + std::to_string(id);
    Mixed out;
    out.context = this;
    out.id = cache->node(key);
    for (int k = 0; k < 4; ++k)
        out.x[k] = cache->evaluate(out.id,
                                   k == 0   ? 0
                                   : k == 1 ? u + 1
                                   : k == 2 ? v + 1
                                            : -1,
                                   [&]() { return build(k); });
    return out;
}
Mixed Context::input(const std::string& name, const std::array<Tensor, 4>& values) const {
    const auto id = cache->node(scope + "/input:" + name);
    for (int k = 0; k < 4; ++k) {
        if (values[k].shape != values[0].shape) throw std::invalid_argument("Mixed input block dimensions disagree.");
        cache->check_input(id, k == 0 ? 0 : k == 1 ? u + 1 : k == 2 ? v + 1 : -1, values[k]);
    }
    Mixed out;
    out.id = id;
    out.context = this;
    for (int k = 0; k < 4; ++k)
        out.x[k] = cache->evaluate(id, k == 0 ? 0 : k == 1 ? u + 1 : k == 2 ? v + 1 : -1, [&]() { return values[k]; });
    return out;
}
Mixed Context::constant(const std::string& name, const Tensor& value) const {
    // Include the immutable constant contents: different local parents must
    // never alias merely because their dimensions and descriptive names match.
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < value.size(); ++i) {
        uint64_t bits;
        std::memcpy(&bits, value.ptr() + i, sizeof(bits));
        hash = (hash ^ bits) * 1099511628211ULL;
    }
    auto zero = Tensor(value.shape);
    return input("constant:" + name + ":" + numbers(value.shape) + ":" + std::to_string(hash),
                 {value, zero, zero, zero});
}
Mixed Mixed::permute(const std::vector<int>& axes) const {
    return context->evaluate("permute:" + numbers(axes), {id}, [&](int k) { return x[k].permute(axes); });
}
Mixed Mixed::select(const std::vector<std::vector<int>>& indices, const std::vector<int>& drop) const {
    std::string key = "select:" + numbers(drop);
    for (const auto& row : indices) key += ";" + numbers(row);
    return context->evaluate(key, {id}, [&](int k) { return x[k].select(indices, drop); });
}
Mixed sum(const std::vector<Mixed>& x) {
    if (x.empty()) throw std::invalid_argument("Empty mixed sum.");
    std::vector<size_t> ids;
    for (const auto& a : x) ids.push_back(a.id);
    return x[0].context->evaluate("sum", ids, [&](int k) {
        auto out = x[0][k];
        for (size_t i = 1; i < x.size(); ++i) out = add(out, x[i][k]);
        return out;
    });
}
Mixed scaled(const Mixed& a, double f) {
    std::ostringstream key;
    key.precision(17);
    key << "scale:" << f;
    return a.context->evaluate(key.str(), {a.id}, [&](int k) { return scale(a[k], f); });
}
Mixed product(const std::string& equation, const std::vector<Mixed>& blocks) {
    if (blocks.empty()) throw std::invalid_argument("Empty mixed product.");
    if (blocks.size() == 1)
        return blocks[0].context->evaluate("contract:" + equation, {blocks[0].id},
                                           [&](int k) { return contract(equation, {blocks[0][k]}); });
    std::vector<std::vector<int>> shapes;
    for (const auto& x : blocks) shapes.push_back(x[0].shape);
    auto work = blocks;
    for (const auto& step : plan(equation, shapes)) {
        const auto a = work[step.i], b = work[step.j];
        auto out =
            a.context->evaluate("contract:" + step.a + "," + step.b + "->" + step.out, {a.id, b.id}, [&](int mask) {
                Tensor value;
                bool first = true;
                for (int i = 0; i < 4; ++i)
                    if ((i & mask) == i) {
                        auto term = binary(step.a, step.b, step.out, a[i], b[mask ^ i]);
                        value = first ? term : add(value, term);
                        first = false;
                    }
                return value;
            });
        work.erase(work.begin() + step.j);
        work.erase(work.begin() + step.i);
        work.push_back(std::move(out));
    }
    return work[0];
}
Mixed Mixed::reshape(std::vector<int> dimensions) const {
    return context->evaluate("reshape:" + numbers(dimensions), {id}, [&](int k) { return x[k].reshape(dimensions); });
}
Mixed stack(const std::vector<Mixed>& rows) {
    if (rows.empty()) throw std::invalid_argument("Empty native response stack.");
    std::vector<size_t> ids;
    for (auto& r : rows) {
        if (r[0].shape != rows[0][0].shape) throw std::invalid_argument("Native stack shape mismatch.");
        ids.push_back(r.id);
    }
    auto shape = rows[0][0].shape;
    shape.insert(shape.begin(), rows.size());
    return rows[0].context->evaluate("stack", ids, [&](int k) {
        Tensor out(shape);
        size_t offset = 0;
        for (auto& r : rows) {
            std::copy_n(r[k].ptr(), r[k].size(), out.mutable_ptr() + offset);
            offset += r[k].size();
        }
        return out;
    });
}
Mixed metric_solve(const Mixed& s, const Mixed& b) {
    if (s[0].shape.size() != 2 || s[0].shape[0] != s[0].shape[1] || b[0].shape.empty() ||
        b[0].shape[0] != s[0].shape[0])
        throw std::invalid_argument("Native mixed solve dimensions disagree.");
    const int n = s[0].shape[0];
    if (!n) return b;
    const auto id = s.context->cache->node(s.context->scope + "/LU:" + std::to_string(s.id));
    // Factor the fixed local metric once. Pivots occupy the final row of this
    // private cache block; DG(E)TRS reads, but never changes, the LU factors.
    auto factors = s.context->cache->evaluate(id, 0, [&]() {
        Tensor result({n + 1, n});
        auto a = s[0].permute({1, 0});
        std::copy_n(a.ptr(), a.size(), result.mutable_ptr());
        std::vector<int> pivots(n);
        if (C_DGETRF(n, n, result.mutable_ptr(), n, pivots.data()))
            throw std::runtime_error("Singular native response metric.");
        for (int i = 0; i < n; ++i) result.mutable_ptr()[size_t(n) * n + i] = pivots[i];
        return result;
    });
    std::vector<int> pivots(n);
    for (int i = 0; i < n; ++i) pivots[i] = int(factors.ptr()[size_t(n) * n + i]);
    auto inverse = [&](const Tensor& rhs) {
        int nrhs = rhs.size() / n;
        auto bt = rhs.reshape({n, nrhs}).permute({1, 0});
        Tensor x(bt.shape, bt.ptr());
        if (nrhs && C_DGETRS('N', n, nrhs, const_cast<double*>(factors.ptr()), n, pivots.data(), x.mutable_ptr(), n))
            throw std::runtime_error("Native response metric solve failed.");
        return x.permute({1, 0}).reshape(rhs.shape);
    };
    std::array<Tensor, 4> y;
    std::array<bool, 4> ready{};
    std::function<Tensor(int)> component = [&](int k) -> Tensor {
        if (ready[k]) return y[k];
        Tensor rhs = b[k];
        for (int i = 1; i < 4; ++i)
            if ((i & k) == i)
                rhs = rhs - contract("ij,jk->ik", {s[i], component(k ^ i).reshape({n, int(b[0].size() / n)})})
                                .reshape(b[0].shape);
        ready[k] = true;
        return y[k] = inverse(rhs);
    };
    return s.context->evaluate("solve", {s.id, b.id}, component);
}

}  // namespace response
}  // namespace dlpno
}  // namespace psi
