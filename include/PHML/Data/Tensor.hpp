#pragma once

#include "Core.hpp"
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace PHML::Data {

    template <typename T>
    class Tensor : public Core<Tensor<T>> {
        using Base = Core<Tensor<T>>;

    public:

        // Constructors
        Tensor() = default;

        /// Allocate with the given shape, zero-initialised
        explicit Tensor(std::vector<std::size_t> shape,
                        Device dev = Device::cpu())
            : Base(dtype_of<T>(), dev)
            , shape_(std::move(shape))
            , strides_(default_strides(shape_))
            , numel_(Core<Tensor<T>>::compute_numel(shape_))
        {
            Base::init_storage(numel_ * sizeof(T));
            std::fill(begin(), end(), T{0});
        }

        /// Allocate with the given shape, filled with default_val
        Tensor(std::vector<std::size_t> shape, T default_val,
               Device dev = Device::cpu())
            : Base(dtype_of<T>(), dev)
            , shape_(std::move(shape))
            , strides_(default_strides(shape_))
            , numel_(Core<Tensor<T>>::compute_numel(shape_))
        {
            Base::init_storage(numel_ * sizeof(T));
            std::fill(begin(), end(), default_val);
        }

        /// Construct from flat initialiser list (row-major)
        Tensor(std::vector<std::size_t> shape,
               std::initializer_list<T> values,
               Device dev = Device::cpu())
            : Base(dtype_of<T>(), dev)
            , shape_(std::move(shape))
            , strides_(default_strides(shape_))
            , numel_(Core<Tensor<T>>::compute_numel(shape_))
        {
            if (values.size() != numel_)
                throw std::invalid_argument(
                    "Tensor initializer_list size mismatch: expected " +
                    std::to_string(numel_) + ", got " +
                    std::to_string(values.size()));
            Base::init_storage(numel_ * sizeof(T));
            std::copy(values.begin(), values.end(), begin());
        }

        // Shape / layout queries
        std::size_t                      ndim()   const { return shape_.size(); }
        const std::vector<std::size_t>&  shape()  const { return shape_; }
        std::size_t                      shape(std::size_t i) const {
            if (i >= shape_.size())
                throw std::out_of_range("shape(" + std::to_string(i) +
                                        "): axis out of range [ndim=" +
                                        std::to_string(shape_.size()) + "]");
            return shape_[i];
        }
        const std::vector<std::size_t>&  strides() const { return strides_; }
        std::size_t                      numel()   const { return numel_; }
        bool                             is_contiguous() const {
            return strides_ == default_strides(shape_);
        }

        // Variadic element access — rank is checked at runtime
        template <typename... Idxs>
        T& operator()(Idxs... idxs) {
            static_assert(sizeof...(Idxs) > 0,
                          "Tensor::operator(): at least one index required");
            static_assert((std::is_integral_v<Idxs> && ...),
                          "Tensor::operator(): indices must be integral");
            const std::size_t arr[] = { static_cast<std::size_t>(idxs)... };
            return Base::template data_ptr<T>()[offset_of(arr, sizeof...(Idxs))];
        }

        template <typename... Idxs>
        const T& operator()(Idxs... idxs) const {
            static_assert(sizeof...(Idxs) > 0,
                          "Tensor::operator(): at least one index required");
            static_assert((std::is_integral_v<Idxs> && ...),
                          "Tensor::operator(): indices must be integral");
            const std::size_t arr[] = { static_cast<std::size_t>(idxs)... };
            return Base::template data_ptr<T>()[offset_of(arr, sizeof...(Idxs))];
        }

        /// Runtime-rank element access
        T& at(std::initializer_list<std::size_t> idx) {
            return Base::template data_ptr<T>()[offset_of(idx.begin(), idx.size())];
        }
        const T& at(std::initializer_list<std::size_t> idx) const {
            return Base::template data_ptr<T>()[offset_of(idx.begin(), idx.size())];
        }

        // Element-wise arithmetic (shape-equal only)

        Tensor<T> operator+(const Tensor<T>& other) const {
            shape_check(other, "operator+");
            Tensor<T> result(shape_, Base::device_);
            binary_op(other, result, [](T a, T b){ return a + b; });
            return result;
        }

        Tensor<T> operator-(const Tensor<T>& other) const {
            shape_check(other, "operator-");
            Tensor<T> result(shape_, Base::device_);
            binary_op(other, result, [](T a, T b){ return a - b; });
            return result;
        }

        /// Hadamard (element-wise) product — not matmul.
        Tensor<T> operator*(const Tensor<T>& other) const {
            shape_check(other, "operator*");
            Tensor<T> result(shape_, Base::device_);
            binary_op(other, result, [](T a, T b){ return a * b; });
            return result;
        }

        Tensor<T> operator*(T scalar) const {
            Tensor<T> result(shape_, Base::device_);
            T* dst = result.template data_ptr<T>();
            if (is_contiguous()) {
                const T* src = this->template data_ptr<T>();
                for (std::size_t i = 0; i < numel_; ++i)
                    dst[i] = src[i] * scalar;
            } else {
                walk_indices([&](const std::vector<std::size_t>& idx,
                                 std::size_t flat) {
                    dst[flat] = at_offset(idx) * scalar;
                });
            }
            return result;
        }

        friend Tensor<T> operator*(T scalar, const Tensor<T>& t) {
            return t * scalar;
        }

        // Views (zero-copy, shared storage)

        /// Reshape — requires a contiguous tensor with matching numel.
        Tensor<T> reshape(std::vector<std::size_t> new_shape) const {
            if (!is_contiguous())
                throw std::runtime_error(
                    "reshape: tensor is not contiguous; call .contiguous() first");
            std::size_t new_numel = Core<Tensor<T>>::compute_numel(new_shape);
            if (new_numel != numel_)
                throw std::invalid_argument(
                    "reshape: numel mismatch (have " + std::to_string(numel_) +
                    ", requested " + std::to_string(new_numel) + ")");
            Tensor<T> view;
            view.dtype_   = Base::dtype_;
            view.device_  = Base::device_;
            view.storage_ = Base::storage_;
            view.shape_   = std::move(new_shape);
            view.strides_ = default_strides(view.shape_);
            view.numel_   = new_numel;
            return view;
        }

        /// Permute dims — `axes` must be a permutation of 0..ndim()-1.
        Tensor<T> permute(const std::vector<std::size_t>& axes) const {
            if (axes.size() != ndim())
                throw std::invalid_argument(
                    "permute: axes.size() (" + std::to_string(axes.size()) +
                    ") != ndim (" + std::to_string(ndim()) + ")");
            std::vector<bool> seen(ndim(), false);
            for (auto a : axes) {
                if (a >= ndim())
                    throw std::out_of_range(
                        "permute: axis " + std::to_string(a) + " out of range");
                if (seen[a])
                    throw std::invalid_argument(
                        "permute: duplicate axis " + std::to_string(a));
                seen[a] = true;
            }
            Tensor<T> view;
            view.dtype_   = Base::dtype_;
            view.device_  = Base::device_;
            view.storage_ = Base::storage_;
            view.shape_.resize(ndim());
            view.strides_.resize(ndim());
            for (std::size_t i = 0; i < ndim(); ++i) {
                view.shape_[i]   = shape_[axes[i]];
                view.strides_[i] = strides_[axes[i]];
            }
            view.numel_ = numel_;
            return view;
        }

        /// Swap two axes — convenience wrapper around permute.
        Tensor<T> transpose(std::size_t a, std::size_t b) const {
            if (a >= ndim() || b >= ndim())
                throw std::out_of_range(
                    "transpose: axis out of range (ndim=" +
                    std::to_string(ndim()) + ")");
            std::vector<std::size_t> axes(ndim());
            std::iota(axes.begin(), axes.end(), std::size_t{0});
            std::swap(axes[a], axes[b]);
            return permute(axes);
        }

        /// Materialise into a fresh contiguous tensor (no-op if already contiguous).
        Tensor<T> contiguous() const {
            if (is_contiguous()) return *this;
            return deep_copy(Base::device_);
        }

        // Device transfer
        Tensor<T> to(Device target) const {
            if (Base::device_ == target) return *this;
            if (!target.is_cpu())
                throw std::runtime_error("CUDA transfer not yet implemented");
            return deep_copy(target);
        }

        // Iterators — valid element order only when is_contiguous().
        T*       begin()       { return Base::template data_ptr<T>(); }
        T*       end()         { return begin() + numel_; }
        const T* begin() const { return Base::template data_ptr<T>(); }
        const T* end()   const { return begin() + numel_; }

        // Static factories
        static Tensor<T> zeros(std::vector<std::size_t> shape,
                               Device dev = Device::cpu())
        {
            return Tensor<T>(std::move(shape), T{0}, dev);
        }

        static Tensor<T> ones(std::vector<std::size_t> shape,
                              Device dev = Device::cpu())
        {
            return Tensor<T>(std::move(shape), T{1}, dev);
        }

        static Tensor<T> full(std::vector<std::size_t> shape, T value,
                              Device dev = Device::cpu())
        {
            return Tensor<T>(std::move(shape), value, dev);
        }

        static Tensor<T> random(std::vector<std::size_t> shape,
                                T low = T{0}, T high = T{1},
                                Device dev = Device::cpu())
        {
            Tensor<T> t(std::move(shape), dev);
            std::mt19937_64 rng{std::random_device{}()};
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> dist(low, high);
                for (auto& v : t) v = dist(rng);
            } else {
                std::uniform_real_distribution<T> dist(low, high);
                for (auto& v : t) v = dist(rng);
            }
            return t;
        }

        // Printing — recursive nested-bracket layout
        void print(std::ostream& os = std::cout) const {
            os << "Tensor<" << dtype_str(Base::dtype()) << "> [";
            for (std::size_t i = 0; i < ndim(); ++i) {
                os << shape_[i];
                if (i + 1 < ndim()) os << "x";
            }
            os << "] " << Base::device().str() << "\n";
            if (ndim() == 0 || numel_ == 0) { os << "  []\n"; return; }
            std::vector<std::size_t> idx(ndim(), 0);
            os << "  ";
            print_dim(os, idx, 0);
            os << "\n";
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
            t.print(os); return os;
        }

    private:
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;
        std::size_t              numel_ = 0;

        template <typename U>
        static DType dtype_of() {
            if constexpr (std::is_same_v<U, float>)    return DType::Float32;
            if constexpr (std::is_same_v<U, double>)   return DType::Float64;
            if constexpr (std::is_same_v<U, int32_t>)  return DType::Int32;
            if constexpr (std::is_same_v<U, int64_t>)  return DType::Int64;
            if constexpr (std::is_same_v<U, bool>)     return DType::Bool;
            throw std::runtime_error("Unsupported scalar type for Tensor");
        }

        static std::vector<std::size_t>
        default_strides(const std::vector<std::size_t>& shape) {
            std::vector<std::size_t> s(shape.size(), 1);
            for (std::size_t i = shape.size(); i-- > 1; )
                s[i - 1] = s[i] * shape[i];
            return s;
        }

        std::size_t offset_of(const std::size_t* idx, std::size_t n) const {
            if (n != shape_.size())
                throw std::invalid_argument(
                    "Tensor index rank mismatch: got " + std::to_string(n) +
                    " indices for " + std::to_string(shape_.size()) + "-D tensor");
            std::size_t off = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (idx[i] >= shape_[i])
                    throw std::out_of_range(
                        "Tensor index " + std::to_string(idx[i]) +
                        " out of range on axis " + std::to_string(i) +
                        " (size " + std::to_string(shape_[i]) + ")");
                off += idx[i] * strides_[i];
            }
            return off;
        }

        const T& at_offset(const std::vector<std::size_t>& idx) const {
            std::size_t off = 0;
            for (std::size_t i = 0; i < idx.size(); ++i)
                off += idx[i] * strides_[i];
            return Base::template data_ptr<T>()[off];
        }

        void shape_check(const Tensor<T>& other, const char* op) const {
            if (shape_ != other.shape_) {
                std::string msg = std::string(op) + ": shape mismatch [";
                for (std::size_t i = 0; i < ndim(); ++i) {
                    msg += std::to_string(shape_[i]);
                    if (i + 1 < ndim()) msg += "x";
                }
                msg += "] vs [";
                for (std::size_t i = 0; i < other.ndim(); ++i) {
                    msg += std::to_string(other.shape_[i]);
                    if (i + 1 < other.ndim()) msg += "x";
                }
                msg += "]";
                throw std::invalid_argument(msg);
            }
        }

        /// Walk the logical index space of shape_ in row-major order,
        /// invoking fn(idx, flat) for each element.
        template <typename Fn>
        void walk_indices(Fn&& fn) const {
            if (numel_ == 0 || ndim() == 0) return;
            std::vector<std::size_t> idx(ndim(), 0);
            std::size_t flat = 0;
            while (true) {
                fn(idx, flat);
                ++flat;
                std::size_t i = ndim();
                while (i-- > 0) {
                    if (++idx[i] < shape_[i]) break;
                    idx[i] = 0;
                    if (i == 0) return;
                }
            }
        }

        /// Apply a binary op between *this and other, storing into dst.
        /// dst must have the same shape as *this and must be contiguous.
        template <typename Op>
        void binary_op(const Tensor<T>& other, Tensor<T>& dst, Op op) const {
            T* d = dst.template data_ptr<T>();
            if (is_contiguous() && other.is_contiguous()) {
                const T* a = this->template data_ptr<T>();
                const T* b = other.template data_ptr<T>();
                for (std::size_t i = 0; i < numel_; ++i)
                    d[i] = op(a[i], b[i]);
            } else {
                walk_indices([&](const std::vector<std::size_t>& idx,
                                 std::size_t flat) {
                    d[flat] = op(at_offset(idx), other.at_offset(idx));
                });
            }
        }

        Tensor<T> deep_copy(Device dev) const {
            Tensor<T> dst(shape_, dev);
            if (is_contiguous()) {
                std::memcpy(dst.template data_ptr<T>(),
                            Base::storage_->data(),
                            numel_ * sizeof(T));
            } else {
                T* d = dst.template data_ptr<T>();
                walk_indices([&](const std::vector<std::size_t>& idx,
                                 std::size_t flat) {
                    d[flat] = at_offset(idx);
                });
            }
            return dst;
        }

        void print_dim(std::ostream& os,
                       std::vector<std::size_t>& idx,
                       std::size_t dim) const {
            os << "[";
            if (dim + 1 == ndim()) {
                for (std::size_t i = 0; i < shape_[dim]; ++i) {
                    idx[dim] = i;
                    os << at_offset(idx);
                    if (i + 1 < shape_[dim]) os << ", ";
                }
            } else {
                for (std::size_t i = 0; i < shape_[dim]; ++i) {
                    idx[dim] = i;
                    print_dim(os, idx, dim + 1);
                    if (i + 1 < shape_[dim]) {
                        os << ",\n  ";
                        for (std::size_t s = 0; s <= dim; ++s) os << " ";
                    }
                }
            }
            os << "]";
        }
    };

    using AnyTensor = std::variant<Tensor<float>, Tensor<double>,
                                   Tensor<int32_t>, Tensor<int64_t>>;

    class TensorFactory {
    public:
        TensorFactory() = delete;
        TensorFactory(const TensorFactory&) = delete;
        TensorFactory& operator=(const TensorFactory&) = delete;

        static AnyTensor random(std::vector<std::size_t> shape, DType dt,
                                double low = 0.0, double high = 1.0,
                                Device dev = Device::cpu())
        {
            switch (dt) {
                case DType::Float32:
                    return Tensor<float>::random(std::move(shape),
                                                 static_cast<float>(low),
                                                 static_cast<float>(high), dev);
                case DType::Float64:
                    return Tensor<double>::random(std::move(shape), low, high, dev);
                case DType::Int32:
                    return Tensor<int32_t>::random(std::move(shape),
                                                   static_cast<int32_t>(low),
                                                   static_cast<int32_t>(high), dev);
                case DType::Int64:
                    return Tensor<int64_t>::random(std::move(shape),
                                                   static_cast<int64_t>(low),
                                                   static_cast<int64_t>(high), dev);
                default:
                    throw std::runtime_error("Unsupported DType for random tensor");
            }
        }

        static AnyTensor zeros(std::vector<std::size_t> shape, DType dt,
                               Device dev = Device::cpu())
        {
            switch (dt) {
                case DType::Float32: return Tensor<float>::zeros(std::move(shape), dev);
                case DType::Float64: return Tensor<double>::zeros(std::move(shape), dev);
                case DType::Int32:   return Tensor<int32_t>::zeros(std::move(shape), dev);
                case DType::Int64:   return Tensor<int64_t>::zeros(std::move(shape), dev);
                default:
                    throw std::runtime_error("Unsupported DType for zeros tensor");
            }
        }
    };

} // namespace PHML::Data
