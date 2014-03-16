#ifndef PYTHONIC_TYPES_NDARRAY_HPP
#define PYTHONIC_TYPES_NDARRAY_HPP

#include "pythonic/utils/proxy.hpp"
#include "pythonic/types/assignable.hpp"
#include "pythonic/types/empty_iterator.hpp"
#include "pythonic/types/attr.hpp"
#include "pythonic/utils/nested_container.hpp"
#include "pythonic/utils/shared_ref.hpp"
#include "pythonic/utils/reserve.hpp"
#include "pythonic/utils/int_.hpp"
#include "pythonic/types/slice.hpp"
#include "pythonic/types/tuple.hpp"
#include "pythonic/types/list.hpp"
#include "pythonic/__builtin__/len.hpp"

#include <cassert>
#include <iostream>
#include <iterator>
#include <array>
#include <initializer_list>
#include <numeric>

#include <boost/simd/sdk/simd/logical.hpp>

#ifdef USE_BOOST_SIMD
#include <boost/simd/sdk/simd/native.hpp>
#include <boost/simd/include/functions/unaligned_load.hpp>
#include <boost/simd/include/functions/unaligned_store.hpp>
#include <boost/simd/include/functions/load.hpp>
#include <boost/simd/include/functions/store.hpp>
#endif

namespace pythonic {

    namespace types {

        struct foreign {}; // used to mark memory as foreigned memory

        template<class T, size_t N>
            struct ndarray;
        template<class Arg, class... S>
            struct numpy_gexpr;
        template<class Expr>
            struct is_array;
        template<class Expr>
            struct is_numexpr_arg;
        template<class T>
            struct type_helper;
        template<class T, class _ = typename std::enable_if<is_numexpr_arg<T>::value, void>::type /* just to filter out scalar types */>
            struct numpy_expr_to_ndarray;

        /* generic function to copy an array to another
         * implements array broadcasting in addtion to regular copy
         */
        template<class E, class F>
            E& broadcast_copy(E& self, F const& other, utils::int_<0>) {
                std::copy(other.begin(), other.end(), self.begin());
                return self;
            }
        template<class E, class F, size_t N>
            E& broadcast_copy(E& self, F const& other, utils::int_<N>) {
                std::fill(self.begin(), self.end(), other);
                return self;
            }

        /* helper function to get the dimension of an array
         * yields 0 for scalar types
         */
        template <class T>
        struct dim_of {
            static const size_t value = T::value;
        };

#define SPECIALIZE_DIM_OF(TYPE) template<> struct dim_of<TYPE> { static const size_t value = 0; }
        SPECIALIZE_DIM_OF(bool);
        SPECIALIZE_DIM_OF(int8_t);
        SPECIALIZE_DIM_OF(int16_t);
        SPECIALIZE_DIM_OF(int32_t);
        SPECIALIZE_DIM_OF(int64_t);
        SPECIALIZE_DIM_OF(uint8_t);
        SPECIALIZE_DIM_OF(uint16_t);
        SPECIALIZE_DIM_OF(uint32_t);
        SPECIALIZE_DIM_OF(uint64_t);
        SPECIALIZE_DIM_OF(float);
        SPECIALIZE_DIM_OF(double);
        SPECIALIZE_DIM_OF(std::complex<float>);
        SPECIALIZE_DIM_OF(std::complex<double>);
#undef SPECIALIZE_DIM_OF

        template<class T, size_t N>
            struct dim_of<array<T,N>> {
                static const size_t value = 1 + dim_of<T>::value;
            };

        /* Wrapper class to store an array pointer
         *
         * for internal use only, meant to be stored in a shared_ptr
         */
        template<class T>
            class raw_array {
                raw_array(raw_array<T> const& );

                public:
                typedef T* pointer_type;

                T* data;
                raw_array() : data(nullptr) {}
                raw_array(size_t n) : data(new T[n]) {}
                raw_array(T* d) : data(d) {}
                raw_array(raw_array<T>&& d) : data(d.data) { d.data = nullptr; }

                ~raw_array() {
                    if(data)
                        delete [] data;
                }
            };

        /* Iterator over whatever provides a fast(long) method to access its element
         */
        template<class E>
            struct nditerator : std::iterator<std::random_access_iterator_tag, typename E::value_type, ptrdiff_t, typename E::value_type *, typename E::value_type /* no ref here */> {
                E & data;
                long index;
                nditerator(E & data, long index) : data(data), index(index) {}

                auto operator*() -> decltype(data.fast(index)) { return data.fast(index); }
                auto operator*() const -> decltype(data.fast(index)) { return data.fast(index); }
                nditerator<E>& operator++() { index ++; return *this;}
                nditerator<E>& operator--() { index --; return *this;}
                nditerator<E> operator+(long i) const { nditerator<E> other(*this); other.index+=  i; return other; }
                nditerator<E> operator-(long i) const { nditerator<E> other(*this); other.index-=  i; return other; }
                nditerator<E>& operator+=(long i) { index +=  i; return *this;}
                nditerator<E>& operator-=(long i) { index -=  i; return *this;}
                long operator-(nditerator<E> const& other) const { return index - other.index; }
                bool operator!=(nditerator<E> const& other) const {
                    return index != other.index;
                }
                bool operator==(nditerator<E> const& other) const {
                    return index == other.index;
                }
                bool operator<(nditerator<E> const& other) const {
                    return index < other.index;
                }
            };

        /* Const iterator over whatever provides a fast(long) method to access its element
         */
        template<class E>
            struct const_nditerator : std::iterator<std::random_access_iterator_tag, typename E::value_type> {
                E const &data;
                long index;
                const_nditerator(E const& data, long index) : data(data), index(index) {
                }

                auto operator*() const -> decltype(data.fast(index)) { return data.fast(index); }
                const_nditerator<E>& operator++() { index ++ ; return *this;}
                const_nditerator<E>& operator--() { index -- ; return *this;}
                const_nditerator<E> operator+(long i) const { const_nditerator<E> other(*this); other.index +=  i; return other; }
                const_nditerator<E> operator-(long i) const { const_nditerator<E> other(*this); other.index -=  i; return other; }
                const_nditerator<E>& operator+=(long i) { index +=  i; return *this;}
                const_nditerator<E>& operator-=(long i) { index -=  i; return *this;}
                long operator-(const_nditerator<E> const& other) const { 
                    return index - other.index;
                }
                bool operator!=(const_nditerator<E> const& other) const {
                    return index != other.index;
                }
                bool operator==(const_nditerator<E> const& other) const {
                    return index == other.index;
                }
                bool operator<(const_nditerator<E> const& other) const {
                    return  index <  other.index;
                }
            };

        /* Type adaptor for scalar values
         *
         * Have them behave like infinite arrays of that value
         */
        template<class T>
            struct broadcast {
                typedef T dtype;
                typedef T value_type;
                static constexpr size_t value = 0;
                T __value;

                broadcast() {}
                broadcast(T v) : __value(v) {}

                T operator[](long ) const {
                    return __value;
                }
                T fast(long ) const {
                    return __value;
                }
                long size() const { return 0; }
            };


        /* helper function to recursively retreive a scalar at given index tuple in a muliple dimension array
         */
        template<class T, class O>
            typename T::dtype getrec(T const & what, O indices, utils::int_<0>)
            {
                return what.fast(*indices);
            }

        template<class T, class O, size_t N>
            typename T::dtype getrec(T const & what, O indices, utils::int_<N>)
            {
                return getrec(what.fast(*indices), indices + 1, utils::int_<N-1>());
            }
        template<class T, class O>
            typename T::dtype & getrec(T & what, O indices, utils::int_<0>)
            {
                return what.fast(*indices);
            }

        template<class T, class O, size_t N>
            typename T::dtype & getrec(T & what, O indices, utils::int_<N>)
            {
                auto next = what.fast(*indices);
                return getrec(next, indices + 1, utils::int_<N-1>());
            }

        /* Expression template for numpy expressions - filter
         */
        template<class Arg, class F>
            struct numpy_fexpr {
                static constexpr size_t value = 1;
                typedef typename std::remove_reference<Arg>::type::dtype value_type;
                typedef typename std::remove_reference<Arg>::type::dtype dtype;

                typedef nditerator<numpy_fexpr> iterator;
                typedef const_nditerator<numpy_fexpr> const_iterator;

                typename assignable<typename std::remove_reference<Arg>::type>::type arg;
                array<long, value> shape;
                utils::shared_ref<raw_array<long>> indices;
                long *buffer;

                numpy_fexpr() {}
                numpy_fexpr(numpy_fexpr const&) = default;
                numpy_fexpr(numpy_fexpr&&) = default;

                template<class FIter, class O>
                void _copy_mask(FIter fiter, FIter fend, O& out, long &index, utils::int_<1>) {
                    for(; fiter!=fend; ++fiter, ++index)
                        if(*fiter) {
                            *out++ = index;
                        }
                }
                template<class FIter, class O, size_t N>
                void _copy_mask(FIter fiter, FIter fend, O& out, long & index, utils::int_<N>) {
                    for(; fiter != fend; ++fiter) {
                        _copy_mask((*fiter).begin(), (*fiter).end(), out, index, utils::int_<N-1>());
                    }
                }

                numpy_fexpr(Arg const &arg, F const& filter) : arg(arg), indices(arg.size()), buffer(indices->data)
                {
                    auto iter = buffer;
                    long index = 0;
                    _copy_mask(filter.begin(), filter.end(), iter, index, utils::int_<Arg::value>());
                    shape[0] = { iter - buffer };
                }

                template<class E>
                typename std::enable_if<is_iterable<E>::value, numpy_fexpr&>::type
                operator=(E const& expr) {
                    std::copy(expr.begin(), expr.end(), begin());
                    return *this;
                }
                template<class E>
                typename std::enable_if<not is_iterable<E>::value, numpy_fexpr&>::type
                operator=(E const& expr) {
                    std::fill(begin(), end(), expr);
                    return *this;
                }
                numpy_fexpr& operator=(numpy_fexpr const& expr) {
                    std::copy(expr.begin(), expr.end(), begin());
                    return *this;
                }

                const_iterator begin() const { return const_iterator(*this, 0); }
                const_iterator end() const { return const_iterator(*this, shape[0]); }

                iterator begin() { return iterator(*this, 0); }
                iterator end() { return iterator(*this, shape[0]); }


                dtype fast(long i) const
                {
                    return *(arg.fbegin() + buffer[i]);
                }
                dtype& fast(long i)
                {
                    return *(arg.fbegin() + buffer[i]);
                }
                auto operator[](long i) const -> decltype(this->fast(i))
                {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator[](long i) -> decltype(this->fast(i))
                {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                long size() const { return shape[0]; }
            };

        /* expression template for Transposed matrix */
        template<class Arg>
            struct numpy_texpr;
       
        // only implemented for N = 2
        template<class T>
            struct numpy_texpr<ndarray<T, 2>> {
                typedef ndarray<T, 2> Arg;

                typedef nditerator<numpy_texpr<Arg>> iterator;
                typedef const_nditerator<numpy_texpr<Arg>> const_iterator;

                static constexpr size_t value = Arg::value;
                typedef numpy_gexpr<Arg, contiguous_slice, long> value_type;
                typedef T dtype;

                Arg arg;
                array<long, 2> shape;

                numpy_texpr() {}
                numpy_texpr(numpy_texpr const&) = default;
                numpy_texpr(numpy_texpr &&) = default;

                numpy_texpr(Arg const& arg) : arg(arg), shape{{arg.shape[1], arg.shape[0]}}
                {
                }
                const_iterator begin() const { return const_iterator(*this, 0); }
                const_iterator end() const { return const_iterator(*this, shape[0]); }

                iterator begin() { return iterator(*this, 0); }
                iterator end() { return iterator(*this, shape[0]); }

                auto fast(long i) const
                    -> decltype(this->arg(contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None), i))
                {
                    return arg(contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None), i);
                }

                auto fast(long i) 
                    -> decltype(this->arg(contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None), i))
                {
                    return arg(contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None), i);
                }

                auto operator[](long i) const -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator[](long i) -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator()(long i) const -> decltype((*this)[i]) {
                    return (*this)[i];
                }
                auto operator()(long i) -> decltype((*this)[i]) {
                    return (*this)[i];
                }

                long size() const {
                    return arg.size();
                }

            };

        /* Expression template for numpy expressions - unary operators
         */
        template<class Op, class Arg>
            struct numpy_uexpr {
                typedef const_nditerator<numpy_uexpr<Op, Arg>> iterator;
                static constexpr size_t value = std::remove_reference<Arg>::type::value;
                typedef decltype(Op()(std::declval<typename std::remove_reference<Arg>::type::value_type>())) value_type;
                typedef decltype(Op()(std::declval<typename std::remove_reference<Arg>::type::dtype>())) dtype;

                Arg arg;
                array<long, value> shape;

                numpy_uexpr() {}
                numpy_uexpr(numpy_uexpr const &) =default;
                numpy_uexpr(numpy_uexpr &&) = default;

                numpy_uexpr(Arg const &arg) : arg(arg), shape(arg.shape) {}

                iterator begin() const { return iterator(*this, 0); }
                iterator end() const { return iterator(*this, shape[0]); }

                auto fast(long i) const -> decltype(Op()(arg[i])) {
                    return Op()(arg[i]);
                }
                auto operator[](long i) const -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                template<class F>
                    typename std::enable_if<is_numexpr_arg<F>::value, numpy_fexpr<numpy_uexpr, F>>::type
                    operator[](F const& filter) const {
                        return numpy_fexpr<numpy_uexpr, F>(*this, filter);
                    }

                long size() const { return arg.size(); }
            };


        /* Expression template for numpy expressions - indexing
         */
        template<size_t N>
            struct numpy_iexpr_helper;

        template<class Arg>
            struct numpy_iexpr {
                static constexpr size_t value = std::remove_reference<Arg>::type::value - 1;
                typedef typename std::remove_reference<decltype(std::declval<typename std::remove_reference<Arg>::type::value_type>()[0L])>::type value_type;
                typedef typename std::remove_reference<Arg>::type::dtype dtype;

                typedef nditerator<numpy_iexpr> iterator;
                typedef const_nditerator<numpy_iexpr> const_iterator;

                Arg arg;
                dtype* buffer;
                array<long, value> shape;

                numpy_iexpr() {}
                numpy_iexpr(numpy_iexpr const&) = default;
                numpy_iexpr(numpy_iexpr&&) = default;

                numpy_iexpr(Arg const &arg, long index) : arg(arg), buffer(arg.buffer)
                {
                    auto siter = shape.begin();
                    for(auto iter = arg.shape.begin() + 1, end = arg.shape.end(); iter != end; ++iter, ++siter)
                        index *= *siter = *iter;
                    buffer += index;
                }

                template<class E>
                numpy_iexpr& operator=(E const& expr) {
                    return broadcast_copy(*this, expr, utils::int_<value - dim_of<E>::value>());
                }
                numpy_iexpr& operator=(numpy_iexpr const& expr) {
                    return broadcast_copy(*this, expr, utils::int_<value - dim_of<numpy_iexpr>::value>());
                }
                template<class E>
                numpy_iexpr& operator+=(E const& expr) {
                    return (*this) = (*this) + expr;
                }
                numpy_iexpr& operator+=(numpy_iexpr const& expr) {
                    return (*this) = (*this) + expr;
                }
                template<class E>
                numpy_iexpr& operator-=(E const& expr) {
                    return (*this) = (*this) - expr;
                }
                numpy_iexpr& operator-=(numpy_iexpr const& expr) {
                    return (*this) = (*this) - expr;
                }
                template<class E>
                numpy_iexpr& operator*=(E const& expr) {
                    return (*this) = (*this) * expr;
                }
                numpy_iexpr& operator*=(numpy_iexpr const& expr) {
                    return (*this) = (*this) * expr;
                }
                template<class E>
                numpy_iexpr& operator/=(E const& expr) {
                    return (*this) = (*this) / expr;
                }
                numpy_iexpr& operator/=(numpy_iexpr const& expr) {
                    return (*this) = (*this) / expr;
                }

                const_iterator begin() const { return const_iterator(*this, 0); }
                const_iterator end() const { return const_iterator(*this, shape[0]); }

                iterator begin() { return iterator(*this, 0); }
                iterator end() { return iterator(*this, shape[0]); }

                auto fast(long i) const -> decltype(numpy_iexpr_helper<value>::get(*this, i)) {
                    return numpy_iexpr_helper<value>::get(*this, i);
                }
                auto fast(long i) -> decltype(numpy_iexpr_helper<value>::get(*this, i)) {
                    return numpy_iexpr_helper<value>::get(*this, i);
                }
                auto operator[](long i) const -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator[](long i) -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator()(long i) const -> decltype((*this)[i]) {
                    return (*this)[i];
                }
                auto operator()(long i) -> decltype((*this)[i]) {
                    return (*this)[i];
                }
                numpy_gexpr<numpy_iexpr, slice> operator()(slice const& s0) const
                {
                    return numpy_gexpr<numpy_iexpr, slice>(*this, s0);
                }
                numpy_gexpr<numpy_iexpr, slice> operator[](slice const& s0) const
                {
                    return numpy_gexpr<numpy_iexpr, slice>(*this, s0);
                }
                numpy_gexpr<numpy_iexpr, contiguous_slice> operator()(contiguous_slice const& s0) const
                {
                    return numpy_gexpr<numpy_iexpr, contiguous_slice>(*this, s0);
                }
                numpy_gexpr<numpy_iexpr, contiguous_slice> operator[](contiguous_slice const& s0) const
                {
                    return numpy_gexpr<numpy_iexpr, contiguous_slice>(*this, s0);
                }
                template<class ...S>
                    numpy_gexpr<numpy_iexpr, slice, S...> operator()(slice const& s0, S const&... s) const
                    {
                        return numpy_gexpr<numpy_iexpr, slice, S...>(*this, s0, s...);
                    }
                template<class ...S>
                    numpy_gexpr<numpy_iexpr, contiguous_slice, S...> operator()(contiguous_slice const& s0, S const&... s) const
                    {
                        return numpy_gexpr<numpy_iexpr, contiguous_slice, S...>(*this, s0, s...);
                    }
                template<class ...S>
                    auto operator()(long s0, S const&... s) const -> decltype( (*this)[s0](s...))
                    {
                        return (*this)[s0](s...);
                    }
                template<class F>
                    typename std::enable_if<is_numexpr_arg<F>::value, numpy_fexpr<numpy_iexpr, F>>::type
                    operator[](F const& filter) const {
                        return numpy_fexpr<numpy_iexpr, F>(*this, filter);
                    }

                long size() const { return /*arg.size()*/ std::accumulate(shape.begin() + 1, shape.end(), *shape.begin(), std::multiplies<long>()); }
            };

        template <size_t N>
            struct numpy_iexpr_helper {
                template<class E>
                static numpy_iexpr<E> get(E && e, long i) { return numpy_iexpr<E>(e, i);}
            };

        template <>
            struct numpy_iexpr_helper<1> {
                template<class E>
                static typename E::dtype get(E const & e, long i) 
                {
                    return e.buffer[i];
                }
                template<class E>
                static typename E::dtype & get(E & e, long i) 
                {
                    return e.buffer[i];
                }
            };

        /* helper that yields true if the first slice of a pack is a contiguous slice
         */
        template<class... S>
            struct is_contiguous {
                static const bool value = false;
            };
        template<class... S>
            struct is_contiguous<contiguous_slice, S...> {
                static const bool value = true;
            };


        /* Expression template for numpy expressions - extended slicing operators
         */

        /* Meta-Function to count the number of slices in a type list
         */
        template<class... Types>
            struct count_long;
        template<>
            struct count_long<long> {
                static constexpr size_t value = 1;
            };
        template<>
            struct count_long<slice> {
                static constexpr size_t value = 0;
            };
        template<>
            struct count_long<contiguous_slice> {
                static constexpr size_t value = 0;
            };

        template<class T, class... Types>
            struct count_long<T, Types...> {
                static constexpr size_t value = count_long<T>::value + count_long<Types...>::value;
            };
        template<>
            struct count_long<> {
                static constexpr size_t value = 0;
            };

        /* helper to get the type of the nth element of an array
         */
        template<class T, size_t N>
            struct nth_value_type {
                typedef typename nth_value_type<typename T::value_type, N-1>::type type;
            };

        template<class T>
            struct nth_value_type<T, 0> {
                typedef T type;
            };
        template <class Arg, class... S>
            struct numpy_gexpr_helper;

        template<class Arg, class... S>
            struct numpy_gexpr {

                typedef typename std::remove_reference<decltype(std::declval<typename nth_value_type<typename std::remove_reference<Arg>::type, count_long<S...>::value>::type>()[0L])>::type value_type;
                typedef typename std::remove_reference<Arg>::type::dtype dtype;
                static constexpr size_t value = std::remove_reference<Arg>::type::value - count_long<S...>::value;

                typedef nditerator<numpy_gexpr<Arg, S...>> iterator;
                typedef const_nditerator<numpy_gexpr<Arg, S...>> const_iterator;

                Arg arg;
                dtype* buffer;
                array<long, value> shape;
                array<long, value> lower;
                array<long, value> step;
                array<long, std::remove_reference<Arg>::type::value - value> indices;

                numpy_gexpr() {}
                numpy_gexpr(numpy_gexpr const& ) = default;
                numpy_gexpr(numpy_gexpr && ) = default;

                template<size_t J>
                void init_shape(std::tuple<S const &...> const& values, contiguous_slice const& cs, utils::int_<1>, utils::int_<J>) {
                    contiguous_normalized_slice cns = cs.normalize(arg.shape[sizeof...(S) - 1]);
                    lower[J] = cns.lower;
                    step[J] = cns.step;
                    shape[J] = cns.size();
                }

                template<size_t I, size_t J>
                    void init_shape(std::tuple<S const&...> const & values, contiguous_slice const& cs, utils::int_<I>, utils::int_<J>) {
                        contiguous_normalized_slice cns = cs.normalize(arg.shape[sizeof...(S) - I]);
                        lower[J] = cns.lower;
                        step[J] = cns.step;
                        shape[J] = cns.size();
                        init_shape(values, std::get<sizeof...(S) - I + 1>(values), utils::int_<I - 1>(), utils::int_<J + 1>());
                    }
                template<size_t J>
                void init_shape(std::tuple<S const &...> const& values, slice const& cs, utils::int_<1>, utils::int_<J>) {
                    normalized_slice cns = cs.normalize(arg.shape[sizeof...(S) - 1]);
                    lower[J] = cns.lower;
                    step[J] = cns.step;
                    shape[J] = cns.size();
                }

                template<size_t I, size_t J>
                    void init_shape(std::tuple<S const&...> const & values, slice const& cs, utils::int_<I>, utils::int_<J>) {
                        normalized_slice cns = cs.normalize(arg.shape[sizeof...(S) - I]);
                        lower[J] = cns.lower;
                        step[J] = cns.step;
                        shape[J] = cns.size();
                        init_shape(values, std::get<sizeof...(S) - I + 1>(values), utils::int_<I - 1>(), utils::int_<J + 1>());
                    }

                template<size_t J>
                void init_shape(std::tuple<S const &...> const& values, long cs, utils::int_<1>, utils::int_<J>) {
                    if(cs < 0) cs += arg.shape[sizeof...(S) - 1];
                    indices[sizeof...(S) - 1 - J] = cs;
                }

                template<size_t I, size_t J>
                    void init_shape(std::tuple<S const&...> const & values, long cs, utils::int_<I>, utils::int_<J>) {
                        if(cs < 0) cs += arg.shape[sizeof...(S) - I];
                        indices[sizeof...(S) - I - J] = cs;
                        init_shape(values, std::get<sizeof...(S) - I + 1>(values), utils::int_<I - 1>(), utils::int_<J>());
                    }

                numpy_gexpr(Arg const &arg, S const &...s) : arg(arg), buffer(arg.buffer) {
                    std::tuple<S const&...> values(s...);
                    init_shape(values, std::get<0>(values), utils::int_<sizeof...(S)>(), utils::int_<0>());
                    for(size_t i = sizeof...(S) - count_long<S...>::value; i < value; ++i)
                        shape[i] = arg.shape[i];
                }

                template<class Argp, class... Sp>
                numpy_gexpr(numpy_gexpr<Argp, Sp...> const &expr, Arg &&arg) : arg(std::move(arg)), buffer(arg.buffer) {
                    std::copy(expr.shape.begin()+1, expr.shape.end(), shape.begin());
                    std::copy(expr.lower.begin()+1, expr.lower.end(), lower.begin());
                    std::copy(expr.step.begin()+1, expr.step.end(), step.begin());
                    std::copy(expr.indices.begin(), expr.indices.end(), indices.begin());
                }

                template<class G>
                numpy_gexpr(G const &expr, Arg &&arg) : arg(std::move(arg)), buffer(arg.buffer) {
                    std::copy(expr.shape.begin()+1, expr.shape.end(), shape.begin());
                    std::copy(expr.lower.begin()+1, expr.lower.end(), lower.begin());
                    std::copy(expr.step.begin()+1, expr.step.end(), step.begin());
                }

                template<class E>
                numpy_gexpr& operator=(E const& expr) {
                    return broadcast_copy(*this, expr, utils::int_<value - dim_of<E>::value>());
                }
                numpy_gexpr& operator=(numpy_gexpr const& expr) {
                    return broadcast_copy(*this, expr, utils::int_<value - dim_of<numpy_gexpr>::value>());
                }
                template<class E>
                numpy_gexpr& operator+=(E const& expr) {
                    return (*this) = (*this) + expr;
                }
                numpy_gexpr& operator+=(numpy_gexpr const& expr) {
                    return (*this) = (*this) + expr;
                }
                template<class E>
                numpy_gexpr& operator-=(E const& expr) {
                    return (*this) = (*this) - expr;
                }
                numpy_gexpr& operator-=(numpy_gexpr const& expr) {
                    return (*this) = (*this) - expr;
                }
                template<class E>
                numpy_gexpr& operator*=(E const& expr) {
                    return (*this) = (*this) * expr;
                }
                numpy_gexpr& operator*=(numpy_gexpr const& expr) {
                    return (*this) = (*this) * expr;
                }
                template<class E>
                numpy_gexpr& operator/=(E const& expr) {
                    return (*this) = (*this) / expr;
                }
                numpy_gexpr& operator/=(numpy_gexpr const& expr) {
                    return (*this) = (*this) / expr;
                }

                const_iterator begin() const { return const_iterator(*this, 0); }
                const_iterator end() const { return const_iterator(*this, shape[0]); }

                iterator begin() { return iterator(*this, 0); }
                iterator end() { return iterator(*this, shape[0]); }

                auto fast(long i) const -> decltype(numpy_gexpr_helper<Arg, S...>::get(*this, i)) {
                    return numpy_gexpr_helper<Arg, S...>::get(*this, lower[0] + (is_contiguous<S...>::value ? i : step[0] * i));
                }
                auto fast(long i) -> decltype(numpy_gexpr_helper<Arg, S...>::get(*this, i)) {
                    return numpy_gexpr_helper<Arg, S...>::get(*this, lower[0] + (is_contiguous<S...>::value ? i : step[0] * i));
                }
                auto operator[](long i) const -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator[](long i) -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator()(long i) const -> decltype((*this)[i]) {
                    return (*this)[i];
                }
                auto operator()(long i) -> decltype((*this)[i]) {
                    return (*this)[i];
                }
                numpy_gexpr operator[](slice const& s) const
                {
                    normalized_slice ns = s.normalize(shape[0]);
                    numpy_gexpr other = (*this);
                    other.shape[0] = ns.size();
                    other.lower[0] += ns.lower;
                    other.step[0] *= ns.step;
                    return other;
                }
                numpy_gexpr operator[](contiguous_slice const& s) const
                {
                    contiguous_normalized_slice ns = s.normalize(shape[0]);
                    numpy_gexpr other = (*this);
                    other.shape[0] += ns.size();
                    other.lower[0] += ns.lower;
                }
                template<class F>
                    typename std::enable_if<is_numexpr_arg<F>::value, numpy_fexpr<numpy_gexpr, F>>::type
                    operator[](F const& filter) const {
                        return numpy_fexpr<numpy_gexpr, F>(*this, filter);
                    }

                long size() const {
                    return std::accumulate(shape.begin() + 1, shape.end(), *shape.begin(), std::multiplies<long>());
                }
            };

        template <class Arg, class S0, class S1, class...S>
            struct numpy_gexpr_helper<Arg, S0, S1, S...> {
                typedef numpy_gexpr<numpy_iexpr<Arg>, S1, S...> type;
                static type get(numpy_gexpr<Arg, S0, S1, S...> const& e, long i) {
                    return type(e, numpy_iexpr<Arg>(e.arg, i));
                }
                static type get(numpy_gexpr<Arg, S0, S1, S...> & e, long i) {
                    return type(e, numpy_iexpr<Arg>(e.arg, i));
                }
            };


        namespace {

            template <size_t N, class Arg, class...S>
                struct finalize_numpy_gexpr_helper;

            template <size_t N, class Arg, class... S>
                struct finalize_numpy_gexpr_helper<N, Arg, contiguous_slice, S...> {

                    typedef numpy_gexpr<Arg, contiguous_slice, S...> type;
                    template<class E, class F>
                        static type get(E const& e, F && f) {
                            return type(e, std::move(f));
                        }
                };

            template <size_t N, class Arg, class... S>
                struct finalize_numpy_gexpr_helper<N, Arg, slice, S...> {

                    typedef numpy_gexpr<Arg, slice, S...> type;
                    template<class E, class F>
                        static type get(E const& e, F && f) {
                            return type(e, std::move(f));
                        }
                };

            template <size_t N, class Arg, class... S>
                struct finalize_numpy_gexpr_helper<N, Arg, long, S...> {
                    template<class E, class F>
                        static auto get(E const& e, F && f)
                        -> decltype(finalize_numpy_gexpr_helper<N + 1, numpy_iexpr<Arg>, S...>::get(e, std::declval<numpy_iexpr<Arg>>()))
                        {
                            return finalize_numpy_gexpr_helper<N + 1, numpy_iexpr<Arg>, S...>::get(e, numpy_iexpr<Arg>(std::move(f), e.indices[N]));
                        }
                    template<class E, class F>
                        static auto get(E & e, F && f)
                        -> decltype(finalize_numpy_gexpr_helper<N + 1, numpy_iexpr<Arg>, S...>::get(e, std::declval<numpy_iexpr<Arg>&>()))
                        {
                            numpy_iexpr<Arg> iexpr(std::move(f), e.indices[N]);
                            return finalize_numpy_gexpr_helper<N + 1, numpy_iexpr<Arg>, S...>::get(e, iexpr);
                        }
                };

            template <size_t N, class Arg>
                struct finalize_numpy_gexpr_helper<N, Arg, long> {
                    template<class E, class F>
                        static auto get(E const& e, F const & f)
                        -> decltype(numpy_iexpr_helper<Arg::value>::get(f, 0))
                        {
                            return numpy_iexpr_helper<Arg::value>::get(f, e.indices[N]);
                        }
                    template<class E, class F>
                        static auto get(E const& e, F & f)
                        -> decltype(numpy_iexpr_helper<Arg::value>::get(f, 0))
                        {
                            return numpy_iexpr_helper<Arg::value>::get(f, e.indices[N]);
                        }
                };
        }

        template <class Arg, class S0, class... S>
            struct numpy_gexpr_helper<Arg, S0, long, S...> {
                static auto get(numpy_gexpr<Arg, S0, long, S...> const& e, long i)
                    ->decltype(finalize_numpy_gexpr_helper<0, numpy_iexpr<Arg>, long, S...>::get(e, std::declval<numpy_iexpr<Arg>>()))
                {
                    return finalize_numpy_gexpr_helper<0, numpy_iexpr<Arg>, long, S...>::get(e, numpy_iexpr<Arg>(e.arg, i));
                }
                static auto get(numpy_gexpr<Arg, S0, long, S...> & e, long i)
                    ->decltype(finalize_numpy_gexpr_helper<0, numpy_iexpr<Arg>, long, S...>::get(e, std::declval<numpy_iexpr<Arg>&>()))
                {
                    return finalize_numpy_gexpr_helper<0, numpy_iexpr<Arg>, long, S...>::get(e, numpy_iexpr<Arg>(e.arg, i));
                }
            };

        template <class Arg, class S>
            struct numpy_gexpr_helper<Arg, S, long> {
                static auto get(numpy_gexpr<Arg, S, long> const& e, long i)
                    -> decltype(numpy_iexpr_helper<numpy_iexpr<Arg>::value>::get(std::declval<numpy_iexpr<Arg>>(), 0))
                {
                    return numpy_iexpr_helper<numpy_iexpr<Arg>::value>::get(numpy_iexpr<Arg>(e.arg, i), e.indices[0]);
                }
                static auto get(numpy_gexpr<Arg, S, long> & e, long i)
                    -> decltype(numpy_iexpr_helper<numpy_iexpr<Arg>::value>::get(std::declval<numpy_iexpr<Arg>&>(), 0))
                {
                    numpy_iexpr<Arg> iexpr(e.arg, i);
                    return numpy_iexpr_helper<numpy_iexpr<Arg>::value>::get(iexpr, e.indices[0]);
                }
            };

        template <class Arg, class S>
            struct numpy_gexpr_helper<Arg, S> : numpy_iexpr_helper<numpy_gexpr<Arg, S>::value> {
            };

        /* utility to pick the right shape */
        template<class U, class V, size_t N>
            typename std::enable_if<U::value!=0 and U::value == N, array<long, U::value>>::type select_shape(U const& u, V const&, utils::int_<N> ) {
                return u.shape;
            }
        template<class U, class V, size_t N>
            typename std::enable_if<U::value!=0 and U::value != N, array<long, V::value>>::type select_shape(U const& , V const& v, utils::int_<N> ) {
                return v.shape;
            }
        template<class U, class V, size_t N>
            typename std::enable_if<U::value==0 and V::value!=0, array<long, V::value>>::type select_shape(U const& , V const&v, utils::int_<N> ) {
                return v.shape;
            }
        template<class U, class V>
            array<long, 0> select_shape(U const& u, V const&, utils::int_<0> ) {
                return array<long, 0>();
            }

        /* Expression template for numpy expressions - binary operators
         */
        template<class Op, class Arg0, class Arg1>
            struct numpy_expr {
                typedef const_nditerator<numpy_expr<Op, Arg0, Arg1>> iterator;
                static constexpr size_t value = std::remove_reference<Arg0>::type::value>std::remove_reference<Arg1>::type::value?std::remove_reference<Arg0>::type::value: std::remove_reference<Arg1>::type::value;
                typedef decltype(Op()(std::declval<typename std::remove_reference<Arg0>::type::value_type>(), std::declval<typename std::remove_reference<Arg1>::type::value_type>())) value_type;
                typedef decltype(Op()(std::declval<typename std::remove_reference<Arg0>::type::dtype>(), std::declval<typename std::remove_reference<Arg1>::type::dtype>())) dtype;

                typename std::remove_reference<Arg0>::type arg0;
                typename std::remove_reference<Arg1>::type arg1;
                array<long, value> shape;

                numpy_expr() {}
                numpy_expr(numpy_expr const&) = default;
                numpy_expr(numpy_expr &&) = default;

                numpy_expr(Arg0 const &arg0, Arg1 const &arg1) : arg0(arg0), arg1(arg1), shape(select_shape(arg0,arg1, utils::int_<value>())) {}

                iterator begin() const { return iterator(*this, 0); }
                iterator end() const { return iterator(*this, shape[0]); }

                auto fast(long i) const -> decltype(Op()(arg0[i], arg1[i])) {
                    return Op()(arg0[i], arg1[i]);
                }
                auto operator[](long i) const -> decltype(this->fast(i)) {
                    if(i<0) i += shape[0];
                    return fast(i);
                }

                template<class F>
                    typename std::enable_if<is_numexpr_arg<F>::value, numpy_fexpr<numpy_expr, F>>::type
                    operator[](F const& filter) const {
                        return numpy_fexpr<numpy_expr, F>(*this, filter);
                    }

                long size() const { return std::max(arg0.size(), arg1.size()); }
            };


        /* Helper for dimension-specific part of ndarray
         *
         * Instead of specializing the whole ndarray class, the dimension-specific behavior are stored here.
         */

        template<class T, size_t N>
            struct type_helper<ndarray<T,N>> {
                typedef ndarray<T,N-1> type;

                typedef nditerator<ndarray<T,N>> iterator;
                typedef const_nditerator<ndarray<T,N>> const_iterator;

                static iterator make_iterator(ndarray<T,N>& n, long i) { return iterator(n, i); }
                static const_iterator make_iterator(ndarray<T,N> const & n, long i) { return const_iterator(n, i); }

                template<class S, class Iter>
                    static T* initialize_from_iterable(S& shape, T* from, Iter&& iter) {
                        shape[std::tuple_size<S>::value - N] = iter.size();
                        for(auto content : iter) {
                            from = type_helper<type>::initialize_from_iterable(shape, from, content);
                        }
                        return from;
                    }

                static numpy_iexpr<ndarray<T,N> const&> get(ndarray<T,N> const& self, long i) {
                    return numpy_iexpr<ndarray<T,N> const &>(self, i);
                }
                static long step(ndarray<T,N> const& self) { return std::accumulate(self.shape.begin() + 1, self.shape.end(), 1L, std::multiplies<long>());}
            };

        template<class T>
            struct type_helper<ndarray<T,1>> {
                typedef T type;

                typedef T* iterator;
                typedef T const * const_iterator;

                static iterator make_iterator(ndarray<T,1>& n, long i) { return n.buffer + i; }
                static const_iterator make_iterator(ndarray<T,1> const & n, long i) { return n.buffer + i; }

                template<class S, class Iter>
                    static T* initialize_from_iterable(S& shape, T* from, Iter&& iter) {
                        shape[std::tuple_size<S>::value - 1] = iter.size();
                        return std::copy(iter.begin(), iter.end(), from);
                    }
                static type& get(ndarray<T,1> const& self, long i) {
                    return self.buffer[i];
                }
                static constexpr long step(ndarray<T,1> const& ) { return 1;}
            };

        template<size_t L>
            struct nget {
                template <class A, size_t M> 
                    auto operator()(A const & self, array<long, M> const& indices)
                    -> decltype(nget<L-1>()(self[0], indices))
                    {
                        return nget<L-1>()(self[indices[M - L - 1]], indices);
                    }
            };
        template<>
            struct nget<0> {
                template<class A, size_t M>
                    auto operator()(A const & self, array<long, M> const& indices)
                    -> decltype(self[indices[M - 1]])
                    {
                        return self[indices[M - 1]];
                    }
            };

        /* Multidimensional array of values
         *
         * An ndarray wraps a raw array pointers and manages multiple dimensions casted overt the raw data.
         * The number of dimensions is fixed as well as the type of the underlying data.
         * A shared pointer is used internally to mimic Python's behavior.
         *
         */
        template<class T, size_t N>
            struct ndarray {

                /* types */
                static const size_t value;
                typedef T dtype;
                typedef typename type_helper<ndarray<T, N>>::type value_type;
                typedef value_type& reference;
                typedef value_type const & const_reference;

                typedef typename type_helper<ndarray<T, N>>::iterator iterator;
                typedef typename type_helper<ndarray<T, N>>::const_iterator const_iterator;
                typedef T* flat_iterator;
                typedef T const * const_flat_iterator;

                /* members */
                utils::shared_ref<raw_array<T>> mem;     // shared data pointer
                T* buffer;                              // pointer to the first data stored in the equivalent flat array
                array<long, N> shape;             // shape of the multidimensional array

                /* constructors */
                ndarray() : mem(utils::no_memory()), buffer(nullptr), shape() {}
                ndarray(ndarray const & ) = default;
                ndarray(ndarray && ) = default;

                /* assignment */
                ndarray& operator=(ndarray const& other) = default;

                /* from other memory */
                ndarray(utils::shared_ref<raw_array<T>> const &mem, array<long,N> const& shape):
                    mem(mem),
                    buffer(mem->data),
                    shape(shape)
                {
                }

                /* from other array */
                template<class Tp, size_t Np>
                    ndarray(ndarray<Tp, Np> const& other):
                        mem(other.size()),
                        buffer(mem->data),
                        shape(other.shape)
                {
                    std::copy(other.fbegin(), other.fend(), fbegin());
                }

                /* from a seed */
                ndarray(array<long, N> const& shape, none_type init ):
                    mem(std::accumulate(shape.begin() + 1, shape.end(), *shape.begin(), std::multiplies<long>())),
                    buffer(mem->data),
                    shape(shape)
                {
                }
                ndarray(array<long, N> const& shape, T init ): ndarray(shape, none_type())
                {
                    std::fill(fbegin(), fend(), init);
                }

                /* from a foreign pointer */
                ndarray(T* data, long * pshape):
                    mem(data),
                    buffer(mem->data),
                    shape()
                {
                    std::copy(pshape, pshape + N, shape.begin());
                }

                ndarray(T* data, long * pshape, foreign const&): ndarray(data, pshape)
                {
                    mem.external(); // make sure we do not releas the pointer
                }
                template<class Iterable,
                    class = typename std::enable_if<
                        !is_array<typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type>::value
                        and is_iterable<typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type>::value,
                    void>::type
                        >
                        ndarray(Iterable&& iterable):
                            mem(utils::nested_container_size<Iterable>::size(std::forward<Iterable>(iterable))),
                            buffer(mem->data),
                            shape()
                {
                    type_helper<ndarray<T,N>>::initialize_from_iterable(shape, mem->data, std::forward<Iterable>(iterable));
                }

                /* from a  numpy expression */
                template<class E>
                    void initialize_from_expr(E const & expr) {
                        std::copy(expr.begin(), expr.end(), begin());
                    }

                template<class Op, class Arg0, class Arg1>
                    ndarray(numpy_expr<Op, Arg0, Arg1> const & expr) :
                        mem(expr.size()),
                        buffer(mem->data),
                        shape(expr.shape)
                {
                    initialize_from_expr(expr);
                }

                template<class Arg>
                    ndarray(numpy_texpr<Arg> const & expr) :
                        mem(expr.size()),
                        buffer(mem->data),
                        shape(expr.shape)
                {
                    initialize_from_expr(expr);
                }

                template<class Op, class Arg>
                    ndarray(numpy_uexpr<Op, Arg> const & expr) :
                        mem(expr.size()),
                        buffer(mem->data),
                        shape(expr.shape)
                {
                    initialize_from_expr(expr);
                }

                template<class Arg, class... S>
                    ndarray(numpy_gexpr<Arg, S...> const & expr) :
                        mem(expr.size()),
                        buffer(mem->data),
                        shape(expr.shape)
                {
                    initialize_from_expr(expr);
                }

                template<class Arg>
                    ndarray(numpy_iexpr<Arg> const & expr) :
                        mem(expr.size()),
                        buffer(mem->data),
                        shape(expr.shape)
                {
                    initialize_from_expr(expr);
                }

                template<class Arg, class F>
                    ndarray(numpy_fexpr<Arg, F> const & expr) :
                        mem(expr.size()),
                        buffer(mem->data),
                        shape(expr.shape)
                {
                    initialize_from_expr(expr);
                }

                /* element indexing */
                auto fast(long i) const -> decltype(type_helper<ndarray<T,N>>::get(*this, i))
                {
                    return type_helper<ndarray<T,N>>::get(*this, i);
                }
                auto fast(long i) -> decltype(type_helper<ndarray<T,N>>::get(*this, i))
                {
                    return type_helper<ndarray<T,N>>::get(*this, i);
                }
                auto operator[](long i) const -> decltype(this->fast(i))
                {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator[](long i) -> decltype(this->fast(i))
                {
                    if(i<0) i += shape[0];
                    return fast(i);
                }
                auto operator()(long i) const -> decltype((*this)[i])
                {
                    return (*this)[i];
                }
                auto operator()(long i) -> decltype((*this)[i])
                {
                    return (*this)[i];
                }
                T operator[](array<long, N> const& indices) const
                {
                    size_t offset = indices[N-1];
                    long mult = shape[N-1];
                    for(size_t i = N - 2; i > 0; --i) {
                        offset +=  indices[i] * mult;
                        mult *= shape[i];
                    }
                    return buffer[offset + indices[0] * mult];
                }
                T& operator[](array<long, N> const& indices)
                {
                    size_t offset = indices[N-1];
                    long mult = shape[N-1];
                    for(size_t i = N - 2; i > 0; --i) {
                        offset +=  indices[i] * mult;
                        mult *= shape[i];
                    }
                    return buffer[offset + indices[0] * mult];
                }
                template<size_t M>
                    auto operator[](array<long, M> const& indices) const
                    -> decltype(nget<M-1>()(*this, indices))
                    {
                        return nget<M-1>()(*this, indices);
                    }

                /* slice indexing */
                numpy_gexpr<ndarray<T,N> const &, slice> operator[](slice const& s) const
                {
                    return numpy_gexpr<ndarray<T,N> const &, slice>(*this, s);
                }

                numpy_gexpr<ndarray<T,N> const &, contiguous_slice> operator[](contiguous_slice const& s) const
                {
                    return numpy_gexpr<ndarray<T,N> const &, contiguous_slice>(*this, s);
                }
                numpy_gexpr<ndarray<T,N> const &, slice> operator()(slice const& s) const
                {
                    return numpy_gexpr<ndarray<T,N> const &, slice>(*this, s);
                }

                numpy_gexpr<ndarray<T,N> const &, contiguous_slice> operator()(contiguous_slice const& s) const
                {
                    return numpy_gexpr<ndarray<T,N> const &, contiguous_slice>(*this, s);
                }

                /* extended slice indexing */
                template<class ...S>
                    numpy_gexpr<ndarray<T,N> const &, slice, S...> operator()(slice const& s0, S const&... s) const
                    {
                        return numpy_gexpr<ndarray<T,N> const &, slice, S...>(*this, s0, s...);
                    }
                template<class ...S>
                    numpy_gexpr<ndarray<T,N> const &, contiguous_slice, S...> operator()(contiguous_slice const& s0, S const&... s) const
                    {
                        return numpy_gexpr<ndarray<T,N> const &, contiguous_slice, S...>(*this, s0, s...);
                    }
                template<class ...S>
                    auto operator()(long s0, S const&... s) const -> decltype( (*this)[s0](s...))
                    {
                        return (*this)[s0](s...);
                    }

                /* element filtering */
                template<class F>
                    typename std::enable_if<is_numexpr_arg<F>::value, numpy_fexpr<ndarray, F>>::type
                    operator[](F const& filter) const {
                        return numpy_fexpr<ndarray, F>(*this, filter);
                    }

                /* through iterators */
                iterator begin() { return type_helper<ndarray<T,N>>::make_iterator(*this, 0); }
                const_iterator begin() const { return type_helper<ndarray<T,N>>::make_iterator(*this, 0); }
                iterator end() { return type_helper<ndarray<T,N>>::make_iterator(*this, shape[0]); }
                const_iterator end() const { return type_helper<ndarray<T,N>>::make_iterator(*this, shape[0]); }

                const_flat_iterator fbegin() const { return buffer; }
                const_flat_iterator fend() const { return buffer + size(); }
                flat_iterator fbegin() { return buffer; }
                flat_iterator fend() { return buffer + size(); }

                /* member functions */
                long size() const { return std::accumulate(shape.begin() + 1, shape.end(), *shape.begin(), std::multiplies<long>()); }
                template<size_t M>
                    ndarray<T,M> reshape(array<long,M> const& shape) const {
                        return ndarray<T, M>(mem, shape);
                    }
                ndarray<T,1> flat() const {
                    return ndarray<T, 1>(mem, array<long, 1>{{size()}});
                }
                ndarray<T,N> copy() const {
                    auto res = ndarray<T,N>(shape, __builtin__::None);
                    std::copy(fbegin(), fend(), res.fbegin());
                    return res;
                }
                intptr_t id() const {
                    return reinterpret_cast<intptr_t>(&(*mem));
                }

            };

        template<class T, size_t N>
            size_t const ndarray<T,N>::value = N;


        /* } */
        /* pretty printing { */
        template<class T, size_t N>
            std::ostream& operator<<(std::ostream& os, ndarray<T,N> const& e)
            {
                std::array<long, N> strides;
                auto shape = e.shape;
                strides[N-1] = shape[N-1];
                if(strides[N-1]==0)
                    return os << "[]";
                std::transform(strides.rbegin(), strides.rend() -1, shape.rbegin() + 1, strides.rbegin() + 1, std::multiplies<long>());
                size_t depth = N;
                int step = -1;
                std::ostringstream oss;
                if( e.size())
                    oss << *std::max_element(e.fbegin(), e.fend());
                int size = oss.str().length();
                auto iter = e.fbegin();
                int max_modulo = 1000;

                os << "[";
                if( shape[0] != 0)
                    do {
                        if(depth==1)
                        {
                            os.width(size);
                            os << *iter++;
                            for(int i=1; i<shape[N-1]; i++)
                            {
                                os.width(size+1);
                                os << *iter++;
                            }
                            step = 1;
                            depth++;
                            max_modulo = std::lower_bound(strides.begin(), strides.end(), iter - e.buffer, [](int comp, int val){ return val%comp!=0; }) - strides.begin();
                        }
                        else if(max_modulo + depth == N + 1)
                        {
                            depth--;
                            step = -1;
                            os << "]";
                            for(size_t i=0;i<depth;i++)
                                os << std::endl;
                            for(size_t i=0;i<N-depth;i++)
                                os << " ";
                            os << "[";
                        }
                        else
                        {
                            depth+=step;
                            if(step==1)
                                os << "]";
                            else
                                os << "[";
                        }
                    }
                    while(depth != N+1);

                return os << "]";
            }

        /* } */


        /*
         *
         * 3 informations are available:
         * - the type `type` of the equivalent array
         * - the number of dimensions `value` of the equivalent array
         * - the type `T` of the value contained by the equivalent array
         */
        template <class E, class _>
            struct numpy_expr_to_ndarray {
                typedef typename E::dtype T;
                static const size_t N = E::value;
                typedef ndarray<T, N> type;
            };
        template <class L, class _>
            struct numpy_expr_to_ndarray<list<L>, _> {
                typedef typename utils::nested_container_value_type<list<L>>::type T;
                static const size_t N = utils::nested_container_depth<list<L>>::value;
                typedef ndarray<T, N> type;
            };

        /* Type trait that checks if a type is a potential numpy expression parameter
         *
         * Only used to write concise expression templates
         */
        template<class T>
            struct is_array {
                static constexpr bool value = false;
            };
        template<class T, size_t N>
            struct is_array<ndarray<T,N>> {
                static constexpr bool value = true;
            };
        template<class A>
            struct is_array<numpy_iexpr<A>> {
                static constexpr bool value = true;
            };
        template<class A, class F>
            struct is_array<numpy_fexpr<A,F>> {
                static constexpr bool value = true;
            };
        template<class A, class... S>
            struct is_array<numpy_gexpr<A,S...>> {
                static constexpr bool value = true;
            };
        template<class O, class A>
            struct is_array<numpy_uexpr<O,A>> {
                static constexpr bool value = true;
            };
        template<class A>
            struct is_array<numpy_texpr<A>> {
                static constexpr bool value = true;
            };
        template<class O, class A0, class A1>
            struct is_array<numpy_expr<O,A0,A1>> {
                static constexpr bool value = true;
            };

        template<class T>
            struct is_numexpr_arg : is_array<T> {
            };
        template<class T>
            struct is_numexpr_arg<list<T>> {
                static constexpr bool value = true;
            };
    }

    /* make sure the size method from ndarray is not used */
    namespace __builtin__ {
        template <class T, size_t N, class I>
            struct _len<types::ndarray<T,N>, I, true> {
                long operator()(types::ndarray<T,N> const &t) {
                    return t.shape[0];
                }
            };

    }

    namespace utils {

        template<class Op, class Arg0, class Arg1>
            struct nested_container_depth<types::numpy_expr<Op, Arg0, Arg1>> {
                static const int  value = types::numpy_expr<Op, Arg0, Arg1>::value;
            };
    }

    template<class Op, class Arg0, class Arg1>
        struct assignable<types::numpy_expr<Op, Arg0, Arg1>>
        {
            typedef typename types::numpy_expr_to_ndarray<types::numpy_expr<Op, Arg0, Arg1>>::type type;
        };
    template<class Arg>
        struct assignable<types::numpy_texpr<Arg>>
        {
            typedef typename types::numpy_expr_to_ndarray<types::numpy_texpr<Arg>>::type type;
        };
    template<class Op, class Arg>
        struct assignable<types::numpy_uexpr<Op, Arg>>
        {
            typedef typename types::numpy_expr_to_ndarray<types::numpy_uexpr<Op, Arg>>::type type;
        };
    template<class Arg, class...S>
        struct assignable<types::numpy_gexpr<Arg, S...>>
        {
            typedef typename types::numpy_expr_to_ndarray<types::numpy_gexpr<Arg, S...>>::type type;
        };
    template<class Arg>
        struct assignable<types::numpy_iexpr<Arg>>
        {
            typedef typename types::numpy_expr_to_ndarray<types::numpy_iexpr<Arg>>::type type;
        };
    template<class Arg, class F>
        struct assignable<types::numpy_fexpr<Arg, F>>
        {
            typedef typename types::numpy_expr_to_ndarray<types::numpy_fexpr<Arg, F>>::type type;
        };

    template<class Op, class Arg0, class Arg1>
        struct lazy<types::numpy_expr<Op,Arg0,Arg1>>
        {
            typedef types::numpy_expr<Op,typename lazy<Arg0>::type, typename lazy<Arg1>::type> type;
        };
    template<class Op, class Arg>
        struct lazy<types::numpy_uexpr<Op,Arg>>
        {
            typedef types::numpy_uexpr<Op,typename lazy<Arg>::type> type;
        };
    template<class Arg, class F>
        struct lazy<types::numpy_fexpr<Arg,F>>
        {
            typedef types::numpy_fexpr<typename lazy<Arg>::type, typename lazy<F>::type> type;
        };
    template<class Arg>
        struct lazy<types::numpy_iexpr<Arg>>
        {
            typedef types::numpy_iexpr<Arg> type;
        };
    template<class Arg, class... S>
        struct lazy<types::numpy_gexpr<Arg, S...>>
        {
            typedef types::numpy_gexpr<Arg, S...> type;
        };
}

/* std::get overloads */
namespace std {

    template <size_t I, class E>
        auto get( E&& a)
        -> typename std::enable_if<pythonic::types::is_array<typename std::remove_cv<typename std::remove_reference<E>::type>::type>::value, decltype(a[I])>::type
        {
            return a[I];
        }

    template <size_t I, class T, size_t N>
        struct tuple_element<I, pythonic::types::ndarray<T,N> > {
            typedef typename pythonic::types::ndarray<T,N>::value_type type;
        };
    template <size_t I, class Op, class Arg0, class Arg1>
        struct tuple_element<I, pythonic::types::numpy_expr<Op,Arg0, Arg1> > {
            typedef typename pythonic::types::numpy_expr_to_ndarray<pythonic::types::numpy_expr<Op,Arg0, Arg1>>::type::value_type type;
        };
}

/* pythran attribute system { */
#include "pythonic/numpy/transpose.hpp"
namespace pythonic {
    namespace types {
        namespace __ndarray {

            template<int I, class E>
                struct getattr;

            template<class E> struct getattr<attr::SHAPE, E> {
                auto operator()(E const& a) -> decltype(a.shape) { return a.shape; }
            };
            template<class E> struct getattr<attr::NDIM, E> {
                long operator()(E const& a) { return numpy_expr_to_ndarray<E>::N; }
            };
            template<class E> struct getattr<attr::STRIDES, E> {
                array<long, numpy_expr_to_ndarray<E>::N> operator()(E const& a) {
                    array<long,numpy_expr_to_ndarray<E>::N> strides;
                    strides[numpy_expr_to_ndarray<E>::N-1] = sizeof(typename numpy_expr_to_ndarray<E>::T);
                    auto shape = a.shape;
                    std::transform(strides.rbegin(), strides.rend() -1, shape.rbegin(), strides.rbegin()+1, std::multiplies<long>());
                    return strides;
                }
            };
            template<class E> struct getattr<attr::SIZE, E> {
                long operator()(E const& a) { return a.size(); }
            };
            template<class E> struct getattr<attr::ITEMSIZE, E> {
                long operator()(E const& a) { return sizeof(typename numpy_expr_to_ndarray<E>::T); }
            };
            template<class E> struct getattr<attr::NBYTES, E> {
                long operator()(E const& a) { return a.size() * sizeof(typename numpy_expr_to_ndarray<E>::T); }
            };
            template<class E> struct getattr<attr::FLAT, E> {
                auto operator()(E const& a) -> decltype(a.flat()) { return a.flat(); }
            };
            template<class E> struct getattr<attr::DTYPE, E> {
                typename numpy_expr_to_ndarray<E>::T operator()(E const& a) { return typename numpy_expr_to_ndarray<E>::T(); }
            };
            template<class E> struct getattr<attr::T, E> {
                auto operator()(E const& a) -> decltype(numpy::transpose(a)) { return numpy::transpose(a); }
            };
        }
    }
    namespace __builtin__ {
        template<int I, class T, size_t N>
            auto getattr(types::ndarray<T,N> const& f)
            -> decltype(types::__ndarray::getattr<I,types::ndarray<T,N>>()(f))
            {
                return types::__ndarray::getattr<I,types::ndarray<T,N>>()(f);
            }
        template<int I, class O, class A0, class A1>
            auto getattr(types::numpy_expr<O,A0,A1> const& f)
            -> decltype(types::__ndarray::getattr<I,types::numpy_expr<O,A0,A1>>()(f))
            {
                return types::__ndarray::getattr<I,types::numpy_expr<O,A0,A1>>()(f);
            }
        template<int I, class A>
            auto getattr(types::numpy_texpr<A> const& f)
            -> decltype(types::__ndarray::getattr<I,types::numpy_texpr<A>>()(f))
            {
                return types::__ndarray::getattr<I,types::numpy_texpr<A>>()(f);
            }
        template<int I, class O, class A>
            auto getattr(types::numpy_uexpr<O,A> const& f)
            -> decltype(types::__ndarray::getattr<I,types::numpy_uexpr<O,A>>()(f))
            {
                return types::__ndarray::getattr<I,types::numpy_uexpr<O,A>>()(f);
            }
        template<int I, class A, class F>
            auto getattr(types::numpy_fexpr<A,F> const& f)
            -> decltype(types::__ndarray::getattr<I,types::numpy_fexpr<A,F>>()(f))
            {
                return types::__ndarray::getattr<I,types::numpy_fexpr<A,F>>()(f);
            }
        template<int I, class A, class... S>
            auto getattr(types::numpy_gexpr<A,S...> const& f)
            -> decltype(types::__ndarray::getattr<I,types::numpy_gexpr<A,S...>>()(f))
            {
                return types::__ndarray::getattr<I,types::numpy_gexpr<A,S...>>()(f);
            }
    }
}

/* } */

/* type inference stuff  {*/
#include "pythonic/types/combined.hpp"

template<size_t N, class T, size_t M>
struct __combined<pythonic::types::ndarray<T,N>, pythonic::types::ndarray<T,M>> {
    typedef pythonic::types::ndarray<T,N> type;
};
template<size_t N, class T, class O>
struct __combined<pythonic::types::ndarray<T,N>, O> {
    typedef pythonic::types::ndarray<T,N> type;
};
template<size_t N, class T, class O>
struct __combined<O, pythonic::types::ndarray<T,N>> {
    typedef pythonic::types::ndarray<T,N> type;
};

template<class Arg0, class Arg1, class Op, class K>
struct __combined<pythonic::types::numpy_expr<Op, Arg0, Arg1>, indexable<K>> {
    typedef pythonic::types::numpy_expr<Op, Arg0, Arg1> type;
};

template<class Arg0, class Arg1, class Op, class K>
struct __combined<indexable<K>, pythonic::types::numpy_expr<Op, Arg0, Arg1>> {
    typedef pythonic::types::numpy_expr<Op, Arg0, Arg1> type;
};

template<class Arg0, class Arg1, class Op, class K, class V>
struct __combined<pythonic::types::numpy_expr<Op, Arg0, Arg1>, indexable_container<K,V>> {
    typedef pythonic::types::numpy_expr<Op, Arg0, Arg1> type;
};

template<class Arg0, class Arg1, class Op, class K, class V>
struct __combined<indexable_container<K,V>, pythonic::types::numpy_expr<Op, Arg0, Arg1>> {
    typedef pythonic::types::numpy_expr<Op, Arg0, Arg1> type;
};

template<class Arg0, class Arg1, class Op, class K>
struct __combined<container<K>, pythonic::types::numpy_expr<Op, Arg0, Arg1>> {
    typedef pythonic::types::numpy_expr<Op, Arg0, Arg1> type;
};

template<class Arg0, class Arg1, class Op, class K>
struct __combined<pythonic::types::numpy_expr<Op, Arg0, Arg1>, container<K>> {
    typedef pythonic::types::numpy_expr<Op, Arg0, Arg1> type;
};

template<class Arg0, class Arg1, class Op, class Op2, class Arg2, class Arg3>
struct __combined<pythonic::types::numpy_expr<Op, Arg0, Arg1>, pythonic::types::numpy_expr<Op2, Arg2, Arg3>> {
    typedef typename pythonic::types::numpy_expr_to_ndarray<pythonic::types::numpy_expr<Op, Arg0, Arg1>>::type type;
};

//
// PB : This led to poor performance (but I don't understand why)
//
//template<class Arg0, class Arg1, class Op, class Arg2, class Arg3>
//struct __combined<pythonic::types::numpy_expr<Op, Arg0, Arg1>, pythonic::types::numpy_expr<Op, Arg2, Arg3>> {
//    typedef pythonic::types::numpy_expr<Op, typename __combine<Arg0, Arg2>, typename __combine<Arg1, Arg3>> type;
//};
//
//template<class Arg0, class Arg1, class Op>
//struct __combined<pythonic::types::numpy_uexpr<Op, Arg0>, pythonic::types::numpy_uexpr<Op, Arg1>> {
//    typedef pythonic::types::numpy_uexpr<Op, typename __combine<Arg0, Arg1>::type> type;
//};

/* } */

#include "pythonic/types/numpy_operators.hpp"

#ifdef ENABLE_PYTHON_MODULE

#include "pythonic/python/register_once.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

#include <boost/python/object.hpp>

namespace pythonic {

    template<class T>
        struct c_type_to_numpy_type {
            static const int value = c_type_to_numpy_type<decltype(std::declval<T>()())>::value;
        };

    template<>
        struct c_type_to_numpy_type<double> {
            static const int value = NPY_DOUBLE;
        };

    template<>
        struct c_type_to_numpy_type<float> {
            static const int value = NPY_FLOAT;
        };

    template<>
        struct c_type_to_numpy_type<std::complex<double>> {
            static const int value = NPY_CDOUBLE;
        };

    template<>
        struct c_type_to_numpy_type<long int> {
            static const int value = NPY_LONG;
        };

    template<>
        struct c_type_to_numpy_type<long unsigned int> {
            static const int value = NPY_ULONG;
        };

    template<>
        struct c_type_to_numpy_type<long long> {
            static const int value = NPY_LONGLONG;
        };

    template<>
        struct c_type_to_numpy_type<int> {
            static const int value = NPY_INT;
        };

    template<>
        struct c_type_to_numpy_type<unsigned int> {
            static const int value = NPY_UINT;
        };
    template<>
        struct c_type_to_numpy_type<short> {
            static const int value = NPY_SHORT;
        };

    template<>
        struct c_type_to_numpy_type<unsigned short> {
            static const int value = NPY_USHORT;
        };


    template<>
        struct c_type_to_numpy_type<signed char> {
            static const int value = NPY_INT8;
        };

    template<>
        struct c_type_to_numpy_type<unsigned char> {
            static const int value = NPY_UINT8;
        };

    template<>
        struct c_type_to_numpy_type<bool> {
            static const int value = NPY_BOOL;
        };

    template<class T>
        struct c_type_to_numpy_type< boost::simd::logical<T>> {
            static const int value = NPY_BOOL;
        };

    template<typename T, size_t N>
        struct python_to_pythran< types::ndarray<T, N> >{
            python_to_pythran(){
                static bool registered=false;
                python_to_pythran<T>();
                if(not registered) {
                    registered=true;
                    boost::python::converter::registry::push_back(&convertible,&construct,boost::python::type_id< types::ndarray<T, N> >());
                }
            }
            //reinterpret_cast needed to fit BOOST Python API. Check is done by template and PyArray_Check
            static void* convertible(PyObject* obj_ptr){
                if(!PyArray_Check(obj_ptr) or PyArray_TYPE(reinterpret_cast<PyArrayObject*>(obj_ptr)) != c_type_to_numpy_type<T>::value )
                    return 0;
                return obj_ptr;
            }

            static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data){
                void* storage=((boost::python::converter::rvalue_from_python_storage<types::ndarray<T,N>>*)(data))->storage.bytes;
                new (storage) types::ndarray< T, N>((T*)PyArray_BYTES(reinterpret_cast<PyArrayObject*>(obj_ptr)), PyArray_DIMS(reinterpret_cast<PyArrayObject*>(obj_ptr)), types::foreign());
                Py_INCREF(obj_ptr);
                data->convertible=storage;
            }
        };

    template <typename T>
        struct custom_boost_simd_logical {
            static PyObject* convert( boost::simd::logical<T> const& n) {
                return boost::python::incref(boost::python::object((T)n).ptr());
            }
        };
    template<typename T>
        struct pythran_to_python< boost::simd::logical<T> > {
            pythran_to_python() { register_once< boost::simd::logical<T>, custom_boost_simd_logical<T> >(); }
        };

    template<class T, size_t N>
        struct custom_array_to_ndarray {
            static PyObject* convert( types::ndarray<T,N> n) {
                n.mem.forget();
                PyObject* result = PyArray_SimpleNewFromData(N, n.shape.data(), c_type_to_numpy_type<T>::value, n.buffer);
                if (!result)
                    return nullptr;
                return result;
            }
        };

    template<class T, size_t N>
        struct pythran_to_python< types::ndarray<T,N> > {
            pythran_to_python() {
                register_once< types::ndarray<T,N> , custom_array_to_ndarray<T,N> >();
            }
        };
}

#endif


#endif

