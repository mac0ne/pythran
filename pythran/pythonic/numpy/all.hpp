#ifndef PYTHONIC_NUMPY_ALL_HPP
#define PYTHONIC_NUMPY_ALL_HPP

#include "pythonic/utils/proxy.hpp"
#include "pythonic/types/ndarray.hpp"
#include "pythonic/__builtin__/ValueError.hpp"
#include "pythonic/numpy/multiply.hpp"

namespace pythonic {

    namespace numpy {
        template<class E, class F>
            void _all(E begin, E end, F& all, utils::int_<1>)
            {
                for(; begin != end; ++begin)
                    if(not *begin) {
                        all = false;
                        return;
                    }
                        
            }
        template<class E, class F, size_t N>
            void _all(E begin, E end, F& all, utils::int_<N>)
            {
                for(; begin != end; ++begin)
                    if(all)
                        _all((*begin).begin(), (*begin).end(), all, utils::int_<N - 1>());
            }
            
        template<class E>
            typename types::numpy_expr_to_ndarray<E>::T
            all(E const& expr, types::none_type _ = types::none_type()) {
                typename types::numpy_expr_to_ndarray<E>::T p = true;
                _all(expr.begin(), expr.end(), p, utils::int_<types::numpy_expr_to_ndarray<E>::N>());
                return p;
            }

        template<class T>
            T all(types::ndarray<T,1> const& array, long axis)
            {
                if(axis != 0)
                    throw types::ValueError("axis out of bounds");
                return all(array);
            }

        template<class T, size_t N>
            typename types::ndarray<T,N>::value_type
            all(types::ndarray<T,N> const& array, long axis)
            {
                if(axis<0 || axis >=long(N))
                    throw types::ValueError("axis out of bounds");
                auto shape = array.shape;
                if(axis==0)
                {
                    return std::accumulate(array.begin() + 1, array.end(), *array.begin(), numpy::proxy::multiply());
                }
                else
                {
                    types::array<long, N-1> shp;
                    std::copy(shape.begin(), shape.end() - 1, shp.begin());
                    types::ndarray<T,N-1> ally(shp, __builtin__::None);
                    std::transform(array.begin(), array.end(), ally.begin(), [=](types::ndarray<T,N-1> const& other) {return all(other, axis-1);});
                    return ally;
                }
            }

        PROXY(pythonic::numpy, all);

    }

}

#endif

