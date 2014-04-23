#ifndef PYTHONIC_NUMPY_MOD_HPP
#define PYTHONIC_NUMPY_MOD_HPP

#include "pythonic/utils/proxy.hpp"
#include "pythonic/types/assignable.hpp"
#include "pythonic/operator_/mod.hpp"


namespace pythonic {

    namespace operator_ {

        template<class A, class B>
            auto mod(A const& a, B const& b)
            -> typename std::enable_if<not std::is_fundamental<B>::value,
                                       typename assignable<decltype(a%b)>::type>::type
            {
                auto t = a % b;
                auto mask = t<0;
                t[mask] += b[mask];
                return t;
            }

        template<class A, class B>
            auto mod(A const& a, B const& b)
            -> typename std::enable_if<std::is_fundamental<B>::value and not std::is_fundamental<A>::value,
                                       typename assignable<decltype(a%b)>::type>::type
            {
                auto t = a % b;
                auto mask = t<0;
                t[mask] += b;
                return t;
            }

    }

    namespace numpy {
        using operator_::mod;

        PROXY(pythonic::numpy, mod);
    }

}

#endif

