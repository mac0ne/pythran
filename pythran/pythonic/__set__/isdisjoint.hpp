#ifndef PYTHONIC_SET_ISDISJOINT_HPP
#define PYTHONIC_SET_ISDISJOINT_HPP

#include "pythonic/utils/proxy.hpp"
#include "pythonic/types/set.hpp"

namespace pythonic {

    namespace __set__ {
        template<class T, class U>
            bool isdisjoint(types::set<T> const& calling_set, U const& arg_set) {
                return calling_set.isdisjoint(arg_set);
            }
        template<class U>
            bool isdisjoint(types::empty_set const& calling_set, U const& arg_set) {
                return true;
            }
        PROXY(pythonic::__set__, isdisjoint);

    }

}

#endif

