#ifndef PYTHONIC_NUMPY_COSH_HPP
#define PYTHONIC_NUMPY_COSH_HPP

#include "pythonic/utils/proxy.hpp"
#include "pythonic/types/ndarray.hpp"
#include <nt2/include/functions/cosh.hpp>

namespace pythonic {

    namespace numpy {
#define NUMPY_UNARY_FUNC_NAME cosh
#define NUMPY_UNARY_FUNC_SYM nt2::cosh
#include "pythonic/types/numpy_unary_expr.hpp"

    }

}

#endif

