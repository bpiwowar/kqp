#include "dense.hpp"

namespace kqp {
    KQP_FOR_ALL_SCALAR_TYPES(template class DenseMatrix<, >;);
}
