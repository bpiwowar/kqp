#include <kqp/kernel_evd/dense_direct.hpp>

namespace kqp {
    // Instanciation of standard classes
    KQP_FOR_ALL_SCALAR_TYPES(template class DenseDirectBuilder<, >;);
}