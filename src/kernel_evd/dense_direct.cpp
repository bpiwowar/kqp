#include <kqp/kernel_evd/dense_direct.hpp>

namespace kqp {
#define KQP_SCALAR_GEN(scalar) template class DenseDirectBuilder<scalar>;
#include <kqp/for_all_scalar_gen.h.inc>
}