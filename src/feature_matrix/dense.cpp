#include <kqp/feature_matrix/dense.hpp>

namespace kqp {
#define KQP_SCALAR_GEN(scalar) template class DenseMatrix<scalar>; template class DenseFeatureSpace<scalar>;
#include <kqp/for_all_scalar_gen>
}
