#include <kqp/feature_matrix/sparse.hpp>

namespace kqp {
#define KQP_SCALAR_GEN(scalar) \
    template class SparseMatrix<scalar>; \
    template class SparseDenseMatrix<scalar>;
#include <kqp/for_all_scalar_gen>
}