#include <kqp/feature_matrix/sparse.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>

namespace kqp {
#define KQP_SCALAR_GEN(scalar) \
    template class SparseMatrix<scalar>; \
    template class SparseDenseMatrix<scalar>; \
    template class SparseFeatureSpace<scalar>; \
    template class SparseDenseFeatureSpace<scalar>;
#include <kqp/for_all_scalar_gen>
}