#include <kqp/feature_matrix/sparse.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>

namespace kqp {
#define KQP_SCALAR_GEN(scalar) \
    template class Sparse<scalar>; \
    template class SparseDense<scalar>; \
    template class SparseSpace<scalar>; \
    template class SparseDenseSpace<scalar>;
#include <kqp/for_all_scalar_gen.h.inc>
}