#include <kqp/feature_matrix.hpp>


# define KQP_SCALAR_GEN(Scalar)  \
    template class kqp::FeatureMatrixBase<Scalar>;\
    template class kqp::SpaceBase<Scalar>;
# include <kqp/for_all_scalar_gen.h.inc>


