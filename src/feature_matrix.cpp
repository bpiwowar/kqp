#include <kqp/feature_matrix.hpp>


# define KQP_SCALAR_GEN(Scalar)  \
    template class kqp::FeatureMatrix<Scalar>;\
    template class kqp::Space<Scalar>;
# include <kqp/for_all_scalar_gen>


