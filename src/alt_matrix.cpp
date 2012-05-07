#include <kqp/alt_matrix.hpp>

namespace kqp {
# define KQP_SCALAR_GEN(type) \
  template class AltMatrix<AltDense<type>::DenseType, AltDense<type>::IdentityType>; \
  template class AltMatrix<AltVector<type>::VectorType, AltVector<type>::ConstantVectorType>;
#include <kqp/for_all_scalar_gen.h.inc>
}

