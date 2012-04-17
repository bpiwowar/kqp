#include <kqp/decomposition.hpp>

#define KQP_FMATRIX_GEN(type) template struct kqp::Decomposition<type>;
#include <kqp/for_all_fmatrix_gen>
