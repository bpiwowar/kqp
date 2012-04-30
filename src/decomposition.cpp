#include <kqp/decomposition.hpp>

#define KQP_SCALAR_GEN(type) template struct kqp::Decomposition<type>;
#include <kqp/for_all_scalar_gen>
