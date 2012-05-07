#include <kqp/probabilities.hpp>

using namespace kqp;

#define KQP_SCALAR_GEN(type) KQP_PROBABILITIES_FMATRIX_GEN(,type)
#include <kqp/for_all_scalar_gen.h.inc>
