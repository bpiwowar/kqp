#include <kqp/probabilities.hpp>

using namespace kqp;

#define KQP_FMATRIX_GEN_EXTERN(type) KQP_PROBABILITIES_FMATRIX_GEN(,type)
#include <kqp/for_all_fmatrix_gen>
