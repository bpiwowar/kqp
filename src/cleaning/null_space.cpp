#include <kqp/cleaning/null_space.hpp>

# define KQP_SCALAR_GEN(Scalar) template class kqp::ReducedSetNullSpace<Scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
