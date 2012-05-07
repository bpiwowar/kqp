#include <kqp/reduced_set/null_space.hpp>

# define KQP_SCALAR_GEN(Scalar) template struct kqp::ReducedSetNullSpace<Scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
