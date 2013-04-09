#include <kqp/cleaning/unused.hpp>

# ifndef SWIG
# define KQP_SCALAR_GEN(Scalar) template class kqp::CleanerUnused<Scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
# endif 