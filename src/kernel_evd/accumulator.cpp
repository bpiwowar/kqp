#include <kqp/kernel_evd/accumulator.hpp>

#define KQP_FMATRIX_GEN(type) template class kqp::AccumulatorKernelEVD<type>;
#include <kqp/for_all_fmatrix_gen>
