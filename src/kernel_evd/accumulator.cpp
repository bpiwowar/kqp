#include <kqp/kernel_evd/accumulator.hpp>

#define KQP_SCALAR_GEN(type) template class kqp::AccumulatorKernelEVD<type,true>; template class kqp::AccumulatorKernelEVD<type,false>;
#include <kqp/for_all_scalar_gen.h.inc>
