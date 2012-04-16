
#include <kqp/feature_matrix/dense.hpp>
#include <kqp/kernel_evd/incremental.hpp>

#define KQP_FMATRIX_GEN(type) template class kqp::IncrementalKernelEVD<type>;
#include <kqp/for_all_fmatrix_gen>
