
#include <cassert>

#include "kqp.hpp"

#include "kernel-evd-tests.hpp"

#include "kernel_evd/accumulator.hpp"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd.accumulator")
namespace kqp {
    namespace kevd_tests {        
        
        template<bool use_lc>
        int _accumulator(const Dense_evd_test &test) {
            AccumulatorKernelEVD<DenseMatrix<double>, use_lc > builder;
            if (AccumulatorKernelEVD<DenseMatrix<double>, use_lc >::use_linear_combination != use_lc)
                abort();
            return test.run(logger, builder);
        }
        
        int accumulator(const Dense_evd_test &test, bool use_lc) {
            if (use_lc) 
                return _accumulator<true>(test);
            return _accumulator<false>(test);
        }
    }
}