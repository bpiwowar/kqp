
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
        
        int kevd_tests::Accumulator::run(const Dense_evd_test &test) const {
            if (this->use_lc) 
                return _accumulator<true>(test);
            return _accumulator<false>(test);
        }
    }
}