#include "kernel-evd-tests.hpp"
#include "kernel_evd/incremental.hpp"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd.incremental")

namespace kqp {
    namespace kevd_tests {        
        int Incremental::run(const Dense_evd_test &test) const {
            IncrementalKernelEVD<DenseMatrix<double> > builder;
            return test.run(logger, builder);
        }
    }
}