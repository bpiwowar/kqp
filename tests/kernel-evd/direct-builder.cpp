#include "kernel-evd-tests.hpp"

#include "kernel_evd/dense_direct.hpp"
DEFINE_LOGGER(logger, "kqp.test.kernel_evd.direct-builder")

namespace kqp {
    namespace kevd_tests {        
        int direct_builder(const Dense_evd_test &test) {
            DenseDirectBuilder<double> builder(test.n);
            return test.run(logger, builder);
        }
    }
}