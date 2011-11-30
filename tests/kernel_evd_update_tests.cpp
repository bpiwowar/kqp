#include "kqp.h"
#include "kernel_evd.h"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd")

namespace kqp {
    

    using namespace Eigen;
    
    void direct_evd() {
        Index n = 10;
        ScalarMatrix<double> matrix(n);
        
        VectorXd v = VectorXd::Random(n);
        AccumulatorBuilder<ScalarMatrix<double> > builder;
        matrix.add(v);
        
    }
    
    int kevd_tests(int argc, const char **argv) {
        direct_evd();
        return 0;
    }
    
    
}