#include "kqp.hpp"

#include "kernel_evd.hpp"
#include "OperatorBuilder/dense_direct_builder.hpp"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd")

namespace kqp {
    

    using namespace Eigen;
    
    void direct_evd() {
        // Dimension
        Index n = 10;
        
        // Number of vectors
        int k = 5;
        
        ScalarMatrix<double> matrix(n);
        
        for(int i = 0; i < k; i++) {
            VectorXd v = VectorXd::Random(n);
            
            DenseDirectBuilder<double> builder(n);
            
            matrix.add(v);
        }
        
    }
    
    int kevd_tests(int argc, const char **argv) {
        direct_evd();
        return 0;
    }
    
    
}