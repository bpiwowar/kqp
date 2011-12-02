#include "kqp.hpp"

#include "kernel_evd.hpp"
#include "OperatorBuilder/dense_direct_builder.hpp"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd")

namespace kqp {
    

    using namespace Eigen;
    
    template<class T> void direct_evd(int n, int k, T &builder) {
        typedef typename T::Scalar Scalar;
        typedef Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    
        KQP_LOG_INFO(logger, "EVD with dense vectors and builder=" << KQP_DEMANGLE(builder));
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix(n,n);
        
        for(int i = 0; i < k; i++) {
            Vector v = Vector::Random(n);
            Scalar alpha = Eigen::internal::random_impl<Scalar>::run();

            matrix.template selfadjointView<Eigen::Lower>().rankUpdate(v, alpha);
            builder.add(alpha, v);
            
        }
                
        
    }
    
    int kevd_tests(int argc, const char **argv) {
        
        Index n = 10;
        DenseDirectBuilder<double> builder(n);
        direct_evd(n, 5, builder);
        
        return 0;
    }
    
    
}