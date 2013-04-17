#include <iostream>
#include <complex>

#include <boost/format.hpp>


#include <kqp/kqp.hpp>
#include <kqp/logging.hpp>
#include <kqp/evd_update.hpp>
#include <Eigen/Eigenvalues>

DEFINE_LOGGER(logger, "kqp.test.evd-update")

namespace kqp {

    using std::cerr;
    using std::endl;
    
    namespace { double tolerance = 1e-10; }
    
    /**
     * Performs an EVD rank one update and compares to the direct computation of the rank one update
     *
     * @param seed The random seed
     * @param N dimension of the problem
     * @param rho Coefficient
     * @param nzeroD Number of zero entries in diagonal matrix
     * @param nzeroZ Number of zero entries in vector
     */
    template<typename scalar> int evd_update_random(const double * rhos, long seed, int dim, int /*nzeroD*/, int /*nzeroZ*/, bool use_update = false) {
        typedef Eigen::Matrix<scalar,Dynamic,Dynamic> Matrix;
        typedef typename Eigen::NumTraits<scalar>::Real Real;
        typedef Eigen::Matrix<Real,Dynamic,1> RealVector;
        typedef Eigen::Matrix<scalar,Dynamic,1> ScalarVector;
        
        std::srand(seed);

        RealVector D(dim);
        for(int i = 0; i < dim; i++)
            D(i) = std::rand();
        
        int i = 0;
        Matrix mQ;
                
        while (rhos[i++] != 0) {
            FastRankOneUpdate<scalar> updater;

            double rho = rhos[i];
            ScalarVector z(dim);    
            for(int i = 0; i < dim; i++)
                z(i) = rand();
            
            EvdUpdateResult<scalar> result;
            updater.update(D, rho, z, true, 0, true, result, use_update ? &mQ : 0);
            
            
            Matrix zzt = z*z.adjoint();
            Matrix delta = (use_update ? mQ : result.mQ) * result.mD.asDiagonal() * (use_update ? mQ : result.mQ).adjoint() - (Matrix(D.template cast<scalar>().asDiagonal()) + rho * z * z.adjoint());
            
            double error = delta.norm();
            
            KQP_LOG_INFO(logger, boost::format("Error is %g") % error);
            if (error < tolerance) return 0;

            return 1;
//            cerr << "D + z * z' = " << std::endl << mD + zzt  << std::endl;
//            cerr << "Q = " << std::endl << result.mQ << std::endl;
//            cerr << "S = " << std::endl << result.mD.diagonal() << std::endl;
//            cerr << "Q * S * Q.T = " << std::endl << result.mQ * result.mD * result.mQ.adjoint() << std::endl;
//            cerr << "Delta" << std::endl << delta << std::endl;
        }
        
        return 0;
    }
    
       
int evd_update_test(std::deque<std::string> &args) {
    if (args.size() != 1) 
        BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("evd_update_test needs one argument"));
    
    const std::string &name = args[0];
    
    double rhos[] = { 1. };
    
    if (name == "simple") {
        return evd_update_random<double>(rhos, 0, 10, 0, 0);
        return evd_update_random<double>(rhos, 0, 10, 0, 0, true);
    } else if (name == "complex") {
        return evd_update_random<std::complex<double> >(rhos, 0, 10, 0, 0);
        return evd_update_random<double>(rhos, 0, 10, 0, 0, true);
    }
        
    
    BOOST_THROW_EXCEPTION(illegal_argument_exception()
                          << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
}


}

int main(int argc, const char **argv) {
    std::deque<std::string> args;
    for(int i = 1; i < argc; i++) 
        args.push_back(argv[i]);
    kqp::evd_update_test(args);
}
