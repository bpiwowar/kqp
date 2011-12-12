#include <iostream>
#include <complex>

#include <boost/format.hpp>

#include <Eigen/Eigenvalues>

#include "kqp.hpp"
#include "test.hpp"
#include "evd_update.hpp"

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
    template<typename scalar> int evd_update_random(const double * rhos, long seed, int dim, int nzeroD, int nzeroZ) {
        typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> Vector;
        
        std::srand(seed);

        Vector D(dim);
        for(int i = 0; i < dim; i++)
            D(i) = std::rand();
        
        int i = 0;
        while (rhos[i++] != 0) {
            FastRankOneUpdate<scalar> updater;

            double rho = rhos[i];
            Vector z(dim);    
            for(int i = 0; i < dim; i++)
                z(i) = rand();
            
            EvdUpdateResult<scalar> result;
            updater.update(D, rho, z, true, 0, true, result);
            
            
            Matrix zzt = z*z.adjoint();
            Matrix mD = D.asDiagonal();
            Matrix delta = result.mQ * result.mD * result.mQ.adjoint() - (mD + rho * z * z.adjoint());
            
            double error = delta.norm();
            
            KQP_LOG_INFO(logger, "Error is ");
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
    
       
int evd_update_test(std::vector<std::string> &args) {
    if (args.size() != 1) 
        BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("evd_update_test needs one argument"));
    
    const std::string &name = args[0];
    
    double rhos[] = { 1. };
    
    if (name == "simple") {
        return evd_update_random<double>(rhos, 0, 10, 0, 0);
    } else if (name == "complex") {
        return evd_update_random<std::complex<double> >(rhos, 0, 10, 0, 0);
    }
        
    
    BOOST_THROW_EXCEPTION(illegal_argument_exception()
                          << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
}


}