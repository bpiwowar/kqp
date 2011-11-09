#include <iostream>
#include <complex>

#include <boost/format.hpp>

#include <Eigen/Eigenvalues>

#include "kqp.h"
#include "test.h"
#include "evd_update.h"

namespace kqp {

    using std::cerr;
    using std::endl;
    
    double tolerance = 1e-10;
    
int evd_update_test(int argc, const char **argv) {
    if (argc != 1) 
        BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("evd_update_test needs one argument"));
    
    std::string name = argv[0];
    
    if (name == "simple") {
        FastRankOneUpdate updater;
        
        Eigen::VectorXd D(3);
        D << 0.2, 0.3, 1.2;
        
        Eigen::VectorXd z(3); 
        z << 0.1, -0.1, 0.2;
        
        EvdUpdateResult<double> result;
        updater.update(boost::shared_ptr<Eigen::MatrixXd>(), D, 1, z, true, 0, true, result);
        
        
        Eigen::MatrixXd zzt = z*z.transpose();
        Eigen::MatrixXd mD = D.asDiagonal();
        Eigen::MatrixXd delta = result.mQ * result.mD * result.mQ.transpose() - (mD + z * z.transpose());
        
        if (delta.norm() < tolerance) return 0;
        
        std::cerr << "D + z * z' = " << std::endl << mD + zzt  << std::endl;
        std::cerr << "Q = " << std::endl << result.mQ << std::endl;
        std::cerr << "S = " << std::endl << result.mD.diagonal() << std::endl;
        std::cerr << "Q * S * Q.T = " << std::endl << result.mQ * result.mD * result.mQ.transpose() << std::endl;
        std::cerr << "Delta" << std::endl << delta << std::endl;
        return 1;
    } else if (name == "complex") {
        FastRankOneUpdate updater;
        typedef std::complex<double> dcomplex;
        
        Eigen::VectorXcd D(3);
        D << dcomplex(0.2), dcomplex(0.3), dcomplex(1);
        
        Eigen::VectorXcd z(3); 
        z << dcomplex(0.2,1), dcomplex(0,-0.4), dcomplex(0.2,1);
        
        EvdUpdateResult<dcomplex> result;
        updater.update<dcomplex>(boost::shared_ptr<Eigen::MatrixXcd>(), D, 1, z, true, 0, true, result);
        
        
        Eigen::MatrixXcd zzt = z*z.adjoint();
        Eigen::MatrixXcd mD = D.asDiagonal();
        Eigen::MatrixXcd updated_D = mD + z * z.adjoint();
        Eigen::MatrixXcd delta = result.mQ * result.mD * result.mQ.adjoint() - (updated_D);
        
        if (delta.norm() < tolerance) return 0;
        
        std::cerr << "D + z * z' = " << std::endl << mD + zzt  << std::endl;
        std::cerr << "Q = " << std::endl << result.mQ << std::endl;
        std::cerr << "S = " << std::endl << result.mD.diagonal() << std::endl;
        std::cerr << "Q * S * Q.T = " << std::endl << result.mQ * result.mD * result.mQ.adjoint() << std::endl;
        std::cerr << "Delta" << std::endl << delta << std::endl;
        
        // Does an EVD
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ces;
        ces.compute(updated_D);
        cerr << "The eigenvalues of A are:" << endl << ces.eigenvalues() << endl;
        cerr << "The matrix of eigenvectors, V, is:" << endl << ces.eigenvectors() << endl << endl;

        return 1;
    }
    
    BOOST_THROW_EXCEPTION(illegal_argument_exception()
                          << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
}


}