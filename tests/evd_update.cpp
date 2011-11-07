#include <iostream>

#include <boost/format.hpp>
#include "kqp.h"
#include "test.h"
#include "evd_update.h"

namespace kqp {

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
        z << 0, 0, 0;
        
        EvdUpdateResult result;
        updater.rankOneUpdate(boost::shared_ptr<Eigen::MatrixXd>(), D, 1, z, true, 0, true, result);
        
        
        Eigen::MatrixXd zzt = z*z.transpose();
        Eigen::MatrixXd mD = D.asDiagonal();
        Eigen::MatrixXd delta = result.mQ * result.mD * result.mQ.transpose() - (mD + z * z.transpose());
        
        if (delta.norm() < tolerance) return 0;
        
        std::cerr << "D + z * z' = " << std::endl << mD + zzt  << std::endl;
        std::cerr << "Q * S * Q.T = " << std::endl << result.mQ * result.mD * result.mQ.transpose() << std::endl;
        std::cerr << "Delta" << std::endl << delta << std::endl;
        return 1;
    }
    
    BOOST_THROW_EXCEPTION(illegal_argument_exception()
                          << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
}


}