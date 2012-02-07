#include "kqp.hpp"
#include "test.hpp"

#include "null_space.hpp"
#include "feature_matrix/dense.hpp"

DEFINE_LOGGER(logger, "kqp.test.null-space")

namespace kqp {
    int do_null_space_tests(std::deque<std::string> &args) {
        if (args.size() != 1) 
            KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Expected the task name, got %d arguments", %args.size());
                            
        std::string name = args[0];
        
        // (1) Simplest test: remove explicitely unused pre-images
        if (name == "unused") {
            Eigen::MatrixXd _mF = Eigen::MatrixXd::Random(10,10);
            DenseMatrix<double> mF(_mF);
            
            Eigen::MatrixXd mY = Eigen::MatrixXd::Random(10,8);
            mY.row(3).setZero();
            mY.row(7).setZero();
            Eigen::MatrixXd _mY = mY;
            
            removeUnusedPreImages(mF, mY);
            std::cerr << mY << std::endl << std::endl;            
            std::cerr << _mY << std::endl << std::endl;            
            
            Eigen::MatrixXd m1 = mF.get_matrix() * mY;
            Eigen::MatrixXd m2 = _mF * _mY;
            double error = (m1 - m2).norm();
            
            Index delta = (_mY.rows() - mY.rows());
            KQP_LOG_INFO_F(logger, "Error is %g and row difference is %d", %error %delta);
            
            return (delta > 0) && (error < EPSILON * delta);
        }
        
        // (2) Remove unused pre-images using the null space method
        if (name == "null-space") {
            
//            double error = ...;
//            removeUnusefulPreImages(mF, mY,);
//            return error < EPSILON;
        }
        
        return 0;
    }
}