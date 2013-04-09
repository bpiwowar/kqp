#include <kqp/kqp.hpp>

#include <kqp/cleaning/null_space.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include "tests_utils.hpp"

DEFINE_LOGGER(logger, "kqp.test.reduced-set.null-space")

namespace kqp {
    
    int test_reduced_set_null_space(std::deque<std::string> &/*args*/) {
        
        // --- Random test

        // Parameters
        Index dim = 10;
        Index n = 3;
        
        // Gets a rank-n matrix and a full rank matrix
        Eigen::MatrixXd _mF = generateMatrix<double>(dim, n);        
        Eigen::MatrixXd _mY = generateMatrix<double>(dim, dim);                        
        
        DenseSpace<double>::SelfPtr fs(new DenseSpace<double>(2));
        
        // Copy
        Eigen::MatrixXd mY = _mY;
        Dense<double>::SelfPtr mF(new Dense<double>(_mF));
        
        ReducedSetNullSpace<double> nullSpace;
        auto result = nullSpace.run(fs, mF, mY);
        mF = boost::dynamic_pointer_cast<Dense<double>>(result.mX);
        mY = result.mY;
        
        
        Eigen::MatrixXd m1 = mF->as<Dense<double>>().getMatrix() * mY;
        Eigen::MatrixXd m2 = _mF * _mY;
        double error = (m1 - m2).norm();
        
        Index delta = (_mY.rows() - mY.rows());
        double threshold = (1e-10 * delta);
        KQP_LOG_INFO_F(logger, "Error is %g [threshold=%g] and row difference is %d", %error %threshold %delta);

        return mF->size() == n && (error < threshold) ? 0 : 1;
    }
}
