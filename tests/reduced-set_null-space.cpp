#include <kqp/kqp.hpp>

#include <kqp/reduced_set/null_space.hpp>
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
        
        FeatureSpace<double> fs(DenseFeatureSpace<double>::create(2));
        
        // Copy
        Eigen::MatrixXd mY = _mY;
        FeatureMatrix<double> mF(DenseMatrix<double>::create(_mF));
        
        ReducedSetNullSpace<double>::run(fs, mF, mY);
        
        Eigen::MatrixXd m1 = mF->as<DenseMatrix<double>>().getMatrix() * mY;
        Eigen::MatrixXd m2 = _mF * _mY;
        double error = (m1 - m2).norm();
        
        Index delta = (_mY.rows() - mY.rows());
        double threshold = (1e-10 * delta);
        KQP_LOG_INFO_F(logger, "Error is %g [threshold=%g] and row difference is %d", %error %threshold %delta);

        return mF.size() == n && (error < threshold) ? 0 : 1;
    }
}
