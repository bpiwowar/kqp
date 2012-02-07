#include "kqp.hpp"

#include "reduced_set/ldl_approach.hpp"
#include "feature_matrix/dense.hpp"

DEFINE_LOGGER(logger, "kqp.test.reduced-set.ldl")

namespace kqp {
    
    int test_reduced_set_ldl(std::deque<std::string> &args) {
        
        // Creates a rank-n matrix
        Index dim = 10;
        Index n = 7;
        
        Eigen::MatrixXd _mF = Eigen::MatrixXd::Random(dim,dim);
        for(Index i = 0; i < n; i++) {
            Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
            _mF.selfadjointView<Eigen::Lower>().rankUpdate(x, 1);
        }
        
        Eigen::MatrixXd _mY = Eigen::MatrixXd::Random(dim,dim);
        
        
        // Copy
        Eigen::MatrixXd mY = _mY;
        DenseMatrix<double> mF(_mF);
        
        kqp::removePreImagesWithLDL(mF, mY);
        
        Eigen::MatrixXd m1 = mF.get_matrix() * mY;
        Eigen::MatrixXd m2 = _mF * _mY;
        double error = (m1 - m2).norm();
        
        Index delta = (_mY.rows() - mY.rows());
        KQP_LOG_INFO_F(logger, "Error is %g and row difference is %d", %error %delta);
        return 0;
    }
}