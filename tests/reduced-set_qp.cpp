#include "kqp.hpp"

#include "reduced_set/qp_approach.hpp"
#include "feature_matrix/dense.hpp"
#include "tests_utils.hpp"

DEFINE_LOGGER(logger, "kqp.test.reduced-set.qp")

namespace kqp {
    
    int test_reduced_set_qp(std::deque<std::string> &args) {
        // Typedefs
        typedef DenseMatrix<double> FMatrix;
        typedef ftraits<FMatrix>::Scalar Scalar;
        
        // Parameters
        Index dim = 10;
        Index n = 3;
        
        // Gets a rank-n matrix and a full rank matrix
        Eigen::MatrixXd _mF = generateMatrix<Scalar>(dim, n);        
        Eigen::MatrixXd _mY = generateMatrix<Scalar>(dim, dim);                        
        
        // Copy
        kqp::AltMatrix<double> mY(_mY);
        FMatrix mF(_mF);
        
        // Diagonal matrix
        Eigen::VectorXd mD(mY.rows());
        for(Index i=0; i < mD.size(); i++)
            mD[i] = Eigen::internal::random(1e-1, 1e3);
        
        // Exact
        ReducedSetWithQP<FMatrix> qp_rs;
        qp_rs.run(dim-n, mF, mY, mD);
        
        Eigen::MatrixXd m1 = mF.get_matrix() * mY;
        Eigen::MatrixXd m2 = _mF * _mY;
        double error = (m1 - m2).norm();
        
        Index delta = (_mY.rows() - mY.rows());
        double threshold = (1e-10 * delta);
        KQP_LOG_INFO_F(logger, "Error is %g [threshold=%g] and row difference is %d", %error %threshold %delta);
        
        return error < threshold;
    }
}