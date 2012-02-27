#include <kqp/kqp.hpp>

#include <kqp/reduced_set/qp_approach.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include "tests_utils.hpp"
#include <kqp/kernel_evd/accumulator.hpp>

DEFINE_LOGGER(logger, "kqp.test.reduced-set.qp")

namespace kqp {
    
    int test_reduced_set_qp(std::deque<std::string> &/*args*/) {
        // Typedefs
        typedef DenseMatrix<double> FMatrix;
        typedef ftraits<FMatrix>::Scalar Scalar;
        typedef ftraits<FMatrix>::ScalarMatrix ScalarMatrix;
        typedef ftraits<FMatrix>::RealVector RealVector;
        
       
        // Parameters
        Index dim = 10;
        Index r_target = 9; // trivial test (just have to remove one)
        Index n = 3;
        
        // Gets a rank-n matrix and a full rank matrix
        Eigen::MatrixXd _mF = generateMatrix<Scalar>(dim, dim);        
//        _mF.col(0) = _mF.col(dim-1) + _mF.col(dim-2);
        Eigen::MatrixXd _mY = generateMatrix<Scalar>(dim, dim).leftCols(n);                        
        
        // Diagonal matrix
        Eigen::VectorXd _mD(n);
        for(Index i=0; i < n; i++)
            _mD[i] = Eigen::internal::random(1, 10);

        // Computes the EVD
        AccumulatorKernelEVD<FMatrix, false> kEVD;
        ScalarMatrix m;
        noalias(m) = _mY * _mD.cwiseSqrt().asDiagonal();
        kEVD.add(1, FMatrix(_mF), m);

        FMatrix mF;
        AltDense<Scalar>::type mY;
        RealVector mD;
        kEVD.get_decomposition(mF, mY, mD, false);

        // Reduced set computation
        ReducedSetWithQP<FMatrix> qp_rs;
        qp_rs.run(r_target, mF, mY, mD);
        
        // Compare
        
        Eigen::MatrixXd m1 = qp_rs.getFeatureMatrix().get_matrix() * qp_rs.getMixtureMatrix() * qp_rs.getEigenValues().asDiagonal()
            * qp_rs.getMixtureMatrix().transpose() * qp_rs.getFeatureMatrix().get_matrix().transpose();
        Eigen::MatrixXd m2 = _mF * _mY * _mD.asDiagonal() * _mY.transpose() * _mF.transpose();
        double error = (m1 - m2).norm() / m2.norm();
        
        Index delta = (_mF.cols() - qp_rs.getFeatureMatrix().size());
        double threshold = (1e-10 * delta);
        
        KQP_LOG_INFO_F(logger, "Error is %g [threshold=%g] and pre-images difference is %d", %error %threshold %delta);
        
        return error < threshold ? 0 : 1;
    }
}