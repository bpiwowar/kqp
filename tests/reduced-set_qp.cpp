#include <algorithm>

#include <kqp/kqp.hpp>

#include <kqp/cleaning/qp_approach.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include "tests_utils.hpp"
#include <kqp/kernel_evd/accumulator.hpp>

DEFINE_LOGGER(logger, "kqp.test.reduced-set.qp")

namespace kqp {
    
    int test_reduced_set_qp_exact(std::deque<std::string> &/*args*/) {        
        // Typedefs
        typedef Dense<double> DMatrix;
        typedef double Scalar;
        KQP_SCALAR_TYPEDEFS(double);
        
       
        // Parameters
        Index dim = 10;
        Index r_target = 9; // trivial test (just have to remove one)
        Index n = 3;
        
        Space<Scalar> fs(DenseSpace<Scalar>::create(dim));
        
        // Gets a rank-n matrix and a full rank matrix
        Eigen::MatrixXd _mF = generateMatrix<Scalar>(dim, dim);        
//        _mF.col(0) = _mF.col(dim-1) + _mF.col(dim-2);
        Eigen::MatrixXd _mY = generateMatrix<Scalar>(dim, dim).leftCols(n);                        
        
        // Diagonal matrix
        Eigen::VectorXd _mD(n);
        for(Index i=0; i < n; i++)
            _mD[i] = Eigen::internal::random(1, 10);
        
        // Ensures we find a solution
        _mY.row(2).setZero();

        // Computes the EVD
        AccumulatorKernelEVD<double, false> kEVD(fs);
        ScalarMatrix m;
        noalias(m) = _mY * _mD.cwiseSqrt().asDiagonal();
        kEVD.add(1, Dense<double>::create(_mF), m);

        Decomposition<double> d = kEVD.getDecomposition();

        // Reduced set computation
        ReducedSetWithQP<double> qp_rs;
        qp_rs.run(r_target, fs, d.mX, d.mY, d.mD);
        
        // Compare
        
        const ScalarMatrix &fm = dynamic_cast<const Dense<double>&>(*qp_rs.getFeatureMatrix()).getMatrix();
        Eigen::MatrixXd m1 = fm * qp_rs.getMixtureMatrix() * qp_rs.getEigenValues().asDiagonal()
            * qp_rs.getMixtureMatrix().transpose() * fm.transpose();
        Eigen::MatrixXd m2 = _mF * _mY * _mD.asDiagonal() * _mY.transpose() * _mF.transpose();
        double error = (m1 - m2).norm() / m2.norm();
        
        Index delta = (_mF.cols() - qp_rs.getFeatureMatrix()->size());
        double threshold = (1e-10 * delta);
        
        KQP_LOG_INFO_F(logger, "Error is %g [threshold=%g] and pre-images difference is %d", %error %threshold %delta);
        
        return error < threshold ? 0 : 1;
    }
    
    namespace {
        Eigen::PermutationMatrix<Dynamic, Dynamic, Index> getRandomPermutation(const Index n) {
            // build up
            std::vector<Index> permutation;
            for(Index i = 0; i < n; i++) permutation.push_back(i);
            std::random_shuffle(permutation.begin(), permutation.end());
            
            // copy to Eigen 
            Eigen::Matrix<Index,Dynamic,1> v(n);
            for(Index i = 0; i < n; i++) v[i] = permutation[i];
            return Eigen::PermutationMatrix<Dynamic, Dynamic, Index>(v);
            
        }
    }
    
        
    int test_reduced_set_qp_approximate(std::deque<std::string> &/*args*/) {        
        // Typedefs
        typedef Dense<double> DMatrix;
        typedef double Scalar;
       KQP_SCALAR_TYPEDEFS(double);
    
        
        // Parameters
        Index dim = 10; // dimension for pre-images
        Index nbPreImages = 8;
        Index p = 4; // Number of vectors for the basis
        Index to_remove = 2;
        Index r_target = nbPreImages - to_remove; // One pre-image to remove
        double alpha = 1e-2;
        
        Space<Scalar> fs(DenseSpace<Scalar>::create(dim));

        // --- Build the operator
        
        // Construct U = (U0 0; 0 0) where U0 is orthonormal
        ScalarMatrix mU(dim,r_target);
        mU.topRows(dim-to_remove) = generateOrthonormalMatrix<Scalar>(dim-to_remove, r_target);
        mU.bottomRows(to_remove).setZero();
        ScalarMatrix pX0 = mU * mU.transpose(); // projector on X0 subspace

        // Construct X = (X0 x0; 0 alpha)
        ScalarMatrix mX(dim,nbPreImages);
        mX.leftCols(r_target) =  mU * generateMatrix<Scalar>(r_target, r_target);
        
        // Last columns are linear combinations of previous vectors plus a small value
        mX.rightCols(to_remove) = mX.leftCols(r_target) * ScalarMatrix::Random(r_target, to_remove);
        mX.bottomRightCorner(to_remove, to_remove) = generateVector<Scalar>(to_remove, alpha, 2.*alpha).asDiagonal();
                
        // Random permutation of columns
        mX = mX * getRandomPermutation(mX.cols());
//        std::cerr << "X\n" << mX << std::endl;
        
        // Linear combination matrix
        ScalarMatrix mY(generateMatrix<Scalar>(nbPreImages, nbPreImages).leftCols(p));
        mY.bottomRows(to_remove).array() *= 1e-2;
        
        // Diagonal
        RealVector s = generateVector<double>(p,0.1,1.5);
        
        

        
        ScalarMatrix opX = mX * mY * s.asDiagonal() * mY.transpose() * mX.transpose();

        // --- Reduced set
        
        // Computes the k-EVD, keeping the pre-images
        AccumulatorKernelEVD<Scalar, false> kEVD(fs);
        ScalarMatrix m;
        noalias(m) = mY * s.cwiseSqrt().asDiagonal();
        kEVD.add(1, Dense<double>::create(mX), m);
        
        Decomposition<double> d = kEVD.getDecomposition();
        
        // Reduced set computation
        ReducedSetWithQP<double> qp_rs;
        qp_rs.run(r_target, fs, d.mX, d.mY, d.mD);
        
        
        // Compare
        
        const ScalarMatrix &mX_r = dynamic_cast<const Dense<Scalar>&>(*qp_rs.getFeatureMatrix()).getMatrix();
//        std::cerr << "mX_r\n" << mX_r << std::endl;
        Eigen::MatrixXd m1 = mX_r * qp_rs.getMixtureMatrix() * qp_rs.getEigenValues().asDiagonal() * qp_rs.getMixtureMatrix().transpose() * mX_r.transpose();
        double diff = (m1 - pX0 * opX * pX0).norm();
                
        double error_qp = (m1 - opX).norm();;
        double error_heuristic = (pX0 * opX * pX0 - opX).norm();;

        KQP_LOG_INFO_F(logger, "Error is %g (threshold: %g) - difference between both approaches is %g", %error_qp %error_heuristic %diff);
        
        return error_qp < error_heuristic ? 0 : 1;
    }

}