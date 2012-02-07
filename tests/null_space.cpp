#include "kqp.hpp"
#include "test.hpp"

#include "null_space.hpp"
#include "feature_matrix/dense.hpp"

DEFINE_LOGGER(logger, "kqp.test.null-space")

namespace kqp {
    
    
    
    /**
     * @brief Removes pre-images with the null space method
     * 
     * Removes pre-images using the null space method
     * 
     * @param mF the feature matrix
     * @param nullSpace the null space vectors of the gram matrix of the feature matrix
     */
    template <class FMatrix, typename Derived>
    void removeNullSpacePreImages(FeatureMatrix<FMatrix> &mF, const Eigen::MatrixBase<Derived> &nullSpace) {
        
    }
    
    /**
     * @brief Removes unuseful pre-images 
     *
     * 1. Removes unused pre-images 
     * 2. Computes a \f$LDL^\dagger\f$ decomposition of the Gram matrix to find redundant pre-images
     * 3. Removes newly unused pre-images
     */
    template <class FMatrix>
    void removePreImagesWithLDL(FMatrix &mF, typename ftraits<FMatrix>::ScalarMatrix &mY) {
        typedef typename ftraits<FMatrix>::Scalar Scalar;
        typedef typename ftraits<FMatrix>::ScalarMatrix ScalarMatrix;
        
        // Removes unused pre-images
        removeUnusedPreImages(mF, mY);
        
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        // LDL decomposition (stores the L^T matrix)
        Eigen::LDLT<ScalarMatrix, Eigen::Upper> ldlt(mF.inner());
        const Eigen::MatrixBase<ScalarMatrix> &mLDLT = ldlt.matrixLDLT();
        
        // Get the rank (of the null space)
        Index rank = 0;
        for(Index i = 0; i < N; i++)
            if (mLDLT(i,i) > EPSILON) rank++;

        std::cerr << mLDLT << std::endl;

        // Stop if we are full rank
        if (rank == mLDLT.rows())
            return;
        
        
        
        const Eigen::Block<const ScalarMatrix> mL1(mLDLT, 0,0,rank,rank);
        ScalarMatrix mL2 = mLDLT.block(0,rank+1,rank,N-rank);
        
        if (rank != N) {
            // Gets the null space vectors in mL2
            mL1.template triangularView<Eigen::Upper>().solveInPlace(mL2);
            mL2 = mL2 * ldlt.transpositionsP().transpose();
            
            // Simplify mL2
            removeNullSpacePreImages(mF, mL2);
            
            // Removes unused pre-images
            removeUnusedPreImages<FMatrix>(mF, mY);
        }
    }
    
    
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
            
            Eigen::MatrixXd m1 = mF.get_matrix() * mY;
            Eigen::MatrixXd m2 = _mF * _mY;
            double error = (m1 - m2).norm();
            
            Index delta = (_mY.rows() - mY.rows());
            KQP_LOG_INFO_F(logger, "Error is %g and row difference is %d", %error %delta);
            
            return (delta > 0) && (error < EPSILON * delta);
        }
        
        // (2) Remove unused pre-images using LDL
        if (name == "ldl") {
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
            
            removePreImagesWithLDL(mF, mY);

            Eigen::MatrixXd m1 = mF.get_matrix() * mY;
            Eigen::MatrixXd m2 = _mF * _mY;
            double error = (m1 - m2).norm();
            
            Index delta = (_mY.rows() - mY.rows());
            KQP_LOG_INFO_F(logger, "Error is %g and row difference is %d", %error %delta);
        }
        
        return 0;
    }
}