#include <Eigen/Dense>

#include "kqp.hpp"

#include "kernel_evd.hpp"
#include "kernel_evd/dense_direct.hpp"
#include "kernel_evd/accumulator.hpp"
#include "kernel_evd/incremental.hpp"
#include "feature_matrix/dense.hpp"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd")

namespace kqp {
    namespace { double tolerance = 1e-10; }
    
    /**
     * Computes || D1^T Y1^T X1^T X2 Y2 D2 ||^2.
     *
     * If one of the matrix is the zero matrix.
     */
    template<class FMatrix>
    double inner(const FMatrix &mX1, 
                 const typename ftraits<FMatrix>::Matrix  &mY1,
                 const typename ftraits<FMatrix>::RealVector &mD1,
                 
                 const FMatrix &mX2, 
                 const typename ftraits<FMatrix>::Matrix  &mY2,
                 const typename ftraits<FMatrix>::RealVector &mD2) {
        
        const typename ftraits<FMatrix>::Matrix k;
        inner(mX1, mX2,k);
        
        switch ((is_empty(mD1) ? 8 : 0) + (is_empty(mY1) ? 4 : 0) + (is_empty(mY2) ? 2 : 0) + (is_empty(mD2) ? 1 : 0)) {
            case 0: return k.squaredNorm();
            case 1: return (k * mD2.asDiagonal()).squaredNorm();
            case 2: return (k * mY2).squaredNorm();
            case 3: return (k * mY2 * mD2.asDiagonal()).squaredNorm();
                
            case 4: return (mY1.adjoint() * k).squaredNorm();
            case 5: return (mY1.adjoint() * k * mD2.asDiagonal()).squaredNorm();
            case 6: return (mY1.adjoint() * k * mY2).squaredNorm();
            case 7: return (mY1.adjoint() * k * mY2 * mD2.asDiagonal()).squaredNorm();
                
            case 8: return (mD1.asDiagonal() * k).squaredNorm();
            case 9: return (mD1.asDiagonal() * k * mD2.asDiagonal()).squaredNorm();
            case 10: return (mD1.asDiagonal() * k * mY2).squaredNorm();
            case 11: return (mD1.asDiagonal() * k * mY2 * mD2.asDiagonal()).squaredNorm();
                
            case 12: return (mD1.asDiagonal() * mY1.adjoint() * k).squaredNorm();
            case 13: return (mD1.asDiagonal() * mY1.adjoint() * k * mD2.asDiagonal()).squaredNorm();
            case 14: return (mD1.asDiagonal() * mY1.adjoint() * k * mY2).squaredNorm();
            case 15: return (mD1.asDiagonal() * mY1.adjoint() * k * mY2 * mD2.asDiagonal()).squaredNorm();
        }
        
        KQP_THROW_EXCEPTION(assertion_exception, "Unknown case for inner product computation");
        
    }
    
    /**
     * Computes || D1^T Y1^T X1^T X1 Y1 D1 ||^2.
     *
     * If one of the matrix pointer is null, it assumes identity.
     */
    template<class FMatrix>
    double inner(const FMatrix &mX, 
                 const typename ftraits<FMatrix>::Matrix  &mY,
                 const typename ftraits<FMatrix>::RealVector &mD) {
        
        const typename ftraits<FMatrix>::Matrix& k = mX.inner();
        
        switch ((is_empty(mY) ? 2 : 0) + (is_empty(mD) ? 1 : 0)) {
            case 0: return k.squaredNorm();
            case 1: return (mD.asDiagonal() * k * mD.asDiagonal()).squaredNorm();
            case 2: return (mY.adjoint() * k * mY).squaredNorm();
            case 3: return (mD.asDiagonal() * mY.adjoint() * k * mY * mD.asDiagonal()).squaredNorm();
        }
        
        KQP_THROW_EXCEPTION(assertion_exception, "Unknown case for inner product computation");
    }
    
    using namespace Eigen;
    
    template<class T> int direct_evd(int n, int k, T &builder) {
        typedef typename T::FTraits FTraits;
        typedef typename FTraits::Scalar Scalar;
        typedef Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        
        KQP_LOG_INFO(logger, "Kernel EVD with dense vectors and builder=" << KQP_DEMANGLE(builder));
        
        Matrix matrix(n,n);
        matrix.setConstant(0);
        
        for(int i = 0; i < k; i++) {
            Vector v = Vector::Random(n);
            
            Scalar alpha = Eigen::internal::abs(Eigen::internal::random_impl<Scalar>::run()) + 1e-3;
            matrix.template selfadjointView<Eigen::Lower>().rankUpdate(v, alpha);
            builder.add(alpha, DenseMatrix<Scalar>(v), EMPTY<Scalar>::matrix);
        }
        
        // Computing via EVD
        KQP_LOG_INFO(logger, "Computing an LDLT decomposition");
        
        typedef Eigen::LDLT<Matrix> LDLT;
        LDLT ldlt = matrix.template selfadjointView<Eigen::Lower>().ldlt();
        Matrix mL = ldlt.matrixL();
        mL = ldlt.transpositionsP().transpose() * mL;
        DenseMatrix<Scalar>  mU(mL);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> mU_d = ldlt.vectorD();
        
        for(Index i = 0; i < mU_d.size(); i++)
            if (mU_d[i] < 0) mU_d[i] = 0;
        mU_d = mU_d.cwiseSqrt();
        
        
        
        // Comparing the results
        KQP_LOG_INFO(logger, "Comparing the decompositions");
        
        typename FTraits::FMatrix mX;
        typename FTraits::Matrix mY;
        typename FTraits::RealVector mD;
        
        builder.get_decomposition(mX, mY, mD);
        
        
        // || XX^T - UU^T||^2 = ||X^T X||^2 + ||U^T U||^2 - 2 ||U^T X||^2
        
        double error = inner(mX, mY, mD);
        
        error += inner<typename FTraits::FMatrix>(mU, EMPTY<Scalar>::matrix, mU_d);
        error -= 2. * inner<typename FTraits::FMatrix>(mX, mY, mD, mU, EMPTY<Scalar>::matrix, mU_d);
        
        KQP_LOG_INFO(logger, "Squared error is " << error);
        
        return error < tolerance ? 0 : 1;
    }
    
    int kevd_tests(std::vector<std::string> &args) {
        std::string name = args[0];
        Index n = 10;
        
        // Constant random seed
        
        
        if (name == "direct-builder") {
            DenseDirectBuilder<double> builder(n);
            return direct_evd(n, 5, builder);
        } else if (name == "accumulator") {
            AccumulatorKernelEVD<DenseMatrix<double> > builder;
            return direct_evd(n, 5, builder);            
        } else if (name == "incremental") {
            IncrementalKernelEVD<DenseMatrix<double> > builder;
            return direct_evd(n, 5, builder);            
        }
        
        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown evd_update_test [%s]", %name);
        
    }
    
    
}