#define KQP_NO_EXTERN_TEMPLATE

#include <kqp/feature_matrix/dense.hpp>
#include <kqp/feature_matrix/sparse.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>

using namespace kqp;

DEFINE_LOGGER(logger, "kqp.test.fmatrix");

namespace kqp {
    
    template class SparseDenseMatrix<double>;
    
    template<typename Scalar> int test_dimension(const SparseMatrix<Scalar>&) { return 0; }
    template<typename Scalar> int test_dimension(const SparseDenseMatrix<Scalar> &sdMatrix) {
        // --- Test that the number of stored rows is less than for the dense case
        KQP_LOG_INFO_F(logger, "Sparse matrix dense dimension = %d (real = %d)", %sdMatrix.denseDimension() %sdMatrix.dimension()); 
        return sdMatrix.denseDimension() >= sdMatrix.dimension();
    }
    
    template<typename FMatrix>
    int test_densesparse() {
        KQP_FMATRIX_TYPES(FMatrix);
        std::string fmatrixName(KQP_DEMANGLE(FMatrix));
        KQP_LOG_INFO_F(logger, "*** Tests with %s ***", %fmatrixName);
        
        // --- Initialisation
        Matrix<double, Dynamic, Dynamic> m = Matrix<double,Dynamic,Dynamic>::Random(5,8);
        m.row(3) *= 0;
        
        DenseMatrix<double> dMatrix(m);
        FMatrix sdMatrix(m,0);
        
        Matrix<double, Dynamic, Dynamic> m2 = Matrix<double,Dynamic,Dynamic>::Random(5,3);
        m2.row(2) *= 0;
        m2.row(4) *= 0;
        
        DenseMatrix<double> dMatrix2(m2);
        FMatrix sdMatrix2(m2,0);
        
        double error;
        int code = 0;
        
        // --- test dimension (if applicable)
        test_dimension(sdMatrix);
        
        // --- Test the difference
        error = (sdMatrix.toDense() - m).norm() / m.norm();
        KQP_LOG_INFO_F(logger, "Delta (matrix, %s) is %g", %fmatrixName %error);
        
        // --- Test the difference when constructed from a sparse (Row major)
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> sm(m.rows(), m.cols());        
        sm.reserve((m.array() != 0).template cast<Index>().rowwise().sum());
        for(Index i = 0; i < m.rows(); i++)
            for(Index j = 0; j < m.cols(); j++)
                if (m(i,j) != 0) sm.insert(i,j) = m(i,j);
        
        error = (FMatrix(sm).toDense() - m).norm() / m.norm();
        KQP_LOG_INFO_F(logger, "Delta (sparse matrix, %s) is %g", %fmatrixName %error);
        code |= error >= EPSILON;
        
        // --- Test the difference when constructed from a sparse (Col major)
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor> smCol(m.rows(), m.cols());        
        smCol.reserve((m.array() != 0).template cast<Index>().colwise().sum());
        for(Index i = 0; i < m.rows(); i++)
            for(Index j = 0; j < m.cols(); j++)
                if (m(i,j) != 0) smCol.insert(i,j) = m(i,j);
        
        error = (FMatrix(smCol).toDense() - m).norm() / m.norm();
        KQP_LOG_INFO_F(logger, "Delta (sparse matrix, %s) is %g", %fmatrixName %error);
        code |= error >= EPSILON;
        
        
        // --- Test the inner product
        error = (dMatrix.inner() - sdMatrix.inner()).norm() / m.norm();
        KQP_LOG_INFO_F(logger, "Delta (inner, %s) is %g", %fmatrixName %error);
        code |= error >= EPSILON;
        
        // --- Test inner product with other matrix
        error = (m.adjoint() * m2 - kqp::inner(sdMatrix,sdMatrix2)).norm() / (m.norm() * m2.norm());
        KQP_LOG_INFO_F(logger, "Delta (inner with other, %s) is %g", %fmatrixName %error);
        code |= error >= EPSILON;
        
        
        // --- Test linear combination
        if (sdMatrix.canLinearlyCombine()) {
            Scalar alpha = 1.2;
            Scalar beta = 2.3;
            ScalarMatrix mA = ScalarMatrix::Random(8,4);
            ScalarMatrix mB = ScalarMatrix::Random(3,4);
            
            ScalarMatrix dLC = alpha * m * mA + beta * m2 * mB;
            FMatrix sdLC = sdMatrix.linear_combination(mA, alpha, sdMatrix2, mB, beta);
            
            error = (sdLC.toDense() - dLC).norm() / dLC.norm();
            KQP_LOG_INFO_F(logger, "Delta (lc, %s) is %g", %fmatrixName %error);
            code |= error >= EPSILON;
        }
        return code;
    }
    
}
int main(int , const char **) {
    int code = 0;
    
    code |= kqp::test_densesparse<SparseDenseMatrix<double>>();
    
    code |= kqp::test_densesparse<SparseMatrix<double>>();
    
    return code;
}