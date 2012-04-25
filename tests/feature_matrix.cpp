#define KQP_NO_EXTERN_TEMPLATE

#include <kqp/feature_matrix/dense.hpp>
#include <kqp/feature_matrix/sparse.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>

using namespace kqp;

DEFINE_LOGGER(logger, "kqp.test.fmatrix");

namespace kqp {
    
    template class SparseDenseMatrix<double>;
    
    int test_densesparse() {
        typedef SparseDenseMatrix<double> FMatrix;
        KQP_FMATRIX_TYPES(FMatrix);
        
        // --- Initialisation
        Matrix<double, Dynamic, Dynamic> m = Matrix<double,Dynamic,Dynamic>::Random(5,8);
        m.row(3) *= 0;
        
        DenseMatrix<double> dMatrix(m);
        SparseDenseMatrix<double> sdMatrix(m,0);

        Matrix<double, Dynamic, Dynamic> m2 = Matrix<double,Dynamic,Dynamic>::Random(5,3);
        m2.row(2) *= 0;
        m2.row(4) *= 0;
        
        DenseMatrix<double> dMatrix2(m2);
        SparseDenseMatrix<double> sdMatrix2(m2,0);

        double error;
        int code = 0;
        
        // --- Test that the number of stored rows is less than for the dense case
        KQP_LOG_INFO_F(logger, "Sparse matrix dense dimension = %d (real = %d)", %sdMatrix.denseDimension() %sdMatrix.dimension()); 
        code |= sdMatrix.denseDimension() >= sdMatrix.dimension();
        
        // --- Test the difference
        error = (sdMatrix.toDense() - m).norm() / m.norm();
        KQP_LOG_INFO_F(logger, "Delta (matrix, sparse dense) is %g", %error);
        
        // --- Test the difference when constructed from a sparse
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> sm(m.rows(), m.cols());        
        sm.reserve((m.array() != 0).template cast<Index>().rowwise().sum());
        for(Index i = 0; i < m.rows(); i++)
            for(Index j = 0; j < m.cols(); j++)
                if (m(i,j) != 0) sm.insert(i,j) = m(i,j);
            
        error = (FMatrix(sm).toDense() - m).norm() / m.norm();
        KQP_LOG_INFO_F(logger, "Delta (sparse matrix, sparse dense) is %g", %error);
        code |= error >= EPSILON;
        
        // --- Test the inner product
        error = (dMatrix.inner() - sdMatrix.inner()).norm() / m.norm();
        KQP_LOG_INFO_F(logger, "Delta (inner, sparse dense) is %g", %error);
        code |= error >= EPSILON;
        
        // --- Test inner product with other matrix
        error = (m.adjoint() * m2 - kqp::inner(sdMatrix,sdMatrix2)).norm() / (m.norm() * m2.norm());
        KQP_LOG_INFO_F(logger, "Delta (inner with other, sparse dense) is %g", %error);
        code |= error >= EPSILON;
        
        
        // --- Test linear combination
        Scalar alpha = 1.2;
        Scalar beta = 2.3;
        ScalarMatrix mA = ScalarMatrix::Random(8,4);
        ScalarMatrix mB = ScalarMatrix::Random(3,4);
        
        ScalarMatrix dLC = alpha * m * mA + beta * m2 * mB;
        FMatrix sdLC = sdMatrix.linear_combination(mA, alpha, sdMatrix2, mB, beta);
        
        error = (sdLC.toDense() - dLC).norm() / dLC.norm();
        KQP_LOG_INFO_F(logger, "Delta (lc, sparse dense) is %g", %error);
        code |= error >= EPSILON;

        return code;
    }

}
int main(int , const char **) {
    int code = 0;
    
    code |= kqp::test_densesparse();
    
    return code;
}