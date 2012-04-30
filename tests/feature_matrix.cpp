#define KQP_NO_EXTERN_TEMPLATE

#include <kqp/feature_matrix/dense.hpp>
#include <kqp/feature_matrix/sparse.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>

using namespace kqp;

DEFINE_LOGGER(logger, "kqp.test.fmatrix");

namespace kqp {
    
    template class SparseDenseMatrix<double>;
    template class SparseMatrix<double>;
    
    template<typename Scalar> int test_dimension(const SparseMatrix<Scalar>&) { return 0; }
    template<typename Scalar> int test_dimension(const SparseDenseMatrix<Scalar> &sdMatrix) {
        // --- Test that the number of stored rows is less than for the dense case
        KQP_LOG_INFO_F(logger, "Sparse matrix dense dimension = %d (real = %d)", %sdMatrix.denseDimension() %sdMatrix.dimension()); 
        return sdMatrix.denseDimension() >= sdMatrix.dimension();
    }
    
    
    template<typename FMatrix>
    struct FMatrixTest {
        KQP_SCALAR_TYPEDEFS(Scalar);
        std::string fmatrixName;
        
        ScalarMatrix m, m2;
        FMatrix sdMatrix, sdMatrix2;
        double error;
        
        FMatrixTest() : fmatrixName(KQP_DEMANGLE(FMatrix)) {
            KQP_LOG_INFO_F(logger, "*** Tests with %s ***", %fmatrixName);
            
            // --- Initialisation
            m = Matrix<double,Dynamic,Dynamic>::Random(5,8);
            m.row(3) *= 0;
            
            sdMatrix = FMatrix(m);
            
            m2 = Matrix<double,Dynamic,Dynamic>::Random(5,3);
            m2.row(2) *= 0;
            m2.row(4) *= 0;
            
            sdMatrix2 = FMatrix(m2);
        }
        
        
        int code = 0;
        
        void checkError(const std::string &name, double error) {
            if (error < EPSILON) {
                KQP_LOG_INFO_F(logger,  "Error for %s (%s) is %g", %name %fmatrixName %error);
            } else {
                KQP_LOG_ERROR_F(logger, "Error for %s (%s) is %g [!]", %name %fmatrixName %error);
                code = 1;
            }
        }
        
        int test() {
            double error;
            
            
            // --- Test the difference
            error = (sdMatrix.toDense() - m).norm() / m.norm();
            checkError("matrix", error);

            // --- Test adding
            FMatrix sdAdd = sdMatrix;
            sdAdd.add(sdMatrix2);
            error = std::sqrt(
                              ( (sdAdd.toDense().leftCols(m.cols()) - m).squaredNorm() + (sdAdd.toDense().rightCols(m2.cols()) - m2).squaredNorm() )
                              / (m.squaredNorm() + m2.squaredNorm())
                              );
            checkError("add matrix", error);
            
            // --- Test adding
            sdAdd = sdMatrix;
            std::vector<bool> which(3,true);
            which[2] = false;
            ScalarMatrix m2Sel;
            select_columns(which, m2, m2Sel);
            sdAdd.add(sdMatrix2, &which);
            error = std::sqrt(
                              ( (sdAdd.toDense().leftCols(m.cols()) - m).squaredNorm() + (sdAdd.toDense().rightCols(m2Sel.cols()) - m2Sel).squaredNorm() )
                              / (m.squaredNorm() + m2.squaredNorm())
                              );
            checkError("add matrix [self]", error);
            

            // --- Test subset
            std::vector<bool> selected(8,true);
            selected[2] = selected[6] = false;
            ScalarMatrix mSelect;
            kqp::select_columns(selected, m, mSelect);
            FMatrix sdSelect;
            sdMatrix.subset(selected.begin(), selected.end(), sdSelect);
            error = (sdSelect.toDense() - mSelect).norm() / mSelect.norm();
            checkError("select matrix", error);
            
            sdSelect = sdMatrix;
            sdSelect.subset(selected.begin(), selected.end(), sdSelect);
            error = (sdSelect.toDense() - mSelect).norm() / mSelect.norm();
            checkError("select matrix [self]", error);
            
            return code;
        }
        
    };
    
    template<typename FMatrix>
    struct SparseTest : public FMatrixTest<FMatrix> {
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        typedef FMatrixTest<FMatrix> Base;
        using Base::code;
        using Base::sdMatrix;
        using Base::sdMatrix2;
        using Base::m;
        using Base::m2;
        using Base::error;
        using Base::fmatrixName;

        int test() {
            code = FMatrixTest<FMatrix>::test();
            
            // --- test dimension (if applicable)
            test_dimension(sdMatrix);
            
            
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
            error = (m.adjoint() * m - sdMatrix.inner()).norm() / m.norm();
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
    };
    
}

int main(int , const char **) {
    int code = 0;
    
    code |= kqp::SparseTest<SparseDenseMatrix<double>>().test();
    
    code |= kqp::SparseTest<SparseDenseMatrix<double>>().test();
    
    code |= kqp::FMatrixTest<DenseMatrix<double>>().test();
    
    return code;
}