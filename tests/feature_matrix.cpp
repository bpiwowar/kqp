#define KQP_NO_EXTERN_TEMPLATE

#include <kqp/feature_matrix/dense.hpp>
#include <kqp/feature_matrix/sparse.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>

using namespace kqp;

DEFINE_LOGGER(logger, "kqp.test.fmatrix");

namespace kqp {
    
    template class SparseDense<double>;
    template class Sparse<double>;
    
    template<typename Scalar> int test_dimension(const Sparse<Scalar>&) { return 0; }
    
    template<typename Scalar> int test_dimension(const SparseDense<Scalar> &sdMatrix) {
        // --- Test that the number of stored rows is less than for the dense case
        KQP_LOG_INFO_F(logger, "Sparse matrix dense dimension = %d (real = %d)", %sdMatrix.denseDimension() %sdMatrix.dimension()); 
        return sdMatrix.denseDimension() >= sdMatrix.dimension();
    }
    
    
    template<typename KQPSpace, typename KQPMatrix>
    struct FMatrixTest {
        Index dimension = 5;
        
        typedef typename KQPMatrix::ScalarMatrix::Scalar Scalar;
        KQP_SCALAR_TYPEDEFS(Scalar);
        std::string fmatrixName;
        
        ScalarMatrix m, m2;
        KQPMatrix sdMatrix, sdMatrix2;
        double error;
        
        FMatrixTest() : fmatrixName(KQP_DEMANGLE(KQPMatrix)) {
            KQP_LOG_INFO_F(logger, "*** Tests with %s ***", %fmatrixName);
            
            // --- Initialisation
            m = Matrix<double,Dynamic,Dynamic>::Random(dimension,8);
            m.row(3) *= 0;
            
            sdMatrix = KQPMatrix(m);
            
            m2 = Matrix<double,Dynamic,Dynamic>::Random(dimension,4);
            m2.row(2) *= 0;
            m2.row(4) *= 0;
            
            sdMatrix2 = KQPMatrix(m2);
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
            KQPSpace fs(this->dimension);
            
            
            // --- Test the difference
            error = (sdMatrix.toDense() - m).norm() / m.norm();
            checkError("matrix", error);
            
            // --- Test adding
            {
                KQPMatrix sdAdd = sdMatrix;
                sdAdd.add(sdMatrix2);
                error = std::sqrt(
                                  ( (sdAdd.toDense().leftCols(m.cols()) - m).squaredNorm() + (sdAdd.toDense().rightCols(m2.cols()) - m2).squaredNorm() )
                                  / (m.squaredNorm() + m2.squaredNorm())
                                  );
                checkError("add matrix", error);
            }
            
            // --- Test adding 
            {
                KQPMatrix sdAdd = sdMatrix;
                std::vector<bool> which(4,true);
                which[2] = false;
                ScalarMatrix m2Sel;
                select_columns(which, m2, m2Sel);
                sdAdd.add(sdMatrix2, &which);
                error = std::sqrt(
                                  ( (sdAdd.toDense().leftCols(m.cols()) - m).squaredNorm() + (sdAdd.toDense().rightCols(m2Sel.cols()) - m2Sel).squaredNorm() )
                                  / (m.squaredNorm() + m2.squaredNorm())
                                  );
                checkError("add matrix [subset]", error);
            }
            
            // --- Test subset

            {
                std::vector<bool> selected(8,true);
                selected[2] = selected[6] = false;
                ScalarMatrix mSelect;
                kqp::select_columns(selected, m, mSelect);

                FMatrixBasePtr sdSelect = sdMatrix.subset(selected.begin(), selected.end());
                error = (sdSelect->template as<KQPMatrix>().toDense() - mSelect).norm() / mSelect.norm();
                checkError("subset", error);
            }
            
            // --- Test the inner product
            error = (m.adjoint() * m - fs.k(sdMatrix)).norm() / m.norm();
            checkError("gram matrix", error);

            // --- Test inner product with other matrix
            error = (m.adjoint() * m2 - fs.k(sdMatrix, sdMatrix2)).norm() / (m.norm() * m2.norm());
            checkError("inner", error);
            
            // --- Test linear combination
            if (fs.canLinearlyCombine()) {
                Scalar alpha = 1.2;
                Scalar beta = 2.3;
                ScalarMatrix mA = ScalarMatrix::Random(m.cols(),4);
                ScalarMatrix mB = ScalarMatrix::Random(m2.cols(),4);
                
                ScalarMatrix dLC = alpha * m * mA + beta * m2 * mB;
                ScalarAltMatrix _mB(mB);
                FMatrixBasePtr sdLC = fs.linearCombination(sdMatrix, mA, alpha, &sdMatrix2, &_mB, beta);
                
                error = (sdLC->template as<KQPMatrix>().toDense() - dLC).norm() / dLC.norm();
                KQP_LOG_INFO_F(logger, "Delta (lc, %s) is %g", %fmatrixName %error);
                code |= error >= EPSILON;
            }
            
            return code;
        }
        
    };
    
    template<typename KQPSpace, typename KQPMatrix>
    struct SparseTest : public FMatrixTest<KQPSpace, KQPMatrix> {
        typedef typename KQPMatrix::ScalarMatrix::Scalar Scalar;
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        typedef FMatrixTest<KQPSpace, KQPMatrix> Base;
        using Base::code;
        using Base::sdMatrix;
        using Base::sdMatrix2;
        using Base::m;
        using Base::m2;
        using Base::error;
        using Base::fmatrixName;
        
        int test() {
            KQPSpace fs(this->dimension);
            
            code = Base::test();
            
            // --- test dimension (if applicable)
            test_dimension(sdMatrix);
            
            
            // --- Test the difference when constructed from a sparse (Row major)
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor> sm(m.rows(), m.cols());        
            sm.reserve((m.array() != 0).template cast<Index>().rowwise().sum());
            for(Index i = 0; i < m.rows(); i++)
                for(Index j = 0; j < m.cols(); j++)
                    if (m(i,j) != 0) sm.insert(i,j) = m(i,j);
            
            error = (KQPMatrix(sm).toDense() - m).norm() / m.norm();
            KQP_LOG_INFO_F(logger, "Delta (sparse matrix, %s) is %g", %fmatrixName %error);
            code |= error >= EPSILON;
            
            // --- Test the difference when constructed from a sparse (Col major)
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor> smCol(m.rows(), m.cols());        
            smCol.reserve((m.array() != 0).template cast<Index>().colwise().sum());
            for(Index i = 0; i < m.rows(); i++)
                for(Index j = 0; j < m.cols(); j++)
                    if (m(i,j) != 0) smCol.insert(i,j) = m(i,j);
            
            error = (KQPMatrix(smCol).toDense() - m).norm() / m.norm();
            KQP_LOG_INFO_F(logger, "Delta (sparse matrix, %s) is %g", %fmatrixName %error);
            code |= error >= EPSILON;
            
            return code;
        }
    };

    int test_dense(std::deque<std::string> &) {
        return kqp::FMatrixTest<DenseSpace<double>, Dense<double>>().test();  
    }
    int test_sparse(std::deque<std::string> &) {
        return kqp::SparseTest<SparseDenseSpace<double>, SparseDense<double>>().test();  
    }
    int test_sparseDense(std::deque<std::string> &) {
        return kqp::SparseTest<SparseSpace<double>, Sparse<double>>().test();  
    }
}

#include "main-tests.inc"
DEFINE_TEST("dense", test_dense);
DEFINE_TEST("sparse-dense", test_sparse);
DEFINE_TEST("sparse", test_sparseDense);
