/*
 This file is part of the Kernel Quantum Probability library (KQP).
 
 KQP is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 KQP is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with KQP.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _KQP_KERNEL_EVD_TESTS_H_
#define _KQP_KERNEL_EVD_TESTS_H_

#include <boost/format.hpp>
#include "feature_matrix/dense.hpp"
#include "kqp.hpp"

namespace kqp {
    namespace kevd_tests {
        extern double tolerance;
        
        /**
         * Computes tr( X1 Y1 D1 Y1^T X1^T  X2 Y2 D2 Y2^T X2^T)
         * 
         * as
         *  tr( D1 Y1^T X1^T  X2 Y2 D2 Y2^T X2^T Y1^T X1^T )
         */
        template<class Derived>
        double trace_function(const FeatureMatrix<Derived> &mX1, 
                              const typename ftraits<Derived>::AltMatrix  &mY1,
                              const typename ftraits<Derived>::RealVector &mD1,
                              
                              const FeatureMatrix<Derived> &mX2, 
                              const typename ftraits<Derived>::AltMatrix  &mY2,
                              const typename ftraits<Derived>::RealVector &mD2) {
            typedef typename ftraits<Derived>::ScalarVector Vector;
            typename ftraits<Derived>::Matrix m;
            inner<Derived>(mX1.derived(), mX2.derived(),m);
            
            m = (mY1.adjoint() * m * mY2).eval();
            
            double trace = 0;
            for(Index i = 0; i < m.rows(); i++) {
                Vector x = m.row(i).adjoint().cwiseProduct(mD2);
                Vector y = m.row(i).adjoint();
                
                trace += mD1[i] * x.dot(y);
            }
            
            return trace;
        }
        
        
        
        
        struct Dense_evd_test {
            int nb_add;
            int n; 
            int max_preimages; 
            int max_lc;
            
            template<class T> 
            int run(const log4cxx::LoggerPtr &logger, T &builder) const {
                typedef typename T::FTraits FTraits;
                typedef typename FTraits::Scalar Scalar;
                typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
                typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
                
                KQP_LOG_INFO(logger, "Kernel EVD with dense vectors and builder=" << KQP_DEMANGLE(builder));
                
                Matrix matrix(n,n);
                matrix.setConstant(0);
                
                // Construction
                for(int i = 0; i < nb_add; i++) {
                    
                    Scalar alpha = Eigen::internal::abs(Eigen::internal::random_impl<Scalar>::run()) + 1e-3;
                    
                    // Generate a number of pre-images
                    int k = (int)(Eigen::internal::random_impl<double>::run() * (double)max_preimages) + 1;
                    Matrix m = Matrix::Random(n, k);
                    
                    // Generate the linear combination matrix
                    int p = (int)(Eigen::internal::random_impl<double>::run() * (double)max_lc) + 1;
                    Matrix mA = Matrix::Random(k, p);
                    
                    KQP_LOG_INFO(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % n % k % k % p);
                    matrix.template selfadjointView<Eigen::Lower>().rankUpdate(m * mA, alpha);
                    
                    
                    builder.add(alpha, DenseMatrix<Scalar>(m), AltMatrix<Scalar>(mA));
                }
                
                // Computing via EVD
                KQP_LOG_INFO(logger, "Computing an LDLT decomposition");
                
                typedef Eigen::LDLT<Matrix> LDLT;
                LDLT ldlt = matrix.template selfadjointView<Eigen::Lower>().ldlt();
                Matrix mL = ldlt.matrixL();
                mL = ldlt.transpositionsP().transpose() * mL;
                DenseMatrix<Scalar>  mU(mL);
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> mU_d = ldlt.vectorD();
                
                
                
                
                // Comparing the results
                
                KQP_LOG_INFO(logger, "Retrieving the decomposition");
                typename FTraits::FMatrix mX;
                typename FTraits::AltMatrix mY;
                typename FTraits::RealVector mD;
                
                builder.get_decomposition(mX, mY, mD);
                
                typename FTraits::AltMatrix mUY = FTraits::AltMatrix::Identity(mU.dimension());
                
                // Computing the difference between operators || U1 - U2 ||^2
                
                KQP_LOG_INFO(logger, "Comparing the decompositions");
                double error = trace_function(mX, mY, mD, mX, mY, mD);       
                error += trace_function(mU, mUY, mU_d, mU, mUY, mU_d);
                error -= 2. * trace_function(mX, mY, mD, mU, mUY, mU_d);
                KQP_LOG_INFO(logger, "Squared error is " << error);
                return error < tolerance ? 0 : 1;
            }
        };
    }
    
}

#endif
