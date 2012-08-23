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

#include <kqp/kqp.hpp>

#include <kqp/kernel_evd.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include <kqp/trace.hpp>

namespace kqp {
    
    template<typename Scalar>
    struct KernelOperators {
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        /**
         * Computes the trace of the product of two kernel operators, i.e. of
         * \f$ tr( X_1 Y_1 D_1 Y_1^\dagger X1^\dagger  X_2 Y_2 D_2 Y_2^\dagger X_2^\dagger) \f$
         *
         */
        static Scalar traceProduct(const Space<Scalar> &fs,
                                   
                                   const FeatureMatrix<Scalar> &mX1, 
                                   const ScalarAltMatrix  &mY1,
                                   const RealVector &mD1,
                                   
                                   const FeatureMatrix<Scalar> &mX2, 
                                   const ScalarAltMatrix  &mY2,
                                   const RealVector &mD2) {
            
            ScalarMatrix m = fs.k(mX1, mY1, mX2, mY2);
            
            return (m.adjoint() * mD1.asDiagonal() * m * mD2.asDiagonal()).trace();
        }
        
        /**
         * Computes the difference between two operators using trace functions
         */
        static Scalar difference(const Space<Scalar> &fs,
                                 
                                 const FeatureMatrix<Scalar> &mX1, 
                                 const ScalarAltMatrix  &mY1,
                                 const RealVector &mD1,
                                 
                                 const FeatureMatrix<Scalar> &mX2, 
                                 const ScalarAltMatrix  &mY2,
                                 const RealVector &mD2) {
            
            double tr1 = traceProduct(fs, mX1, mY1, mD1, mX1, mY1, mD1);       
            double tr2 = traceProduct(fs, mX2, mY2, mD2, mX2, mY2, mD2);
            double tr12 = traceProduct(fs, mX1, mY1, mD1, mX2, mY2, mD2);
            
            return tr1 + tr2 - 2. * tr12;
        }
    };
    
    namespace kevd_tests {
        
        
        extern double tolerance;
        
        
        struct Dense_evd_test {
            //! Number of updates to make
            int nb_add;
            //! Dimension of the space
            int n; 
            
            //! Minimum of pre-images for each updates
            int min_preimages;
            //! Maximum of pre-images for each updates
            int max_preimages; 
            
            //! Minimum and maximum number of vectors to add at each update
            int min_lc;
            int max_lc;
            
            Dense_evd_test() : min_preimages(1), min_lc(1) {}
            
            template<class Scalar> 
            int run(const log4cxx::LoggerPtr &logger, KernelEVD<Scalar> &builder) const {
                KQP_SCALAR_TYPEDEFS(Scalar);
                
                KQP_LOG_INFO_F(logger, "Kernel EVD with dense vectors and builder \"%s\" (pre-images = %d, linear combination = %d)", %KQP_DEMANGLE(builder) %max_preimages %max_lc);
                
                ScalarMatrix matrix(n,n);
                matrix.setConstant(0);
                
                // Construction
                for(int i = 0; i < nb_add; i++) {
                    
                    Scalar alpha = Eigen::internal::abs(Eigen::internal::random_impl<Scalar>::run()) + 1e-3;
                    
                    int k = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_preimages-min_preimages)) + min_preimages;
                    int p = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_lc-min_lc)) + min_lc;
                    KQP_LOG_INFO(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % n % k % k % p);
                    
                    // Generate a number of pre-images
                    ScalarMatrix m = ScalarMatrix::Random(n, k);
                    
                    // Generate the linear combination matrix
                    ScalarMatrix mA = ScalarMatrix::Random(k, p);
                    
                    matrix.template selfadjointView<Eigen::Lower>().rankUpdate(m * mA, alpha);
                    
                    
                    builder.add(alpha, Dense<Scalar>::create(m), mA);
                }
                
                // Computing via EVD
                KQP_LOG_INFO(logger, "Computing an LDLT decomposition");
                
                typedef Eigen::LDLT<ScalarMatrix> LDLT;
                LDLT ldlt = matrix.template selfadjointView<Eigen::Lower>().ldlt();
                ScalarMatrix mL = ldlt.matrixL();
                mL = ldlt.transpositionsP().transpose() * mL;
                FeatureMatrix<Scalar>  mU(Dense<Scalar>::create(mL));
                Eigen::Matrix<Scalar,Dynamic,1> mU_d = ldlt.vectorD();
                
                
                
                
                // Comparing the results
                
                KQP_LOG_INFO(logger, "Retrieving the decomposition");
                
                auto kevd = builder.getDecomposition();
                
                ScalarAltMatrix mUY = Eigen::Identity<Scalar>(mL.rows(), mL.rows());
                
                KQP_LOG_DEBUG(logger, "=== Decomposition ===");
//                KQP_LOG_DEBUG(logger, "X = " << kevd.mX);
                KQP_LOG_DEBUG(logger, "Y = " << kevd.mY);
                KQP_LOG_DEBUG(logger, "D = " << kevd.mD);
                
                
                // Computing the difference between operators || U1 - U2 ||^2
                
                KQP_LOG_INFO(logger, "Comparing the decompositions");
                double error = KernelOperators<Scalar>::difference(kevd.fs, kevd.mX, kevd.mY, kevd.mD, mU, mUY, mU_d);
                
                KQP_LOG_INFO_F(logger, "Squared error is %e", %error);
                return error < tolerance ? 0 : 1;
            }
        };
        
        // --- Kernel EVD builders
        
        struct Builder {
            virtual int run(const Dense_evd_test &) const = 0;  
        };
        
        
        struct Direct_builder : public Builder {
            virtual int run(const Dense_evd_test &) const;
        };
        
        struct Accumulator : public Builder {
            Accumulator(bool use_lc) : use_lc(use_lc) {}
            virtual int run(const Dense_evd_test &) const;
            
            bool use_lc;
        };
        struct Incremental : public Builder {
            virtual int run(const Dense_evd_test &) const;
        };
        
        struct DivideAndConquer : public Builder {
            virtual int run(const Dense_evd_test &) const;
        };
        
    }
    
}

#endif
