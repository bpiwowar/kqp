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

#ifndef __KQP_INCREMENTAL_BUILDER_H__
#define __KQP_INCREMENTAL_BUILDER_H__

#include <iostream>
#include <limits>

#include <kqp/cleanup.hpp>
#include <kqp/evd_update.hpp>
#include <kqp/kernel_evd.hpp>
#include <kqp/alt_matrix.hpp>
#include <kqp/kernel_evd/utils.hpp>

namespace kqp {
    
    
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup KernelEVD
     */
    template <class FMatrix> class IncrementalKernelEVD : public KernelEVD<FMatrix>, public Cleaner<FMatrix> {
    public:
        KQP_FMATRIX_TYPES(FMatrix);
        
        // Indirect comparator (decreasing order)
        struct Comparator {
            const RealVector &v;
            Comparator(const RealVector &v) : v(v) {}
            bool operator()(Index i, Index j) {
                return this->v(i) > this->v(j);
            }
        };
        
        IncrementalKernelEVD() {}
        
               
        virtual void _add(Real alpha, const FMatrix &mU, const ScalarAltMatrix &mA) override {
            // --- Pre-computations
            
            // Compute W = Y^T X^T
            inner(mX, mU, k);
            ScalarMatrix mW;
            noalias(mW) = mY.transpose() * k * mA;
            
            // Compute V^T V
            ScalarMatrix vtv = mA.transpose() * mU.inner() * mA;
            vtv -= mW.adjoint() * mW;
            
            
            // (thin) eigen-value decomposition of V^T V
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(vtv);
            ScalarMatrix mQ;
            ScalarMatrix mQ_null;
            RealVector mQD;
            kqp::thinEVD(evd, mQ, mQD, &mQ_null);

            Index rank_Q = mQ.cols();             
            mQD = mQD.cwiseAbs().cwiseSqrt();
            for(Index i = 0; i < rank_Q; i++)
                mQ.col(i) /= mQD(i);

            // --- Update
           
            ScalarMatrix m(mW.rows() + rank_Q, mW.cols());
                        
            m.topLeftCorner(mW.rows(), rank_Q) =  mW * mQ * mQD.asDiagonal();
            m.topRightCorner(mW.rows(), mW.cols() - rank_Q) = mW * mQ_null;                
            
            m.bottomLeftCorner(rank_Q, rank_Q) = mQD.template cast<Scalar>().asDiagonal();
            m.bottomRightCorner(rank_Q, mW.cols() - rank_Q).setConstant(0) ;
            

            for(Index i = 0; i < m.cols(); i++) {
                EvdUpdateResult<Scalar> result;
                // Update
                ScalarVector v(m.rows());
                v.head(mZ.cols()) = mZ.adjoint() * m.col(i).head(mZ.cols());
                v.tail(m.rows() - mZ.cols()) = m.col(i).tail(m.rows() - mZ.cols());
                
                
                evdRankOneUpdate.update(mD, alpha, v, false, m_cleaner.get() ? m_cleaner->selector.get() : nullptr, false, result, &mZ);
                // Take the new diagonal
                mD = result.mD;
            }
            
            // Update X and Y if the rank has changed
            if (rank_Q > 0) {
                // Add the pre-images from U
                mX.add(mU);
                
                // Update mY
                Index old_Y_rows = mY.rows();
                Index old_Y_cols = mY.cols();
                mY.conservativeResize(mY.rows() + mA.rows(), mY.cols() + mQ.cols());

                mY.bottomRightCorner(mA.rows(), mQ.cols()) = mA * mQ;
                mY.bottomLeftCorner(mA.rows(), old_Y_cols).setConstant(0);
                if (old_Y_rows > 0)
                    mY.topRightCorner(old_Y_rows, mQ.cols()) = - mY.topLeftCorner(old_Y_rows, mW.rows()) * mW * mQ;
            }
                              
            // (4) Clean-up

            // First, tries to remove unused pre-images images
            removePreImagesWithNullSpace(mX, mY);
            
            // --- Ensure we have a small enough number of pre-images
            if (mX.size() > (this->preImageRatios.second * mD.rows())) {
                
                // Get rid of Z
                mY = mY * mZ;
                mZ = ScalarMatrix::Identity(mX.size(), mX.size());

                if (mX.can_linearly_combine()) {
                    // Easy case: we can linearly combine pre-images
                    mX = mX.linear_combination(mY);
                    mY = ScalarMatrix::Identity(mX.size(), mX.size());
                } else {
                    // Use QP approach
                    ReducedSetWithQP<FMatrix> qp_rs;
                    qp_rs.run(this->preImageRatios.first * mD.rows(), mX, mY, mD);
                    
                    // Get the decomposition
                    mX = qp_rs.getFeatureMatrix();
                    mY = qp_rs.getMixtureMatrix();
                    mD = qp_rs.getEigenValues();
                    
                    // The decomposition is not orthonormal anymore
                    kqp::orthonormalize(mX, mY, mD);
                }
                
                
            }
            
        }
        
        // Gets the decomposition
        virtual Decomposition<FMatrix> getDecomposition() const override {
            Decomposition<FMatrix> d;
            
            d.mX = this->mX;
            
            const_cast<ScalarMatrix&>(this->mY) = this->mY * this->mZ;
            const_cast<ScalarMatrix&>(this->mZ) = ScalarMatrix::Identity(mY.rows(), mY.rows());
            
            d.mY = ScalarAltMatrix(this->mY);
            
            d.mD = this->mD;
            
            return d;
        }
        
        
    private:
        //! The feature matrix with n pre-images
        FMatrix mX;
        
        //! The n x r matrix such that \f$X Y\f$ is orthonormal
        ScalarMatrix mY;
        
        //! A unitary r x r matrix
        ScalarMatrix mZ;
        
        //! A diagonal matrix (vector representation)
        RealVector mD;
        
        // Rank-one EVD update
        FastRankOneUpdate<Scalar> evdRankOneUpdate;
        
        // Used in computation
        mutable ScalarMatrix k;
        
        
        
        boost::shared_ptr<Cleaner<FMatrix>> m_cleaner;
    };
    
    KQP_KERNEL_EVD_INSTANCIATION(extern, IncrementalKernelEVD);
    
}

#endif
