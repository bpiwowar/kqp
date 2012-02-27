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

#include <kqp/evd_update.hpp>
#include <kqp/kernel_evd.hpp>
#include <kqp/alt_matrix.hpp>
#include <kqp/kernel_evd/utils.hpp>

namespace kqp {
    
    
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup KernelEVD
     */
    template <class FMatrix> class IncrementalKernelEVD : public KernelEVD<FMatrix> {
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
        
               
        virtual void _add(Real alpha, const FMatrix &mU, const ScalarAltMatrix &mA) {
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
                
                
                evdRankOneUpdate.update(mD, alpha, v, false, this->selector.get(), false, result, &mZ);
                // Take the new diagonal
                mD = result.mD;
            }
            
            // Update X and Y if the rank has changed
            if (rank_Q > 0) {
                // Add the pre-images
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
            this->cleanup(mX, mY, mD);

        }
        
        
        virtual void _get_decomposition(FMatrix& mX, ScalarAltMatrix &mY, RealVector& mD) const {
            mX = this->mX;
            if (mZ.rows() > 0) {
                this->mY = this->mY * mZ;
                mZ.resize(0,0);
            }
            mY = ScalarAltMatrix(this->mY);
            mD = this->mD;
            
        }
        
        
    private:
        mutable FMatrix mX;
        mutable ScalarMatrix mY;
        mutable ScalarMatrix mZ;
        typename FTraits::RealVector mD;
        
        // Rank-one EVD update
        FastRankOneUpdate<Scalar> evdRankOneUpdate;
        
        // Used in computation
        mutable ScalarMatrix k;
        
        
    };
    
    KQP_KERNEL_EVD_INSTANCIATION(extern, IncrementalKernelEVD);
    
}

#endif
