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

#include "evd_update.hpp"
#include "kernel_evd.hpp"
#include "alt_matrix.hpp"
#include "coneprog.hpp"
#include "utils.hpp"

namespace kqp {
    
    
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup KernelEVD
     */
    template <class FMatrix> class IncrementalKernelEVD : public KernelEVD<FMatrix> {
    public:
        typedef ftraits<FMatrix> FTraits;
        typedef typename FTraits::Matrix Matrix;
        typedef typename FTraits::Scalar Scalar;
        typedef typename FTraits::Real Real;
        typedef typename FTraits::RealVector RealVector;
        typedef typename FTraits::ScalarVector ScalarVector;
        
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        
        // Indirect comparator (decreasing order)
        struct Comparator {
            const RealVector &v;
            Comparator(const RealVector &v) : v(v) {}
            bool operator()(Index i, Index j) {
                return this->v(i) > this->v(j);
            }
        };
        
               
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrix &mU, const typename FTraits::AltMatrix &mA) {
            // --- Pre-computations
            
            // Compute W = Y^T X^T
            inner(mX, mU, k);
            Matrix mW;
            mW.noalias() = mY.adjoint() * k * mA;
            
            // Compute V^T V
            Matrix vtv = mA.adjoint() * mU.inner() * mA - mW.adjoint() * mW;
            
            
            // (thin) eigen-value decomposition of V^T V
            Eigen::SelfAdjointEigenSolver<Matrix> evd(vtv);
            Matrix mQ;
            Matrix mQ_null;
            RealVector mQD;
            kqp::thinEVD(evd, mQ, mQD, &mQ_null);

            Index rank_Q = mQ.cols();             
            mQD = mQD.cwiseAbs().cwiseSqrt();
            for(Index i = 0; i < rank_Q; i++)
                mQ.col(i) /= mQD(i);

            // --- Update
           
            Matrix m(mW.rows() + rank_Q, mW.cols());
                        
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
                
                
                evdRankOneUpdate.update(mD, alpha, v, false, selector.get(), false, result, &mZ);
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
            
            // Remove unused images
//            removeUnusedPreImages(mX, mY);

            // Ensure we have a small enough number of pre-images
            if (mX.size() > (pre_images_per_rank * mD.rows())) {
                if (mX.can_linearly_combine()) {
                    // Easy case: we can linearly combine pre-images
                    AltMatrix<Scalar> m;
                    m.swap_dense(mY);
                    mX = mX.linear_combination(m);
                    mY.setIdentity(mX.size(), mX.size());
                } else {
                    // Optimise
                }
                
            }
        }
        
        
        virtual void _get_decomposition(typename FTraits::FMatrix& mX, typename FTraits::AltMatrix &mY, typename FTraits::RealVector& mD) {
            mX = this->mX;
            if (mZ.rows() > 0) {
                this->mY = this->mY * mZ;
                mZ.resize(0,0);
            }
            mY = AltMatrix<Scalar>(this->mY);
            mD = this->mD;
            
        }
        
        
    private:
        FMatrix mX;
        Matrix mY;
        Matrix mZ;
        typename FTraits::RealVector mD;
        
        // Rank-one EVD update
        FastRankOneUpdate<Scalar> evdRankOneUpdate;
        
        // Used in computation
        mutable Matrix k;
        
        // Rank selector
        boost::shared_ptr<Selector> selector;

        //! Ratio of pre-images per rank (target)
        float pre_images_per_rank_target;
        
        //! Ratio of the number of pre-images to the rank (must be >= 1)
        float pre_images_per_rank;
        
    };
    
    KQP_KERNEL_EVD_INSTANCIATION(extern, IncrementalKernelEVD);
    
}

#endif
