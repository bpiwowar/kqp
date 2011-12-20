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
#include "null_space.hpp"
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
        
        IncrementalKernelEVD() :preImagesPerRank(std::numeric_limits<float>::infinity()) {
            
        }
        
        IncrementalKernelEVD(double minimumRelativeError, Index maxRank, float preImagesPerRank) :
        preImagesPerRank(preImagesPerRank) {
            
        }
        
        
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrix &mU, const typename FTraits::AltMatrix &mA) {
            // --- Pre-computations
            
            // Compute W = Y^T X^T
            inner(mX, mU, k);
            Matrix mW;
            mW.noalias() = mY.adjoint() * k * mA;
            
            // Compute V^T V
            Matrix vtv = mA.adjoint() * mU.inner() * mA - mW.adjoint() * mW;
            
            std::cerr << "alpha is :::" << alpha << std::endl;
            std::cerr << "mW is :::" << std::endl << mW << std::endl;
            std::cerr << "vtv is :::" << std::endl << vtv << std::endl;
            std::cerr << "mA is ::: " << std::endl << (mA * Matrix::Identity(mA.rows(), mA.rows())) << std::endl;
            
            // (thin) eigen-value decomposition of V^T V
            Eigen::SelfAdjointEigenSolver<Matrix> evd(vtv);
            Matrix mQ;
            RealVector mQD;
            kqp::thinEVD(evd, mQ, mQD);
            Index rank_Q = mQ.cols();             
            for(Index i = 0; i < rank_Q; i++)
                mQ.col(i) /= Eigen::internal::sqrt(mQD(i));
            std::cerr << "Rank of Q is " << rank_Q << std::endl;
            std::cerr << "mQ is :::" << std::endl << mQ << std::endl;
            std::cerr << "mQD is :::" << std::endl << mQD << std::endl;

            // --- Update
           
            Matrix mZtW(mZ.cols() + rank_Q, mW.cols());
            mZtW.topRows(mZ.cols()) = mZ.adjoint() * mW;
            mZtW.bottomRightCorner(rank_Q, mW.cols() - rank_Q).setConstant(0);
            mZtW.bottomLeftCorner(rank_Q, rank_Q) = mQD.template cast<Scalar>().asDiagonal();
            
            for(Index i = 0; i < mW.cols(); i++) {
                EvdUpdateResult<Scalar> result;
                // Update
                std::cerr << "Rank-one update with " << mZtW.col(i).adjoint() << std::endl;
                evdRankOneUpdate.update(mD, alpha, mZtW.col(i), false, selector.get(), false, result, &mZ);
                std::cerr << "mZ is now ::: " << mZ << std::endl;
                // Take the new diagonal
                mD = result.mD;
                std::cerr << "D is now ::: " << mD.adjoint() << std::endl;
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
            std::cerr << "Y is now ::: " << mY << std::endl;

                              
            // (4) Clean-up
            
            // Remove unused images
//            removeUnusedPreImages(mX, mY);

            // Ensure we have a small enough number of pre-images
            if (mX.size() > (preImagesPerRank * mD.rows())) {
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
        
        
        virtual void get_decomposition(typename FTraits::FMatrix& mX, typename FTraits::AltMatrix &mY, typename FTraits::RealVector& mD) {
            mX = this->mX;
            if (mZ.rows() > 0) {
                this->mY = this->mY * mZ;
                mZ.resize(0,0);
            }
            mY = AltMatrix<Scalar>(this->mY);
            mD = this->mD;
            
            std::cerr << "inner(X) = " << std::endl << mX.inner() << std::endl;
            std::cerr << "Y = " << std::endl << mY * Matrix::Identity(mY.rows(), mY.rows()) << std::endl;
            std::cerr << "D = " << std::endl << mD.adjoint() << std::endl;
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
        
        //! Ratio of the number of pre-images to the rank (must be >= 1)
        float preImagesPerRank;
        
    };
    
    KQP_KERNEL_EVD_INSTANCIATION(extern, IncrementalKernelEVD);
    
}

#endif
