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

#include <limits>

#include "evd_update.hpp"
#include "kernel_evd.hpp"
#include "null_space.hpp"
#include "coneprog.hpp"

namespace kqp {
    
    
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup OperatorBuilder
     */
    template <class FMatrix> class IncrementalKernelEVD : public OperatorBuilder<typename FMatrix::FVector> {
    public:
        typedef OperatorBuilder<typename FMatrix::FVector> Ancestor;
        typedef typename Ancestor::Matrix Matrix;
        typedef typename Ancestor::Scalar Scalar;
        typedef typename Ancestor::FVector FVector;
        typedef typename Ancestor::Real Real;
        typedef typename Ancestor::RealVector RealVector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Vector;
        
        // Indirect comparator (decreasing order)
        struct Comparator {
            const Vector &v;
            Comparator(const Vector &v) : v(v) {}
            bool operator()(Index i, Index j) {
                return this->v(i) > this->v(j);
            }
        };
        
        IncrementalKernelEVD() :preImagesPerRank(std::numeric_limits<float>::infinity()) {
            
        }
        
        IncrementalKernelEVD(double minimumRelativeError, Index maxRank, float preImagesPerRank) :
            preImagesPerRank(preImagesPerRank) {
            
        }
        
        virtual void add(Real alpha, const FVector &v) {
            // Just transform the data to add it with the generic method
            FMatrix _fMatrix;
            _fMatrix.add(v);
            Matrix m(1,1); 
            m(0,0) = alpha;
            this->add(1, _fMatrix, m);
        }
        
        
        virtual void add(Real alpha, const typename Ancestor::FMatrix &mU, const typename Ancestor::Matrix &mA) {          
            // (0) Initialisations
            if (!mX.get()) {
                mX.reset(new FMatrix());
                mY.reset(new Matrix());
                mD.reset(new RealVector());
            }            
            
            // (1) Pre-computations
            
            // Compute W = Y^T X^T
            mX->inner(mU, k);
            Matrix mW;
            mW.noalias() = mY->adjoint() * k * mA;
            
            // Compute V^T V
            Matrix vtv = mA.adjoint() * mU.inner() * mA - mW.adjoint() * mW;
            
            // (2) Choose the rank
                       
            // Eigen-value decomposition of V^T V
            Eigen::SelfAdjointEigenSolver<Matrix> evd(vtv);

            // (3) Update
            Vector z(mD->rows());
            
            Matrix mZtW(mZ.cols() + vtv.rows(), mW.cols());
            mZtW.topRows(mZ.cols()) = mZ.adjoint() * mW;
            mZtW.bottomRows(vtv.rows()) = evd.eigenvalues().asDiagonal();
            
            for(Index i = 0; i < mW.cols(); i++) {
                EvdUpdateResult<Scalar> result;
                // Update
                evdRankOneUpdate.update(*mD, alpha, z, false, selector.get(), false, result, &mZ);
                
                // Take the new diagonal
                *mD.resize(result.mD.rows());
                *mD = result.mD.diagonal();
            }
            
            
            // (4) Clean-up
            
            // Remove unused images
            removeUnusedPreImages(*mX, *mY);

            // Ensure we have a small enough number of pre-images
            if (mX->size() > (preImagesPerRank * mD->rows())) {
                if (mX->canLinearlyCombine()) {
                    // Easy case: we can linearly combine pre-images
                    mX->linearCombinationUpdate(*mY);
                    mY = 0;
                } else {
                    // Optimise
                    
                }
                
            }
        }
        
        virtual typename Ancestor::FMatrixCPtr getX() const {
            return mX;
        }
        
        virtual typename Ancestor::MatrixCPtr getY() const {
            if (mZ.rows() > 0) {
                *mY = *mY * mZ;
                mZ.resize(0,0);
            }
            return mY;
        }
        
        virtual typename Ancestor::RealVectorPtr getD() const {
            return mD;
        }
        
    private:
        typename FMatrix::Ptr mX;
        typename Ancestor::MatrixPtr mY;
        Matrix mZ;
        typename Ancestor::RealVectorPtr mD;

        // Rank-one EVD update
        FastRankOneUpdate<Scalar> evdRankOneUpdate;
        
        // Used in computation
        mutable Matrix k;

        // Rank selector
        boost::shared_ptr<Selector> selector;
 
        //! Ratio of the number of pre-images to the rank (must be >= 1)
        float preImagesPerRank;
        
    };
    
}

#endif
