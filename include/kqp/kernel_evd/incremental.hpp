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

#   include <kqp/define_header_logger.hpp>
    DEFINE_KQP_HLOGGER("kqp.kernel-evd.incremental");
    

    
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup KernelEVD
     */
    template <typename Scalar> class IncrementalKernelEVD : public KernelEVD<Scalar> , public Cleaner<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        using KernelEVD<Scalar>::getFSpace;
        
        // Indirect comparator (decreasing order)
        struct Comparator {
            const RealVector &v;
            Comparator(const RealVector &v) : v(v) {}
            bool operator()(Index i, Index j) {
                return this->v(i) > this->v(j);
            }
        };
        
        IncrementalKernelEVD(const FSpace &fs) : KernelEVD<Scalar>(fs), mX(fs.newMatrix()) {}
        virtual ~IncrementalKernelEVD() {}
        
        void reset() {
            *this = IncrementalKernelEVD(this->getFSpace());
        }
               
        virtual void _add(Real alpha, const FMatrix &mU, const ScalarAltMatrix &mA) override {
            // --- Info
            
//            KQP_LOG_DEBUG_F(KQP_HLOGGER, "Dimensions: X [%d], Y [%dx%d], Z [%dx%d], D [%d], U [%d], A [%dx%d]", 
//                           %mX.size() %mY.rows() %mY.cols() %mZ.rows() %mZ.cols() %mD.rows() %mU.size() %mA.rows() %mA.cols());
            
            // --- Pre-computations
            
            // Compute W = Y^T X^T
            
            ScalarMatrix mW =  getFSpace().k(mX, mY, mU, mA);
            
            // Compute V^T V
            ScalarMatrix vtv = getFSpace().k(mU, mA);
            vtv -= mW.adjoint() * mW;
            
            
            // (thin) eigen-value decomposition of V^T V
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(vtv);
            ScalarMatrix mQ;
            ScalarMatrix mQ0;
            RealVector mDQ;
            kqp::thinEVD(evd, mQ, mDQ, &mQ0);

            Index rank_Q = mQ.cols();             
            mDQ = mDQ.cwiseAbs().cwiseSqrt();
            for(Index i = 0; i < rank_Q; i++)
                mQ.col(i) /= mDQ(i);

            // --- Update
           
            // m is the matrix [WQD^1/2 WQ_0; D^1/2 0 ] 
            ScalarMatrix m(mW.rows() + rank_Q, mW.cols());
                        
            m.topLeftCorner(mW.rows(), rank_Q) =  mW * mQ * mDQ.asDiagonal();
            m.topRightCorner(mW.rows(), mW.cols() - rank_Q) = mW * mQ0;                
            
            m.bottomLeftCorner(rank_Q, rank_Q) = mDQ.template cast<Scalar>().asDiagonal();
            m.bottomRightCorner(rank_Q, mW.cols() - rank_Q).setConstant(0) ;
            

            for(Index i = 0; i < m.cols(); i++) {
                EvdUpdateResult<Scalar> result;
                // Update
                ScalarVector v(m.rows());
                v.head(mZ.cols()) = mZ.adjoint() * m.col(i).head(mZ.cols());
                v.tail(m.rows() - mZ.cols()) = m.col(i).tail(m.rows() - mZ.cols());
                
                
                // Rank-1 update:
                // For better accuracy, we don't decrease the rank yet using the selector,
                // but this might be an option in the future (so as to improve speed)
                evdRankOneUpdate.update(mD, alpha, v, false, 0, false, result, &mZ);
                
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
            


            // --- Rank selection   
            DecompositionList<Real> list(mD);
            bool identityZ = false;
            if (this->selector) {
                // Selects the eigenvalues
                this->selector->selection(list);
                
                // Remove corresponding entries
                select_rows(list.getSelected(), mD, mD);                
                mY = mY * mZ;
                select_columns(list.getSelected(), mY, mY);

                identityZ = true;
            }
            
            
            // First, tries to remove unused pre-images images
            RemoveUnusedPreImages<Scalar>::run(mX, mY);
            
            // --- Ensure we have a small enough number of pre-images
            Index maxRank = this->preImageRatios.second * (float)mD.rows();
            if (mX.size() > maxRank) {
                
                // Get rid of Z
                if (!identityZ) mY = mY * mZ;
                identityZ = true;

                // Try again to remove unused pre-images
                RemoveUnusedPreImages<Scalar>::run(mX, mY);
                KQP_LOG_DEBUG_F(KQP_HLOGGER, "Rank after unused pre-images algorithm: %d [%d]", %mY.rows() %maxRank);

                // Try to remove null space pre-images
                ReducedSetNullSpace<Scalar>::run(getFSpace(), mX, mY);
                KQP_LOG_DEBUG_F(KQP_HLOGGER, "Rank after null space algorithm: %d [%d]", %mY.rows() %maxRank);

                if (mX.size() > maxRank) {
                    if (getFSpace().canLinearlyCombine() && this->useLinearCombination) {
                        // Easy case: we can linearly combine pre-images
                        mX = getFSpace().linearCombination(mX, mY);
                        mY = ScalarMatrix::Identity(mX.size(), mX.size());
                    } else {
                        // Use QP approach
                        ReducedSetWithQP<Scalar> qp_rs;
                        qp_rs.run(this->preImageRatios.first * (float)mD.rows(), getFSpace(), mX, mY, mD);
                        
                        // Get the decomposition
                        mX = qp_rs.getFeatureMatrix();
                        mY = qp_rs.getMixtureMatrix();
                        mD = qp_rs.getEigenValues();
                        
                        // The decomposition is not orthonormal anymore
                        Orthonormalize<Scalar>::run(getFSpace(), mX, mY, mD);
                    }
                }
            
            }
            
            if (identityZ) 
                mZ = ScalarMatrix::Identity(mY.cols(), mY.cols());
            
        }
        
        // Gets the decomposition
        virtual Decomposition<Scalar> _getDecomposition() const override {
            Decomposition<Scalar> d(this->getFSpace());
            
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
        
    };

#ifndef SWIG    
#define KQP_SCALAR_GEN(type) extern template class kqp::IncrementalKernelEVD<type>;
#include <kqp/for_all_scalar_gen>
#endif

}

#endif
