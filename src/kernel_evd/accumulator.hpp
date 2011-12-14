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

#ifndef __KQP_ACCUMULATOR_BUILDER_H__
#define __KQP_ACCUMULATOR_BUILDER_H__

#include <boost/static_assert.hpp>

#include "kernel_evd.hpp"
#include "utils.hpp"

namespace kqp{
    
    /**
     * @brief Accumulation based computation of the density.
     *
     * Supposes that we can compute a linear combination of the pre-images.
     * Performs an SVD of the feature vectors (if doable) or and EVD of the 
     * inner product of feature vectors.
     * 
     * @ingroup KernelEVD
     */
    template <class FMatrix, bool can_linearly_combine = (bool)ftraits<FMatrix>::can_linearly_combine >
    class AccumulatorKernelEVD : public KernelEVD<FMatrix> {};
    
    
    template <class FMatrix> class AccumulatorKernelEVD<FMatrix, true> : public KernelEVD<FMatrix> {
    public:
        enum {
            use_linear_combination = 1
        };
        typedef typename FMatrix::FVector FVector;
        
        typedef ftraits<FMatrix> FTraits;
        typedef typename FTraits::Scalar Scalar;
        typedef typename FTraits::Real Real;
        
        AccumulatorKernelEVD() {
        }
        
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrixView &mX, const typename FTraits::Matrix &mA) {           
            // Just add the vectors using linear combination
            FMatrix fm;
            mX.linear_combination(mA, fm, Eigen::internal::sqrt(alpha));
            fMatrix.addAll(fm);
        }
        
        
        
        //! Actually performs the computation
        virtual void get_decomposition(FMatrix& mX, typename FTraits::Matrix &mY, typename FTraits::RealVector& mD) {
            const typename FMatrix::Matrix& gram = fMatrix.inner();
            
            Eigen::SelfAdjointEigenSolver<typename FTraits::Matrix> evd(gram.template selfadjointView<Eigen::Lower>());
            kqp::thinEVD(evd, mY, mD);
            
            mY = mY * mD.cwiseSqrt().cwiseInverse().asDiagonal();
            mX = fMatrix;
        }
        
    private:
        //! concatenation of pre-image matrices
        FMatrix fMatrix;        
    };
    
    // Specialisation when we know how to combine linearly
    template <class FMatrix> class AccumulatorKernelEVD<FMatrix, false> : public KernelEVD<FMatrix> {
    public:
        enum {
            use_linear_combination = 0
        };
        typedef typename FMatrix::FVector FVector;
        
        typedef ftraits<FMatrix> FTraits;
        typedef typename FTraits::Scalar Scalar;
        typedef typename FTraits::Real Real;
        typedef typename FTraits::Matrix Matrix;
        
        AccumulatorKernelEVD() {
            offsets_X.push_back(0);
            offsets_A.push_back(0);
        }
        
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrixView &mX, const typename FTraits::Matrix &mA) {           
            // Just add the vectors using linear combination
            if (mA.rows() > 0 && mA.cols() == 0) return;
            
            fMatrix.addAll(mX);
            if (mA.cols() == 0)
                combination_matrices.push_back(boost::shared_ptr<Matrix>());
            else
                combination_matrices.push_back(boost::shared_ptr<Matrix>(new Matrix(mA)));
            
            alphas.push_back(Eigen::internal::sqrt(alpha));
            
            offsets_X.push_back(offsets_X.back() + mX.size());
            offsets_A.push_back(offsets_A.back() + (is_empty(mA) ? mX.size() : mA.cols()));
        }
        
        
        //! Actually performs the computation
        virtual void get_decomposition(FMatrix& mX, typename FTraits::Matrix &mY, typename FTraits::RealVector& mD) {
            // Compute A^T X^T X A^T 
            // where A = diag(A_1 ... A_n) and X = (X_1 ... X_n)
            
            Index size = offsets_A.back();
            
            Matrix gram_X = fMatrix.inner();
            Matrix gram(size, size);
            
            for(Index i = 0; i < combination_matrices.size(); i++) {
                const Matrix *mAi = combination_matrices[i].get();
                for(Index j = 0; j <= i; j++) {
                    const Matrix *mAj = combination_matrices[i].get();
                    if (mAi && mAj)
                        getBlock(gram, offsets_A, i, j) =  (alphas[i] * alphas[j]) * mAi->adjoint() *  getBlock(gram_X, offsets_X, i, j) * *mAj;
                    else if (mAi && !mAj)
                        getBlock(gram, offsets_A, i, j) =  (alphas[i] * alphas[j]) * mAi->adjoint() * getBlock(gram_X, offsets_X, i, j);
                    else if (!mAi && mAj)
                        getBlock(gram, offsets_A, i, j) =  (alphas[i] * alphas[j]) * getBlock(gram_X, offsets_X, i, j) * *mAj;
                    else 
                        getBlock(gram, offsets_A, i, j) =  (alphas[i] * alphas[j]) * getBlock(gram_X, offsets_X, i, j);
                }
            }
            
            
            // Direct EVD
            Eigen::SelfAdjointEigenSolver<typename FTraits::Matrix> evd(gram.template selfadjointView<Eigen::Lower>());
            kqp::thinEVD(evd, mY, mD);
            
            // Y <- A * Y * D^-1/2
            
            mY = mY * mD.cwiseSqrt().cwiseInverse().asDiagonal();
            for(Index i = 0; i < combination_matrices.size(); i++) {
                if (const Matrix *mAi = combination_matrices[i].get())
                    mY.block(offsets_A[i], 0, offsets_A[i+1]-offsets_A[i], mY.cols()) = alphas[i] * *mAi * mY.block(offsets_A[i], 0,  offsets_A[i+1]-offsets_A[i], mY.cols());
                else 
                    mY.block(offsets_A[i], 0, offsets_A[i+1]-offsets_A[i], mY.cols()) = alphas[i] * mY.block(offsets_A[i], 0,  offsets_A[i+1]-offsets_A[i], mY.cols());
            }
            
            mX = fMatrix;
        }
        
    private:
        inline Eigen::Block<Matrix> getBlock(Matrix &m, std::vector<Index> &offsets, Index i, Index j) {
            return m.block(offsets[i], offsets[j], offsets[i+1] - offsets[i], offsets[j+1]-offsets[j]);
        }
                                             
        //! Pre-images matrices
        FMatrix fMatrix;        
        
        //! Linear combination matrices
        std::vector<boost::shared_ptr<Matrix> > combination_matrices;
        
        //! Offsets
        std::vector<Index> offsets_A;
        std::vector<Index> offsets_X;
        
        
        //! Alphas 
        std::vector<Scalar> alphas;
        
    };
    
    KQP_KERNEL_EVD_INSTANCIATION(extern, AccumulatorKernelEVD);
    
    
}

#endif
