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

#include "alt_matrix.hpp"
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
        
        typedef ftraits<FMatrix> FTraits;
        typedef typename FTraits::Scalar Scalar;
        typedef typename FTraits::Real Real;
        
        AccumulatorKernelEVD() {
        }
        
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrix &mX, const typename FTraits::AltMatrix &mA) {           
            // Just add the vectors using linear combination
            FMatrix fm = mX.linear_combination(mA, Eigen::internal::sqrt(alpha));
            fMatrix.add(fm);
        }
        
        
        
        //! Actually performs the computation
        virtual void get_decomposition(FMatrix& mX, typename FTraits::AltMatrix &mY, typename FTraits::RealVector& mD) {
            const typename FMatrix::Matrix& gram = fMatrix.inner();
            Eigen::SelfAdjointEigenSolver<typename FTraits::Matrix> evd(gram.template selfadjointView<Eigen::Lower>());
            
            typename FTraits::Matrix _mY;
            kqp::thinEVD(evd, _mY, mD);
            
            mY = _mY * mD.cwiseSqrt().cwiseAbs().cwiseInverse().asDiagonal();
            mX = fMatrix;

            std::cerr << "X=" << mX << std::endl;
            std::cerr << "Y=" << mY << std::endl;
            std::cerr << "D=" << mD.adjoint() << std::endl;
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
        
        typedef ftraits<FMatrix> FTraits;
        typedef typename FTraits::Scalar Scalar;
        typedef typename FTraits::Real Real;
        typedef typename FTraits::Matrix Matrix;
        typedef typename FTraits::AltMatrix AltMatrix;
        
        AccumulatorKernelEVD() {
            offsets_X.push_back(0);
            offsets_A.push_back(0);
        }
        
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrix &mX, const typename FTraits::AltMatrix &mA) {           
            // If there is nothing to add            
            if (mA.cols() == 0)
                return;
            
            // Do a deep copy of mA
            combination_matrices.push_back(boost::shared_ptr<typename FTraits::AltMatrix>(new typename FTraits::AltMatrix(mA)));
            
            alphas.push_back(Eigen::internal::sqrt(alpha));
            fMatrix.add(mX);
            offsets_X.push_back(offsets_X.back() + mX.size());
            offsets_A.push_back(offsets_A.back() + mA.cols());
        }
        
        
        //! Actually performs the computation
        virtual void get_decomposition(FMatrix& mX, typename FTraits::AltMatrix &mY, typename FTraits::RealVector& mD) {
            // Compute A^T X^T X A^T 
            // where A = diag(A_1 ... A_n) and X = (X_1 ... X_n)
            
            Index size = offsets_A.back();
            
            Matrix gram_X = fMatrix.inner();
            Matrix gram(size, size);
            
            for(Index i = 0; i < combination_matrices.size(); i++) {
                const AltMatrix &mAi = *combination_matrices[i];
                for(Index j = 0; j <= i; j++) {
                    const AltMatrix &mAj = *combination_matrices[j];
                    getBlock(gram, offsets_A, i, j) =  (Eigen::internal::conj(alphas[i]) * alphas[j]) * (mAi.adjoint() *  getBlock(gram_X, offsets_X, i, j)) * mAj;
                }
            }
            
            
            // Direct EVD
            typename FTraits::Matrix _mY;
            Eigen::SelfAdjointEigenSolver<typename FTraits::Matrix> evd(gram.template selfadjointView<Eigen::Lower>());
            kqp::thinEVD(evd, _mY, mD);
            
            // Y <- A * Y * D^-1/2
            
            _mY = _mY * mD.cwiseSqrt().cwiseAbs().cwiseInverse().asDiagonal();
            Matrix __mY(offsets_X.back(), _mY.cols());
            
            for(Index i = 0; i < combination_matrices.size(); i++) {
                const AltMatrix &mAi = *combination_matrices[i];
                __mY.block(offsets_X[i], 0, offsets_X[i+1]-offsets_X[i], __mY.cols()) = alphas[i] * (mAi * _mY.block(offsets_A[i], 0,  offsets_A[i+1]-offsets_A[i], _mY.cols()));
            }
            
            mY.swap_dense(__mY);
            mX = fMatrix;
        }
        
    private:
        static inline Eigen::Block<Matrix> getBlock(Matrix &m, std::vector<Index> &offsets, Index i, Index j) {
            return m.block(offsets[i], offsets[j], offsets[i+1] - offsets[i], offsets[j+1]-offsets[j]);
        }
        
        //! Pre-images matrices
        FMatrix fMatrix;        
        
        //! Linear combination matrices
        std::vector<boost::shared_ptr<typename FTraits::AltMatrix> > combination_matrices;
        
        //! Offsets
        std::vector<Index> offsets_A;
        std::vector<Index> offsets_X;
        
        
        //! Alphas 
        std::vector<Scalar> alphas;
        
    };
    
    KQP_KERNEL_EVD_INSTANCIATION(extern, AccumulatorKernelEVD);
    
    
}

#endif
