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

#include <kqp/alt_matrix.hpp>
#include <kqp/kernel_evd.hpp>
#include <kqp/kernel_evd/utils.hpp>

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
    template <typename Scalar, bool can_linearly_combine> class AccumulatorKernelEVD : public KernelEVD<Scalar> {};
    
    // Specialisation when we know how to combine linearly
    template <typename Scalar> class AccumulatorKernelEVD<Scalar, true> : public KernelEVD<Scalar>  {
    public:
        enum {
            use_linear_combination = 1
        };
        
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        AccumulatorKernelEVD(const FSpace &fs) : KernelEVD<Scalar>(fs), fMatrix(fs.newMatrix()) {
        }
        
        virtual ~AccumulatorKernelEVD() {
        }
        
        
        virtual void _add(Real alpha, const FMatrix &mX, const ScalarAltMatrix &mA) override {           
            // Just add the vectors using linear combination
            FMatrix fm = this->getFSpace().linearCombination(mX, mA, Eigen::internal::sqrt(alpha));
            fMatrix->add(*fm);
        }
        
        void reset() override {
            *this = AccumulatorKernelEVD(this->getFSpace());
        }
        
        //! Actually performs the computation
        virtual Decomposition<Scalar> _getDecomposition() const override {
            
            const ScalarMatrix& gram = this->getFSpace().k(fMatrix);
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(gram.template selfadjointView<Eigen::Lower>());
            
            ScalarMatrix _mY;
            RealVector _mD;
            kqp::thinEVD(evd, _mY, _mD);
            
            ScalarMatrix __mY;
            __mY.noalias() = _mY * _mD.cwiseAbs().cwiseSqrt().cwiseInverse().asDiagonal();
            
            return Decomposition<Scalar>(this->getFSpace(), fMatrix, __mY, _mD, true);
        }
        
    private:
        //! concatenation of pre-image matrices
        FMatrix fMatrix;        
    };
    
    
    
    
    // Specialisation when we don't know how to combine linearly
    template <typename Scalar> class AccumulatorKernelEVD<Scalar, false> : public KernelEVD<Scalar>  {
    public:

        KQP_SCALAR_TYPEDEFS(Scalar);
        
        AccumulatorKernelEVD(const FSpace &fs) : KernelEVD<Scalar>(fs), fMatrix(fs.newMatrix()) {
            offsets_X.push_back(0);
            offsets_A.push_back(0);
        }
        
        virtual ~AccumulatorKernelEVD() {}
        
    protected:
        virtual void _add(Real alpha, const FMatrix &mX, const ScalarAltMatrix &mA) override {           
            // If there is nothing to add            
            if (mA.cols() == 0)
                return;
            
            // Do a deep copy of mA
            combination_matrices.push_back(mA);
            
            alphas.push_back(Eigen::internal::sqrt(alpha));
            fMatrix.add(mX);
            offsets_X.push_back(offsets_X.back() + mX.size());
            offsets_A.push_back(offsets_A.back() + mA.cols());
        }
        
        void reset() {
            *this = AccumulatorKernelEVD(this->getFSpace());
        }

        
    protected:
        //! Actually performs the computation
        virtual Decomposition<Scalar> _getDecomposition() const override {
            Decomposition<Scalar> d;
            // Compute A^T X^T X A^T 
            // where A = diag(A_1 ... A_n) and X = (X_1 ... X_n)
            
            Index size = offsets_A.back();
            
            // Nothing to do
            if (size == 0) {
                d.mX = this->getFSpace().newMatrix();
                d.mY.resize(0,0);
                d.mD.resize(0,1);
                return d;
            }
            
            ScalarMatrix gram_X = this->getFSpace().k(fMatrix);
            ScalarMatrix gram(size, size);
            
            for(size_t i = 0; i < combination_matrices.size(); i++) {
                const ScalarAltMatrix &mAi = combination_matrices[i];
                for(size_t j = 0; j <= i; j++) {
                    const ScalarAltMatrix &mAj = combination_matrices[j];
                    getBlock(gram, offsets_A, i, j) 
                            =  (mAi.transpose() *  ((Eigen::internal::conj(alphas[i]) * alphas[j]) * getBlock(gram_X, offsets_X, i, j))) * mAj;
                }
            }
            
            
            // Direct EVD
            ScalarMatrix _mY;
            RealVector _mD;
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(gram.template selfadjointView<Eigen::Lower>());
            kqp::thinEVD(evd, _mY, _mD);
            d.mD.swap(_mD);
            
            // Y <- A * Y * D^-1/2
            
            ScalarMatrix _mY2 = _mY * d.mD.cwiseSqrt().cwiseAbs().cwiseInverse().asDiagonal();
            _mY = _mY2;
            ScalarMatrix __mY(offsets_X.back(), _mY.cols());
            
            for(size_t i = 0; i < combination_matrices.size(); i++) {
                const ScalarAltMatrix &mAi = combination_matrices[i];
                __mY.block(offsets_X[i], 0, offsets_X[i+1]-offsets_X[i], __mY.cols()) = mAi * (alphas[i] * _mY.block(offsets_A[i], 0,  offsets_A[i+1]-offsets_A[i], _mY.cols()));
            }
            
            d.mY.swap(__mY);
            d.mX = fMatrix;
            return d;
        }
        
    private:
        static inline Eigen::Block<ScalarMatrix> getBlock(ScalarMatrix &m, const std::vector<Index> &offsets, size_t i, size_t j) {
            return m.block(offsets[i], offsets[j], offsets[i+1] - offsets[i], offsets[j+1]-offsets[j]);
        }
        
        //! Pre-images matrices
        FMatrix fMatrix;        
        
        //! Linear combination matrices
        std::vector<ScalarAltMatrix> combination_matrices;
        
        //! Offsets
        std::vector<Index> offsets_A;
        std::vector<Index> offsets_X;
        
        
        //! Alphas 
        std::vector<Scalar> alphas;
        
    };
    
}

#ifndef SWIG
#define KQP_SCALAR_GEN(type) extern template class kqp::AccumulatorKernelEVD<type, true>; extern template class kqp::AccumulatorKernelEVD<type, false>;
#include <kqp/for_all_scalar_gen>
#endif

#endif
