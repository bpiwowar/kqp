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
    template <class FMatrix>
    class AccumulatorKernelEVD : public KernelEVD<FMatrix> {   
    public:
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
    
    
    KQP_KERNEL_EVD_INSTANCIATION(extern, AccumulatorKernelEVD);
    
    
}

#endif
