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

#include "kernel_evd.hpp"

namespace kqp{
    /**
     * @brief Accumulation based computation of the density.
     *
     * Supposes that we can compute a linear combination of the pre-images.
     * Performs an SVD of the feature vectors (if doable) or and EVD of the 
     * inner product of feature vectors.
     * 
     * @ingroup OperatorBuilder
     */
    template <class FMatrix>
    class AccumulatorBuilder : public OperatorBuilder<typename FMatrix::FVector> {   
    public:
        typedef OperatorBuilder<typename FMatrix::FVector> Ancestor;
        
        typedef typename FMatrix::Vector FVector;
        typedef typename OperatorBuilder<FVector>::Matrix Matrix;
        typedef typename OperatorBuilder<FVector>::MatrixCPtr MatrixCPtr;
        typedef boost::shared_ptr<const FMatrix> FMatrixCPtr;
        
        AccumulatorBuilder() {
        }
        
        
        virtual void add(const typename Ancestor::FMatrix &_fMatrix, const typename Ancestor::Matrix &coefficients) {
            // Just add            
            for(Index j = 0; j < coefficients.cols(); j++) 
                fMatrix.add(_fMatrix.linearCombination(coefficients.col(j)));
        }
        
        //! Actually performs the computation
        void compute() {
        }
        
    private:
        
        //! concatenation of pre-image matrices
        FMatrix fMatrix;
    };
}

#endif
