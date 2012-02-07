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

#ifndef __KQP_REDUCED_SET_LDL_APPROACH_H__
#define __KQP_REDUCED_SET_LDL_APPROACH_H__

// LDL decomposition
#include <Eigen/Cholesky>

#include "feature_matrix.hpp"
#include "reduced_set/unused.hpp"

namespace kqp {
    
    /**
     * @brief Removes pre-images with the null space method
     * 
     * Removes pre-images using the null space method
     * 
     * @param mF the feature matrix
     * @param nullSpace the null space vectors of the gram matrix of the feature matrix
     */
    template <class FMatrix>
    void removeNullSpacePreImages(FMatrix &mF, const typename ftraits<FMatrix>::ScalarMatrix &nullSpace) {
        
    }
    
    /**
     * @brief Removes unuseful pre-images 
     *
     * 1. Removes unused pre-images 
     * 2. Computes a \f$LDL^\dagger\f$ decomposition of the Gram matrix to find redundant pre-images
     * 3. Removes newly unused pre-images
     */
    template <class FMatrix>
    void removePreImagesWithLDL(FMatrix &mF, typename ftraits<FMatrix>::ScalarMatrix &mY) {
        typedef typename ftraits<FMatrix>::Scalar Scalar;
        typedef typename ftraits<FMatrix>::ScalarMatrix ScalarMatrix;
        
        // Removes unused pre-images
        removeUnusedPreImages(mF, mY);
        
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        // LDL decomposition (stores the L^T matrix)
        Eigen::LDLT<ScalarMatrix, Eigen::Upper> ldlt(mF.inner());
        const Eigen::MatrixBase<ScalarMatrix> &mLDLT = ldlt.matrixLDLT();
        
        // Get the rank (of the null space)
        Index rank = 0;
        for(Index i = 0; i < N; i++)
            if (mLDLT(i,i) > EPSILON) rank++;
        
        std::cerr << mLDLT << std::endl;
        
        // Stop if we are full rank
        if (rank == mLDLT.rows())
            return;
        
        
        
        const Eigen::Block<const ScalarMatrix> mL1(mLDLT, 0,0,rank,rank);
        ScalarMatrix mL2 = mLDLT.block(0,rank+1,rank,N-rank);
        
        if (rank != N) {
            // Gets the null space vectors in mL2
            mL1.template triangularView<Eigen::Upper>().solveInPlace(mL2);
            mL2 = mL2 * ldlt.transpositionsP().transpose();
            
            // Simplify mL2
            removeNullSpacePreImages(mF, mL2);
            
            // Removes unused pre-images
            removeUnusedPreImages<FMatrix>(mF, mY);
        }
    }
    
    
}

#endif
