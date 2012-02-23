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

#ifndef __KQP_REDUCED_SET_UNUSED_H__
#define __KQP_REDUCED_SET_UNUSED_H__

#include <kqp/feature_matrix.hpp>
#include <kqp/subset.hpp>

namespace kqp {
    /**
     * @brief Removes unused pre-images
     */
    // TODO make it work for EigenBase
    template <class FMatrix> 
    void removeUnusedPreImages(FMatrix &mF, typename ftraits<FMatrix>::ScalarMatrix &mY) {
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        std::vector<bool> to_keep(N, true);
        
        // Removes unused pre-images
        for(Index i = 0; i < N; i++) 
            if (mY.row(i).norm() < EPSILON) 
                to_keep[i] = false;
        
        select_rows(to_keep.begin(), to_keep.end(), mY, mY);
        mF.subset(to_keep.begin(), to_keep.end());
    }
    
    template <class FMatrix> 
    void removeUnusedPreImages(FMatrix &mF, typename ftraits<FMatrix>::ScalarAltMatrix &mY) {
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        
        switch (mY.type()) {
            case IDENTITY:
                // Nothing to do
                return;
                
            case DIAGONAL:
            case DENSE:
                break;
                
            default:
                KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
        }
        
        // Removes unused pre-images
        typename ftraits<FMatrix>::ScalarMatrix m = mY.dense_matrix();
        
        std::vector<bool> to_keep(N, true);
        for(Index i = 0; i < N; i++) 
            if (m.row(i).squaredNorm() < EPSILON) 
                to_keep[i] = false;
        
        select_rows(to_keep.begin(), to_keep.end(), mY, mY);
                                     
        mF.subset(to_keep.begin(), to_keep.end());
    }
    
}

#endif
