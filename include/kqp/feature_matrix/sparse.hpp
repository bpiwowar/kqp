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

#ifndef __KQP_SPARE_FEATURE_MATRIX_H__
#define __KQP_SPARE_FEATURE_MATRIX_H__

#include <kqp/feature_matrix.hpp>

namespace kqp {
    //! A scalar sparse vector
    template <typename _Scalar>
    class SparseVector {
    public:
        typedef _Scalar Scalar;
    };
    
    /**
     * @brief A class that supports sparse vectors in a (high) dimensional space.
     * @ingroup FeatureMatrix
     */
    template <typename Scalar> 
    class SparseScalarMatrix : public FeatureList<SparseVector<Scalar> > {
        //! The dimension of vectors (0 if no limit)
        Index dimension;
    public:
        SparseScalarMatrix(Index dimension) {
        }
    };
    
    
} // end namespace kqp

#endif