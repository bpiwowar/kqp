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

#include "kernel_evd.hpp"

namespace kqp {
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup OperatorBuilder
     */
    template <class FMatrix> class IncrementalBuilder : public OperatorBuilder<FMatrix> {
    public:
        typedef typename OperatorBuilder<FMatrix>::Scalar Scalar;
        typedef typename OperatorBuilder<FMatrix>::FVector FVector;
        
        virtual void add(double alpha, const FVector &v) {
            
        }
    };
}

#endif
