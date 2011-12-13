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

#ifndef __KQP_DIVIDE_AND_CONQUER_BUILDER_H__
#define __KQP_DIVIDE_AND_CONQUER_BUILDER_H__

#include "kernel_evd.hpp"

namespace kqp {
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup KernelEVD
     */
    template <class FMatrix> class DivideAndConquerBuilder : public KernelEVD<FMatrix> {
    public:
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrixView &mX, const typename FTraits::Matrix &mA) {
        }
    };
}

#endif

