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

#ifndef __KQP__H__
#define __KQP__H__

#include "feature_matrix.hpp"

namespace kqp {
    
    /**
     * @brief A class that supposes that feature vectors know how to compute their inner product, i.e. inner(a,b)
     *        is defined.
     * @ingroup FeatureMatrix
     */
    template <class FVector> 
    class FeatureList : public FeatureMatrix<FVector> {
    public:
        
        Index size() const { return this->list.size(); }
        
        const FVector& get(Index i) const { return this->list[i]; }
        
        virtual void add(const FVector &f) {
            this->list.push_back(f);
        }
        
    private:
        std::vector<FVector> list;
    };
    
} // end namespace kqp

#endif