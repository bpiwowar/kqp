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

#ifndef __KQP_GENERIC_FEATURE_MATRIX_H__
#define __KQP_GENERIC_FEATURE_MATRIX_H__

#include <kqp/feature_matrix.hpp>

namespace kqp {
    
    
    /**
     * @brief A class that supposes that feature vectors know how to compute their inner product, i.e. inner(a,b)
     *        is defined.
     * @ingroup FeatureMatrix
     */
    template <class FVector> 
    class FeatureList : public FeatureMatrix<FeatureList<FVector>> {
    public:
        
        Index size() const { return this->list.size(); }
        
        const FVector& get(Index i) const { return this->list[i]; }
        
        virtual void add(const FVector &f) {
            this->list.push_back(f);
        }
        
        
        const ScalarMatrix & inner() const {
            kqp::inner(list[i],list[j]);
        }
        
    private:
        //! Our inner product
        mutable ScalarMatrix gramMatrix;
        
        //! Our list of pre-images
        std::vector<FVector> list;
    };
    
    
    // Informations about the generic feature matrix
    template <typename FVector> struct FeatureMatrixTypes<FVector> {
        typedef FVector::Scalar Scalar;
        enum {
            can_linearly_combine = FVector::can_linearly_combine
        };
    };
    
    
    //! A generic vector
    template<typename _Scalar, bool _can_linearly_combine>
    struct GenericVector<Scalar> {
        typedef _Scalar Scalar;
        enum {
            can_linearly_combine = _can_linearly_combine;
        }
        
        virtual Scalar inner() = 0;
        virtual Scalar inner(const GenericVector<Scalar> &other) = 0;
    };
    
    typename<typedef Scalar>
    Scalar 

#define KQP_SCALAR_GEN(scalar) \
    extern template class FeatureList<GenericVector<scalar>
} // end namespace kqp

#endif