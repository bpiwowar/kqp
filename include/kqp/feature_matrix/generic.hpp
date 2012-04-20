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
    class FeatureList : public FeatureMatrix< FeatureList<FVector> > {
    public:
        KQP_FMATRIX_COMMON_DEFS(FeatureMatrix<FVector>);

        FeatureList(Index dimension) : m_dimension(dimension) {}
        
        
                
    protected:
        Index _size() const { return this->list.size(); }

        void _add(const Self &other, std::vector<bool> *which = NULL)  {
            this->list.push_back(f);
        }
        
        const ScalarMatrix &_inner() const {
        }
        
        template<class DerivedMatrix>
        void _inner(const Self &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            if (result.rows() != _size() || result.cols() != other.size())
                result.derived().resize(result.rows(), result.cols());
        }
        
        // Computes alpha * X * A + beta * Y * B (X = *this)
        Self _linear_combination(const ScalarAltMatrix &mA, Scalar alpha, const Self *mY, const ScalarAltMatrix *mB, Scalar beta) const {
        }
        
        void _subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Self &into) const {
        }
        
    private:
        //! Our inner product
        mutable ScalarMatrix m_gramMatrix;
        
        //! Our list of pre-images
        std::vector<FVector> m_list;
        
        //! Dimension
        Index m_dimension;
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

	template<Scalar>
	inline Scalar inner(const GenericVector<Scalar> &a, const GenericVector<Scalar> &b) {
		return a.inner(b);
	}

	template<Scalar>
	inline Scalar inner(GenericVector<Scalar> &a) {
		return a.inner();
	}
    
} // end namespace kqp

#endif