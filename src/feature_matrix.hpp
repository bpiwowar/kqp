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

#ifndef __KQP_FEATURE_MATRIX_H__
#define __KQP_FEATURE_MATRIX_H__


#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#include "Eigen/Core"

#include "intrusive_ptr_object.hpp"

#include "kqp.hpp"
#include "coneprog.hpp"

namespace kqp {
    
    /** By default, vectors cannot be combined */
    template<typename FVector> 
    struct linear_combination { 
        //! Scalar
        typedef typename FVector::Scalar Scalar;
        
        //! Cannot combine by default
        static const bool canCombine = false; 
        
        /**
         * Computes x <- x + b * y
         */
        static void axpy(const FVector& x, const Scalar &b, const FVector &y) {
            BOOST_THROW_EXCEPTION(illegal_operation_exception());
        };
    };
    
    
    /**
     * Defines a generic inner product that has to be implemented
     * as a last resort
     */
    template <class FVector>
    struct Inner {
        static typename FVector::Scalar compute(const FVector &a, const FVector &b) {
            BOOST_THROW_EXCEPTION(not_implemented_exception());
        }
    };    
    
    /**
     * @brief Base for all feature matrix classes
     * 
     * This class holds a list of vectors whose exact representation might not be
     * known. All sub-classes must implement basic list operations (add, remove).
     * Vectors added to the feature matrix are considered as immutable by default
     * since they might be kept as is.
     * 
     * @ingroup FeatureMatrix
     * @param _FVector the type of the feature vectors
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class _FVector> 
    class FeatureMatrix : public boost::intrusive_ptr_base, public boost::intrusive_ptr_object {
    public:
        
        //! Own type
        typedef FeatureMatrix<_FVector> Self;
        
        //! Pointers
        typedef boost::intrusive_ptr<Self> Ptr;
        typedef boost::intrusive_ptr<const Self> CPtr;
        
        //! Feature vector type
        typedef _FVector FVector;
        
        //! Scalar type
        typedef typename FVector::Scalar Scalar;
        
        //! Matrix type for inner products
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> InnerMatrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        
        /**
         * @brief Computes the inner product between the i<sup>th</sup>
         * vector of the list and the given vector.
         *
         * By default, uses the inner product between any two vectors of this type 
         * 
         * @param i
         *            The index of the vector in the list
         * @param vector
         *            The vector provided
         * @return The inner product between both vectors
         */
        virtual Scalar inner(Index i, const FVector& vector) const {
            return Inner<FVector>::compute(get(i), vector);
        }
        
        /**
         * Compute the inner products with each of the vectors. This method might be
         * re-implemented for efficiency reason, and by default calls iteratively
         * {@linkplain #computeInnerProduct(int, Object)}.
         * 
         * @param vector
         *            The vector with which the inner product is computed each time
         * @return A series of inner products, one for each of the base vectors
         */
        virtual Vector inner(const FVector& vector) const {
            typename FeatureMatrix::Vector innerProducts(size());
            for (int i = size(); --i >= 0;)
                innerProducts[i] = Inner<FVector>::compute(get(i), vector);
            return innerProducts;
        }
        
        
        
        /**
         * @brief Compute the inner product with another feature matrix.
         *
         * @warning m will be modified (Eigen const trick)
         */
        template <typename DerivedMatrix>
        void inner(const FeatureMatrix<FVector>& other, const Eigen::MatrixBase<DerivedMatrix> &m) const {
            // Eigen-const-cast-trick
            Eigen::MatrixBase<DerivedMatrix> &r = const_cast< Eigen::MatrixBase<DerivedMatrix> &>(m);
            
            // Resize if this is possible
            r.derived().resize(size(), other.size());
            
            for (int i = 0; i < size(); i++)
                for (int j = 0; j < other.size(); j++)
                    r(i, j) = Inner<FVector>::compute(get(i), other.get(j));
            
        }
        
        
        /**
         * @brief Computes the Gram matrix of this feature matrix
         * @param full if the matrix has to be entirely computed
         * @return A dense symmetric matrix (use only lower part)
         */
        virtual boost::shared_ptr<const InnerMatrix> inner(bool full) const {
            // We lose space here, could be used otherwise???
            boost::shared_ptr<InnerMatrix> m(new InnerMatrix(size(), size()));
            
            for (Index i = size(); --i >= 0;)
                for (Index j = 0; j <= i; j++) {
                    Scalar x = Inner<FVector>::compute(get(i), get(j));
                    (*m)(i,j) = x;
                    if (full && i != j) (*m)(j,i) = Eigen::internal::conj(x);
                }
            return m;
        }
        
        /**
         * Our combiner
         */
        typedef linear_combination<FVector> Combiner;
        
        /**
         * Returns true if the vectors can be linearly combined
         */
        virtual bool canLinearlyCombine() {
            return Combiner::canCombine;
        }
        
        /** 
         * @brief Linear combination of vectors
         *
         * In this implementation, we just use the pairwise combiner if it exists
         */
        virtual FVector linearCombination(const Vector & lambdas) {
            if (lambdas.size() != this->size()) 
                BOOST_THROW_EXCEPTION(illegal_argument_exception());
            
            // A null vector
            FVector result;
            
            for(Index i = 0; i < lambdas.rows(); i++) 
                Combiner::axpy(result, lambdas[i], this->get(i));
            
            return result;
        }
        
        /** Get the number of feature vectors */
        virtual Index size() const = 0;
        
        /** Get the i<sup>th</sup> feature vector */
        virtual FVector get(Index i) const = 0;
        
        /** Get the i<sup>th</sup> feature vector */
        virtual void add(const FVector &f) = 0;
        
        /** Get the i<sup>th</sup> feature vector */
        virtual void set(Index i, const FVector &f) = 0;
        
        /** 
         * Remove the i<sup>th</sup> feature vector 
         * @param if swap is true, then the last vector will be swapped with one to remove (faster)
         */
        virtual void remove(Index i, bool swap = false) = 0;
    };
    
    
    
}

#endif
