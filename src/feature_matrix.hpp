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


#include "kqp.hpp"

namespace kqp {

    // Forward declaration of feature matrix traits
    template<class T> struct ftraits;
    
    template<class Derived>
    class FeatureMatrixView {
    public:
        typedef ftraits<Derived> FTraits;
        typedef typename FTraits::FMatrix FMatrix;
        typedef typename FTraits::FVector FVector;
        typedef typename FTraits::Scalar Scalar;

        virtual ~FeatureMatrixView() {}
        
        /** 
         * @brief Linear combination of the feature vectors.
         * 
         * Computes \f$ XA \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
         */
        inline void linear_combination(const typename FTraits::Matrix & mA, Derived &result, Scalar alpha = (Scalar)1) const {
            if (const FMatrix * _this = dynamic_cast<const FMatrix *>(this)) {
                _this->_linear_combination(alpha, mA, result);
            } else if (const FVector * _this = dynamic_cast<const FVector *>(this)) {
                _this->_linear_combination(alpha, mA, result);
            } else 
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot handle a type which is neither feature-vector nor feature-matrix", % KQP_DEMANGLE(*this));
        }
        
        /** Get a const reference to the i<sup>th</sup> feature vector */
        virtual const typename FTraits::FVector get(Index i) const = 0;

        /** Get the number of feature vectors */
        virtual Index size() const = 0;
        
        
        /**
         * @brief Computes the Gram matrix of this feature matrix
         * @return A dense self-adjoint matrix
         */
        virtual const typename FTraits::Matrix & inner() const = 0;
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
    template <class _Derived> 
    class FeatureMatrix : public FeatureMatrixView<_Derived> {
    public:       

        typedef _Derived Derived;
        typedef ftraits<Derived> FTraits;
        typedef typename FTraits::FMatrix FMatrix;
        typedef typename FTraits::FVector FVector;
        

        /**
         * Returns true if the vectors can be linearly combined
         */
        bool can_linearly_combine() {
            return static_cast<Derived*>(this)->Derived::can_linearly_combine();
        }

        /** Add a feature vector */
        virtual void add(const typename FTraits::FVector &f) = 0;

        virtual void add(const Derived &f) {
            for(Index i = 0; i < f.size(); i++)
                this->add(f.get(i));
        }
        
        /** Add a list of feature vectors from a feature matrix */
        virtual void addAll(const FeatureMatrixView<Derived> &f) {
            if (const FMatrix * _f = dynamic_cast<const FMatrix *>(&f)) {
                this->add(*_f);
            } else if (const FVector * _f = dynamic_cast<const FVector *>(&f)) {
                this->add(*_f);
            } else 
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot handle a type which is neither feature-vector nor feature-matrix", % KQP_DEMANGLE(f));
        }
        
        
        inline Derived &derived() {
            return static_cast<Derived&>(*this);
        }
        
        inline const Derived &derived() const {
            return static_cast<const Derived&>(*this);
        }

        

        /** Get the i<sup>th</sup> feature vector */
        inline void set(Index i, const Derived &f) {
            static_cast<Derived*>(this)->Derived::set(i, f);
        }
        
        /** 
         * Remove the i<sup>th</sup> feature vector 
         * @param if swap is true, then the last vector will be swapped with one to remove (faster)
         */
        inline void remove(Index i, bool swap = false) {
            static_cast<Derived*>(this)->Derived::remove(i, swap);
        }
        
    };
    
    
    //! Compute an inner product of two feature matrices
    template<typename Derived, class DerivedMatrix>
    void inner(const typename FeatureMatrix<Derived>::Derived &mA, const typename FeatureMatrix<Derived>::Derived &mB, const typename Eigen::MatrixBase<DerivedMatrix> &result) {
        static_cast<const Derived&>(mA).inner<DerivedMatrix>(static_cast<const Derived&>(mB), const_cast<typename Eigen::MatrixBase<DerivedMatrix>&>(result));
    }

    //! Compute an inner product one inner matrix and one feature vector
    template<typename Derived, class DerivedMatrix>
    void inner(const typename FeatureMatrix<Derived>::Derived &mA, const typename FeatureMatrix<Derived>::FVector &x, const typename Eigen::MatrixBase<DerivedMatrix> &result) {
        mA.inner(x, result);
    }

    //! Compute an inner product between a feature vector and a feature matrix
    template<typename Derived, class DerivedMatrix>
    void inner(const typename FeatureMatrix<Derived>::FVector &x, const typename FeatureMatrix<Derived>::Derived &mA, const typename Eigen::MatrixBase<DerivedMatrix> &result) {
        mA.inner(x, result);
        const_cast<typename Eigen::MatrixBase<DerivedMatrix> &>(result).adjointInPlace();
    }
    
    //! Inner product of two feature vectors
    template<typename Derived>
    typename ftraits<Derived>::Scalar
    inner(const typename FeatureMatrix<Derived>::FVector &x, const typename FeatureMatrix<Derived>::FVector &y) {
        typedef typename FeatureMatrix<Derived>::FVector FVector;
        return static_cast<FVector&>(x).inner(static_cast<const FVector&>(y));
    }
    
    //! inner product of two views: use dynamic typing to determine which is our case
    template<typename Derived, class DerivedMatrix>
    void inner_views(const FeatureMatrixView<Derived> &mA, const FeatureMatrixView<Derived> &mB, const typename Eigen::MatrixBase<DerivedMatrix> &result) {    
        typedef ftraits<Derived> FTraits;
        typedef typename FTraits::FMatrix FMatrix;
        typedef typename FTraits::FVector FVector;
        typedef typename FTraits::Scalar Scalar;

        if (const FMatrix * _mA = dynamic_cast<const FMatrix *>(&mA)) {
            if (const FMatrix * _mB = dynamic_cast<const FMatrix *>(&mB)) {
                inner<Derived, DerivedMatrix>(_mA->derived(), _mB->derived(), result);
                return;
            } 
            
            if (const FVector * _mB = dynamic_cast<const FVector *>(&mB)) {
                inner<Derived, DerivedMatrix>(*_mA, *_mB, result);                
                return;
            }
            
        } 
        
        if (const FVector * _mA = dynamic_cast<const FVector *>(&mA)) {
            if (const FMatrix * _mB = dynamic_cast<const FMatrix *>(&mB)) {
                inner<Derived, DerivedMatrix>(*_mA, *_mB, result);
                return;
            } 
            
            if (const FVector * _mB = dynamic_cast<const FVector *>(&mB)) {
                // both vectors: returns a scalar
                typename Eigen::MatrixBase<DerivedMatrix> & _result = const_cast<typename Eigen::MatrixBase<DerivedMatrix> &>(result);
                Scalar z = _mA->inner<DerivedMatrix>(*_mB);
                _result.derived().resize(1,1);
                _result(0,0) = z;
                return;
            }
        }
        
        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot do an inner product between %s and %s", % KQP_DEMANGLE(mA) % KQP_DEMANGLE(mB));

    }
    
    //! Type informatino for feature matrices (feeds the ftraits structure)
    template <class _FMatrix> struct FeatureMatrixTypes {
    };
    
    /**
     * Feature Vector traits
     */
    template <class _FMatrix>
    struct ftraits {
        //! Ourselves
        typedef _FMatrix FMatrix;
        
        //! Definitions
        enum {
            can_linearly_combine = FeatureMatrixTypes<FMatrix>::can_linearly_combine  
        };
        
        //! Scalar value
        typedef typename FeatureMatrixTypes<FMatrix>::Scalar Scalar;
        
        //! Vector view on this type
        typedef typename FeatureMatrixTypes<FMatrix>::FVector FVector;

        //! View on this type
        typedef FeatureMatrixView<FMatrix> FMatrixView;
                
        //! Vector of scalars
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ScalarVector;
        
        //! Real value associated to the scalar one
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        //! Vector with reals
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RealVector;
        
        /** Inner product matrix.
         * @deprecated
         */
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        
        //! Inner product matrix
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
    }; 
    
}

#endif
