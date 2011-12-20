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
#include "alt_matrix.hpp"

namespace kqp {
    template <class Derived> struct ftraits;
    
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
    class FeatureMatrix {
    public:       

        typedef _Derived Derived;
        typedef ftraits<Derived> FTraits;
        typedef typename FTraits::FMatrix FMatrix;
        typedef typename FTraits::Scalar Scalar;
        

        /**
         * Returns true if the vectors can be linearly combined
         */
        bool can_linearly_combine() {
            return static_cast<Derived*>(this)->Derived::can_linearly_combine();
        }

        /** Add all vectors */
        virtual void add(const Derived &f) {
            for(Index i = 0; i < f.size(); i++)
                this->add(f.view(i));
        }
        
        inline Derived &derived() {
            return static_cast<Derived&>(*this);
        }
        
        inline const Derived &derived() const {
            return static_cast<const Derived&>(*this);
        }

        /**
         * Set the feature matrix to another one
         */
        inline void set(Index i, const Derived &f) {
            this->view(i)._set(f);
        }
        
        //! View on the i<sup>th</sup> feature vector
        const Derived view(Index i) const { 
            return this->view(i,1);
        }

        //! View on the i<sup>th</sup> feature vector
        const Derived view(Index i) { 
            return this->view(i,1);
        }

        /** Get a view of a range of pre-images */
        inline const Derived view(Index start, Index size) const {
            if (start + size >= this->size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot get the %d-%d vector range among %d vectors", %(start+1) % (start+size+1) % this->size());
            
            return static_cast<const Derived*>(this)->Derived::view(start, size);
        }
        
        /** Get a view of a range of pre-images */
        inline Derived view(Index start, Index size) {
            if (start + size >= this->size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot get the %d-%d vector range among %d vectors", %(start+1) % (start+size+1) % this->size());
            
            return static_cast<Derived*>(this)->Derived::view(start, size);
        }
        
        /** Get the i<sup>th</sup> feature vector */
        inline void set(const Derived &f) {
            if (f.size() != this->size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Can only assign feature matrices of same size (%d vs %d)", %this->size() %f.size());
            static_cast<Derived*>(this)->Derived::_set(f);
        }
        
        
        /** 
         * @brief Linear combination of the feature vectors.
         * 
         * Computes \f$ XA \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
         */
        inline Derived linear_combination(const kqp::AltMatrix<Scalar> &mA, Scalar alpha = (Scalar)1) const {
            // Check for correctedness
            if (mA.rows() != this->size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot linearly combine with a matrix with %d rows (we have %d pre-images)", %mA.rows() %this->size());

            // If we have no columns in A, then return an empty feature matrix
            if (mA.cols() == 0)
                return Derived();
                    
            // Call the derived
            return static_cast<const Derived*>(this)->Derived::_linear_combination(mA, alpha);
        }
                

        /** Get the number of feature vectors */
        inline Index size() const {
            return static_cast<const Derived*>(this)->Derived::size();            
        }
        
        
        /**
         * @brief Computes the Gram matrix of this feature matrix
         * @return A dense self-adjoint matrix
         */
        const typename FTraits::Matrix & inner() const {
            return static_cast<Derived*>(this)->Derived::inner();
        }
        
        /** 
         * Remove the i<sup>th</sup> feature vector 
         * @param if swap is true, then the last vector will be swapped with one to remove (faster)
         */
        inline void remove(Index i, bool swap = false) {
            static_cast<Derived*>(this)->Derived::remove(i, swap);
        }
        
    };
    
    template<class Derived>
    std::ostream& operator<<(std::ostream &out, const FeatureMatrix<Derived> &f) {
        return out << "[" << KQP_DEMANGLE(f) << "]";
    }
    
    //! Compute an inner product of two feature matrices
    template<typename Derived, class DerivedMatrix>
    void inner(const FeatureMatrix<Derived> &mA, const FeatureMatrix<Derived> &mB, const typename Eigen::MatrixBase<DerivedMatrix> &result) {
        typename Eigen::MatrixBase<DerivedMatrix>& _result = const_cast<typename Eigen::MatrixBase<DerivedMatrix>&>(result);
        if (mA.size() == 0 || mB.size() == 0) 
            // No need to compute anything - we just resize for consistency
            _result.derived().resize(mA.size(), mB.size());
        else
            static_cast<const Derived&>(mA).inner<DerivedMatrix>(static_cast<const Derived&>(mB), _result);
    }

    
    //! Type informatino for feature matrices (feeds the ftraits structure)
    template <class _FMatrix> 
    struct FeatureMatrixTypes {};
    
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

        //! Matrix used for linear combinations
        typedef kqp::AltMatrix<Scalar> AltMatrix;
    }; 
    
}

#endif
