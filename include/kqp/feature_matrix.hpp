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


#include <kqp/kqp.hpp>
#include <kqp/alt_matrix.hpp>

namespace kqp {
    //! Traits for feature matrices
    template <class Derived> struct ftraits;
    
    /**
     * @brief Base for all feature matrix classes
     * 
     * This class holds a list of vectors whose exact representation might not be
     * known. All sub-classes must implement basic list operations (add, remove).
     *
     * Vectors added to the feature matrix are considered as immutable by default.
     *
     * Copy constructor are not assumed to perform a deep copy.
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
        
        typedef typename FTraits::ScalarAltMatrix ScalarAltMatrix;

        /**
         * Returns true if the vectors can be linearly combined
         */
        bool can_linearly_combine() {
            return FTraits::can_linearly_combine;
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

        /** @brief Reduces the feature matrix to a subset of its vectors.
         * 
         * The list of indices is supposed to be ordered.
         *
         * @param begin Beginning of the list of indices
         * @param end End of the list of indices
         */
        void subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) {
            static_cast<Derived*>(this)->Derived::_subset(begin, end, *static_cast<Derived*>(this));
        }
        
        /** @brief Reduces the feature matrix to a subset of its vectors.
         * 
         * The list of indices is supposed to be ordered.
         *
         * @param begin Beginning of the list of indices
         * @param end End of the list of indices
         */
        void subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Derived &other) const {
            static_cast<const Derived*>(this)->Derived::_subset(begin, end, other);
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
        
        /** Assignation operator */
        inline void set(const Derived &f) {
            if (f.size() != this->size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Can only assign feature matrices of same size (%d vs %d)", %this->size() %f.size());
            if (f.size() != 0)
                static_cast<Derived*>(this)->Derived::_set(f);
        }
        
        
        /** 
         * @brief Linear combination of the feature vectors.
         * 
         * Computes \f$ \alpha XA + \beta Y B  \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
         */
        inline Derived linear_combination(const ScalarAltMatrix &mA, Scalar alpha, const Derived &mY, const ScalarAltMatrix &mB, Scalar beta) const {
            // Check for correctedness
            if (mA.rows() != this->size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot linearly combine with a matrix with %d rows (we have %d pre-images)", %mA.rows() %this->size());
            
            // If we have no columns in A, then return an empty feature matrix
            if (mA.cols() == 0)
                return Derived();
            
            // Call the derived
            return static_cast<const Derived*>(this)->Derived::_linear_combination(mA, alpha, &mY, &mB, beta);
        }
        
        
        /** 
         * @brief Linear combination of the feature vectors.
         * 
         * Computes \f$ XA \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
         */
        inline Derived linear_combination(const ScalarAltMatrix &mA, Scalar alpha = (Scalar)1) const {
            // Check for correctedness
            if (mA.rows() != this->size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot linearly combine with a matrix with %d rows (we have %d pre-images)", %mA.rows() %this->size());

            // If we have no columns in A, then return an empty feature matrix
            if (mA.cols() == 0)
                return Derived();
                    
            // Call the derived
            return static_cast<const Derived*>(this)->Derived::_linear_combination(mA, alpha, nullptr, nullptr, 0);
        }

        
        
        

        /** Get the number of feature vectors */
        inline Index size() const {
            return static_cast<const Derived*>(this)->Derived::size();            
        }
        
        
        /** 
         * \brief Get the dimesion of the underlying space.
         * 
         * \todo Say what to do when infinite
         */
        inline Index dimension() const {
            return static_cast<const Derived*>(this)->Derived::dimension();            
        }
        
        /**
         * @brief Computes the Gram matrix of this feature matrix
         * @return A dense self-adjoint matrix
         */
        const typename FTraits::ScalarMatrix & inner() const {
            return static_cast<const Derived*>(this)->Derived::inner();
        }
        
        /** 
         * Remove the i<sup>th</sup> feature vector 
         * @param if swap is true, then the last vector will be swapped with one to remove (faster)
         */
        inline Index remove(Index i, bool swap = false) {
            return static_cast<Derived*>(this)->Derived::remove(i, swap);
        }
        
    };
    

    //! Type information for feature matrices (feeds the ftraits structure)
    template <class _FMatrix> struct FeatureMatrixTypes {};
    
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
            mA.derived().inner<DerivedMatrix>(mB.derived(), _result);
    }

    /// Compute the inner product between two feature matrices and return
    template<typename Derived>
    typename ftraits<Derived>::ScalarMatrix inner(const FeatureMatrix<Derived> &mA, const FeatureMatrix<Derived> &mB) {
        // Define only one result to return for compiler optimisation (direct return)
        typedef typename ftraits<Derived>::ScalarMatrix ScalarMatrix;
        ScalarMatrix result;
        
        // Check for sizes
        if (mA.size() == 0 || mB.size() == 0) 
            // No need to compute anything - we just resize for consistency
            result.resize(mA.size(), mB.size());
        else
            // Compute
            mA.derived().inner<ScalarMatrix>(mB.derived(), result);

        return result;
    }

    
    
#define KQP_FMATRIX_TYPES(FMatrix) \
    typedef ftraits< FMatrix > FTraits; \
    typedef typename FTraits::Scalar Scalar; \
    typedef typename FTraits::ScalarMatrix  ScalarMatrix; \
    typedef typename FTraits::ScalarVector  ScalarVector; \
    typedef typename FTraits::Real Real; \
    typedef typename FTraits::RealVector RealMatrix; \
    typedef typename FTraits::RealVector RealVector; \
    typedef typename FTraits::ScalarAltMatrix  ScalarAltMatrix; \
    typedef typename FTraits::RealAltVector  RealAltVector; \
    typedef typename FTraits::GramMatrix & GramMatrix; \
    typedef typename FTraits::InnerMatrix InnerMatrix;
  
    
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
                       
        //! Inner product matrix
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;

        //! Gram matrix type
        typedef ScalarMatrix& InnerMatrix;
    
        //! Gram matrix type
        typedef ScalarMatrix& GramMatrix;
        
        //! Matrix used for linear combinations       
        typedef typename AltDense<Scalar>::type ScalarAltMatrix;
        
        //! Matrix used as a diagonal
        typedef typename AltVector<Real>::type RealAltVector;
    }; 
    
}

#endif
