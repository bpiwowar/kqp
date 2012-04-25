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

    
#define KQP_FMATRIX_TYPES(FMatrix) \
typedef ftraits< FMatrix > FTraits; \
typedef typename ftraits< FMatrix >::Scalar Scalar; \
typedef typename ftraits< FMatrix >::ScalarMatrix  ScalarMatrix; \
typedef typename ftraits< FMatrix >::ScalarVector  ScalarVector; \
typedef typename ftraits< FMatrix >::Real Real; \
typedef typename ftraits< FMatrix >::RealVector RealMatrix; \
typedef typename ftraits< FMatrix >::RealVector RealVector; \
typedef typename ftraits< FMatrix >::ScalarAltMatrix  ScalarAltMatrix; \
typedef typename ftraits< FMatrix >::RealAltVector  RealAltVector; \
typedef typename ftraits< FMatrix >::GramMatrix & GramMatrix; \
typedef typename ftraits< FMatrix >::InnerMatrix InnerMatrix;
    

#define KQP_FMATRIX_COMMON_DEFS(_Self) \
    typedef _Self Self; \
    KQP_FMATRIX_TYPES(_Self); \
    typedef FeatureMatrix< _Self > Base; \
    friend class FeatureMatrix< _Self >; \
    using FeatureMatrix< _Self >::add; \
    using FeatureMatrix< _Self >::subset; \
    using FeatureMatrix< _Self >::linear_combination; \
    using FeatureMatrix< _Self >::dimension; \
    using FeatureMatrix< _Self >::size;


    //! Traits for feature matrices
    template <class Derived> struct ftraits;
    
    /**
     * @brief Base for all feature matrix classes
     * 
     * This class holds a list of vectors whose exact representation might not be
     * known. 
     *
     * The subclasses must define the following methods:
     * - _subset(begin, end, other) 
     * - _linear_combination(mA, alpha, ptr_mY, ptr_mB, beta)
     * - _inner() The gram matrix
     * - _inner(other, result) The inner product between pre-images in the other matrix, storing the result in result
     * - 
     * @ingroup FeatureMatrix
     * @param _FVector the type of the feature vectors
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class Derived> 
    class FeatureMatrix {
    public:       

        KQP_FMATRIX_TYPES(Derived);

        /**
         * Returns true if the vectors can be linearly combined
         */
        bool can_linearly_combine() {
            return FTraits::can_linearly_combine;
        }

        
        inline Derived &derived() {
            return static_cast<Derived&>(*this);
        }
        
        inline const Derived &derived() const {
            return static_cast<const Derived&>(*this);
        }

        /** Add pre-images vectors */
        void add(const Derived &f, const std::vector<bool> *which = NULL) {
            static_cast<Derived*>(this)->Derived::_add(f, which);
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
         * @param list The list of boolean values (true = keep)
         */
        void subset(const std::vector<bool> &list) {
            static_cast<Derived*>(this)->Derived::_subset(list.begin(), list.end(), *static_cast<Derived*>(this));
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
            return static_cast<const Derived*>(this)->Derived::_size();            
        }
        
        
        /** 
         * \brief Get the dimesion of the underlying space.
         * 
         * \todo Say what to do when infinite
         */
        inline Index dimension() const {
            return static_cast<const Derived*>(this)->Derived::_dimension();            
        }
        
        /**
         * @brief Computes the Gram matrix of this feature matrix
         * @return A dense self-adjoint matrix
         */
        inline const typename FTraits::ScalarMatrix & inner() const {
            return static_cast<const Derived*>(this)->Derived::_inner();
        }
        
        template<class DerivedMatrix>
        inline void inner(const Derived &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            static_cast<const Derived*>(this)->Derived::_inner(other, result);
        }
        
        
    };
    

    //! Type information for feature matrices (feeds the ftraits structure)
    template <class _FMatrix> struct FeatureMatrixTypes { 
        typedef void Scalar; 
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
            mA.derived().template inner<DerivedMatrix>(mB.derived(), _result);
    }

    /// Compute the inner product between two feature matrices and return
    template<typename Derived>
    typename ftraits<Derived>::ScalarMatrix inner(const FeatureMatrix<Derived> &mA, const FeatureMatrix<Derived> &mB) {
        // Define only one result to return for compiler optimisation (direct return)
        typedef typename ftraits<Derived>::ScalarMatrix ScalarMatrix;
        ScalarMatrix result(mA.size(), mB.size());
        
        // Check for sizes
        if (mA.size() > 0 && mB.size() > 0) 
            mA.template inner<ScalarMatrix>(mB.derived(), result);

        return result;
    }

    
    
    /**
     * Feature Vector traits
     */
    template <class _FMatrix>
    struct ftraits {
        //! Ourselves
        typedef _FMatrix FMatrix;
        
        //! Definitions
        enum {
            can_linearly_combine = FeatureMatrixTypes<_FMatrix>::can_linearly_combine  
        };
        
        //! Scalar value
        typedef typename FeatureMatrixTypes<_FMatrix>::Scalar Scalar;
                        
        //! Vector of scalars
        typedef Eigen::Matrix<Scalar,Dynamic,1> ScalarVector;
        
        //! Real value associated to the scalar one
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        //! Vector with reals
        typedef Eigen::Matrix<Real,Dynamic,1> RealVector;
                       
        //! Inner product matrix
        typedef Eigen::Matrix<Scalar,Dynamic,Dynamic> ScalarMatrix;

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
