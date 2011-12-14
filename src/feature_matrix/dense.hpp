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
#ifndef __KQP_DENSE_FEATURE_MATRIX_H__
#define __KQP_DENSE_FEATURE_MATRIX_H__

#include "feature_matrix.hpp"

namespace kqp {
    template <typename Scalar> class DenseMatrix;
    
    
    template<typename Scalar>
    class DenseVector : public FeatureMatrixView<DenseMatrix<Scalar> > {
    public:
        typedef typename DenseMatrix<Scalar>::FTraits FTraits;
        
        DenseVector(const DenseMatrix<Scalar> &matrix, Index column) : matrix(matrix), column(column) {}
        
        Scalar inner(const DenseVector &other) const {
            return get().dot(other.get());
        }
        
        typename Eigen::DenseBase<typename FTraits::Matrix>::ConstColXpr get() const { 
            return matrix.get_matrix().col(column); 
        }
        
        void _linear_combination(Scalar alpha, const typename FTraits::Matrix & mA, typename FTraits::FMatrix &result) const {
            typename FTraits::Matrix m = alpha * this->get() * mA;
            result.swap(m);
        }
        
        template<class DerivedMatrix>
        Scalar inner(const DenseVector<Scalar> &mB) const {
            return get().dot(mB.get());
        }
        
        virtual const typename FTraits::FVector get(Index i) const {
            if (i == 0) return *this;
            KQP_THROW_EXCEPTION_F(out_of_bound_exception, "A feature matrix composed of a vector has only one element (index %d asked)", %index);
        }
        
        const typename FTraits::Matrix & inner() const {
            if (is_empty(gram_matrix)) {
                gram_matrix.resize(1,1);
                gram_matrix(0,0) = this->get().dot(this->get());
            }
            return gram_matrix;
        }
        
        virtual Index size() const {
            return 1;
        }
        
    private:
        mutable typename FTraits::Matrix gram_matrix;
        const DenseMatrix<Scalar> &matrix;
        Index column;
        friend class DenseMatrix<Scalar>;
    };
    
    /**
     * @brief A feature matrix where vectors are dense vectors in a fixed dimension.
     * @ingroup FeatureMatrix
     */
    template <typename _Scalar> 
    class DenseMatrix : public FeatureMatrix< DenseMatrix<_Scalar> > {
    public:
        //! Scalar
        typedef _Scalar Scalar;
        
        //! Our feature vector
        typedef DenseVector<Scalar> FVector;
        
        //! Ourselves
        typedef DenseMatrix<Scalar> Self;
        
        //! Our traits
        typedef ftraits<DenseMatrix<Scalar> > FTraits;
        
        //! The type of inner product matrices
        typedef typename FTraits::Matrix Matrix;
        
        //! Null constructor: will set the dimension with the first feature vector
        DenseMatrix() {}
        
        DenseMatrix(Index dimension) : matrix(dimension, 0) {
        }
        
        template<class Derived>
        explicit DenseMatrix(const Eigen::EigenBase<Derived> &m) : matrix(m) {
        }
        
        Index size() const { 
            return this->matrix.cols();  
        }
        
        const FVector get(Index i) const { 
            return FVector(*this, i);
        }
        
        
        virtual void add(const typename FTraits::FVector &f)  {
            Index n = matrix.cols();
            matrix.conservativeResize(matrix.rows() != 0 ? matrix.rows() : f.matrix.matrix.rows(), n + 1);
            matrix.col(n) = f.get();
        }
        
        /**
         * Add a vector (from a template expression)
         */
        template<typename Derived>
        void add(const Eigen::DenseBase<Derived> &vector) {
            if (vector.rows() != matrix.rows())
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, "Cannot add a vector of dimension %d (dimension is %d)", % vector.rows() % matrix.rows());
            if (vector.cols() != 1)
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, "Expected a vector got a matrix with %d columns", % vector.cols());
            
            Index n = matrix.cols();
            matrix.conservativeResize(matrix.rows(), n + 1);
            matrix.col(n) = vector;
        }
        
        virtual void set(Index i, const FVector &f) {
            matrix.col(i) = f.get();
            // FIXME: Recompute just the needed inner vectors
            this->gramMatrix.resize(0,0);
        }
        
        void remove(Index i, bool swap) {
            Index last = matrix.cols() - 1;
            if (swap) {
                if (i != last) 
                    matrix.col(i) = matrix.col(last);
            } else {
                for(Index j = i + 1; j <= last; j++)
                    matrix.col(i-1) = matrix.col(i);
            }
            matrix.conservativeResize(matrix.rows(), last);
            
        }
        
        void swap(Matrix &m) {
            matrix.swap(m);
            this->gramMatrix.resize(0,0);
        }
        
        //! Get a reference to our matrix
        const Matrix& get_matrix() const {
            return matrix;
        }
        
        const Matrix & inner() const {
            // We lose space here, could be used otherwise???
            Index current = gramMatrix.rows();
            if (current < size()) 
                gramMatrix.conservativeResize(size(), size());
            
            Index tofill = size() - current;
            
            // Compute the remaining inner products
            gramMatrix.bottomRightCorner(tofill, tofill).noalias() = matrix.rightCols(tofill).adjoint() * matrix.rightCols(tofill);
            gramMatrix.topRightCorner(current, tofill).noalias() = matrix.leftCols(current).adjoint() * matrix.rightCols(tofill);
            gramMatrix.bottomLeftCorner(tofill, current) = gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return gramMatrix;
        }
        
        
        template<class DerivedMatrix>
        void inner(const Self &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            const_cast<Eigen::MatrixBase<DerivedMatrix>&>(result)= matrix.adjoint() * other.matrix;
        }
        
        template<class DerivedMatrix>
        void inner(const FVector &x, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            
            const_cast<Eigen::MatrixBase<DerivedMatrix>&>(result) = matrix.adjoint() * x.get();
        }
        
    protected:
        friend class FeatureMatrixView<typename FTraits::FMatrix>;
        
        void _linear_combination(Scalar alpha, const typename FTraits::Matrix & mA, typename FTraits::FMatrix &result) const {
            if (&result != this) {
                if (!is_empty(mA))
                    result.matrix.noalias() = alpha * get_matrix() * mA;
                else 
                    result.matrix = alpha * get_matrix();
            } else
                if (!is_empty(mA))
                    result.matrix = alpha * get_matrix() * mA;
                else if (alpha != (Scalar)1)
                    result.matrix *= alpha;
            
        }
        
    private:
        //! Our inner product
        mutable Matrix gramMatrix;
        
        //! Our matrix
        Matrix matrix;
    };
    
    // The scalar for dense feature matrices
    template <typename _Scalar> struct FeatureMatrixTypes<DenseMatrix<_Scalar> > {
        typedef _Scalar Scalar;
        typedef DenseVector<Scalar> FVector;
    };
    
    
    // Extern templates
    KQP_FOR_ALL_SCALAR_TYPES(extern template class DenseVector<, >;);
    KQP_FOR_ALL_SCALAR_TYPES(extern template class DenseMatrix<, >;);
    
} // end namespace kqp

#endif