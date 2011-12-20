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
    
    /**
     * @brief A feature matrix where vectors are dense vectors in a fixed dimension.
     * @ingroup FeatureMatrix
     */
    template <typename _Scalar> 
    class DenseMatrix : public FeatureMatrix< DenseMatrix<_Scalar> > {
    public:
        //! Scalar
        typedef _Scalar Scalar;
        
        //! Ourselves
        typedef DenseMatrix<Scalar> Self;
        
        //! Our traits
        typedef ftraits<DenseMatrix<Scalar> > FTraits;
        
        //! The type of inner product matrices
        typedef typename FTraits::Matrix Matrix;
        
        //! Null constructor: will set the dimension with the first feature vector
        DenseMatrix() : view_mode(false), column_start(0), _size(0) {}
        
        DenseMatrix(Index dimension) : view_mode(false), column_start(0), _size(0), matrix(new Matrix(dimension, 0)) {
        }
        
        template<class Derived>
        explicit DenseMatrix(const Eigen::EigenBase<Derived> &m) : view_mode(false), column_start(0), _size(m.cols()), matrix(new Matrix(m)) {
        }
        
        Index size() const { 
            return _size;
        }
        
        
        const Index dimension() const {
            return matrix.get() ? matrix->rows() : 0;
        }
        
        //! Adds a list of pre-images
        void add(const Self &other)  {
            this->check_can_modify();
            
            if (!matrix.get()) 
                matrix.reset(new Matrix(other.dimension(), 0));
            
            this->add(other.get_matrix());
        }
        
        /**
         * Add a vector (from a template expression)
         */
        template<typename Derived>
        void add(const Eigen::DenseBase<Derived> &m) {
            this->check_can_modify();
            
            if (m.rows() != dimension())
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, 
                                      "Cannot add a vector of dimension %d (dimension is %d)", % m.rows() % dimension());
            
            Index n = matrix->cols();
            matrix->conservativeResize(matrix->rows(), n + m.cols());
            _size += m.cols();
            this->matrix->block(0, n+column_start, dimension(), m.cols()) = m; 
        }
        
        
        void remove(Index i, bool swap) {
            this->check_can_modify();
            
            Index last = matrix->cols() - 1;
            if (swap) {
                if (i != last) 
                    matrix->col(i) = matrix->col(last);
            } else {
                for(Index j = i + 1; j <= last; j++)
                    matrix->col(i-1) = matrix->col(i);
            }
            matrix->conservativeResize(matrix->rows(), last);
            _size--;
        }
        
        /**
         * Swap with another matrix
         */
        template<class Derived>
        void swap(Eigen::MatrixBase<Derived> &m) {
            if (!matrix.get()) 
                matrix.reset(new Matrix());
            matrix->swap(m);
            view_mode = false;
            column_start = 0;
            _size = matrix->cols();
            this->gramMatrix.resize(0,0);
        }
        
        //! Get a reference to the matrix
        const Eigen::Block<Matrix> get_matrix() const {
            return this->matrix->block(0, column_start, matrix->rows(), _size);
        }

        Eigen::Block<Matrix> get_matrix() {
            return this->matrix->block(0, column_start, matrix->rows(), _size);
        }
        
        const Matrix & inner() const {
            if (_size == 0) return gramMatrix;
            
            // We lose space here, could be used otherwise???
            Index current = gramMatrix.rows();
            if (current < size()) 
                gramMatrix.conservativeResize(size(), size());
            
            Index tofill = size() - current;
            
            // Compute the remaining inner products
            gramMatrix.bottomRightCorner(tofill, tofill).noalias() = this->get_matrix().rightCols(tofill).adjoint() * this->get_matrix().rightCols(tofill);
            gramMatrix.topRightCorner(current, tofill).noalias() = this->get_matrix().leftCols(current).adjoint() * this->get_matrix().rightCols(tofill);
            gramMatrix.bottomLeftCorner(tofill, current) = gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return gramMatrix;
        }
        
        //! Computes the inner product with another matrix
        template<class DerivedMatrix>
        void inner(const Self &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            const_cast<Eigen::MatrixBase<DerivedMatrix>&>(result)= this->get_matrix().adjoint() * other.get_matrix();
        }
        
        bool is_view() {
            return this->view_mode; 
        }
        
        
        
    protected:
        Self _linear_combination(const AltMatrix<Scalar> & mA, Scalar alpha) const {
            return Self(alpha * (get_matrix() * mA))
            ;
        }
        
        void _set(const Self &f) {            
            if (f.size() > 0)
                this->get_matrix() = f.get_matrix();            
            this->gramMatrix.resize(0,0);
        }
        
        
    private:   
        friend class FeatureMatrix<Self>;
        
        //! Creates a view
        DenseMatrix(const Self &other, Index i, Index size) : 
        view_mode(true), column_start(other.column_start+i), _size(size), matrix(other.matrix) {
            
        }
        
        const Self _view(Index i, Index view_size) const { 
            if (i + view_size >= this->size() - column_start)
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot get the %d-%d vector range among %d vectors", 
                                      % (i+1) % (i+view_size+1) % size());
            return Self(*this, i, view_size);
        }
        
        
        inline void check_can_modify() const {
            if (view_mode)
                KQP_THROW_EXCEPTION(illegal_operation_exception, "Cannot change the dimension of a view: clone it first");
        }
        
        //! Are we a view
        bool view_mode;
        
        //! Our view
        Index column_start, _size;
        
        //! Our inner product
        mutable Matrix gramMatrix;
        
        //! Our matrix
        boost::shared_ptr<Matrix> matrix;
    };
    
    
    
    template<typename Scalar>
    std::ostream& operator<<(std::ostream &out, const DenseMatrix<Scalar> &f) {
        return out << "[Dense Matrix with scalar " << KQP_DEMANGLE((Scalar)0) << "]" << std::endl << f.get_matrix();
    }
    
    
    // The scalar for dense feature matrices
    template <typename _Scalar> struct FeatureMatrixTypes<DenseMatrix<_Scalar> > {
        typedef _Scalar Scalar;
        enum {
            can_linearly_combine = 1
        };
    };
    
    
    // Extern templates
    KQP_FOR_ALL_SCALAR_TYPES(extern template class DenseMatrix<, >;);
    
} // end namespace kqp

#endif