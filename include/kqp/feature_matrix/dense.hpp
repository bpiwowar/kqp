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

#include <kqp/subset.hpp>
#include <kqp/feature_matrix.hpp>

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
        typedef typename FTraits::ScalarMatrix ScalarMatrix;
        typedef typename FTraits::ScalarAltMatrix ScalarAltMatrix;
        
        //! Null constructor: will set the dimension with the first feature vector
        DenseMatrix() : view_mode(false), column_start(0), _size(0) {}
        
        DenseMatrix(Index dimension) : view_mode(false), column_start(0), _size(0), matrix(new ScalarMatrix(dimension, 0)) {
        }
        
        template<class Derived>
        explicit DenseMatrix(const Eigen::EigenBase<Derived> &m) : view_mode(false), column_start(0), _size(m.cols()), matrix(new ScalarMatrix(m)) {
        }
        
        Index size() const { 
            return _size;
        }
        
        
        Index dimension() const {
            return matrix.get() ? matrix->rows() : 0;
        }
        
        //! Adds a list of pre-images
        void add(const Self &other)  {
            this->check_can_modify();
            
            if (!matrix.get()) 
                matrix.reset(new ScalarMatrix(other.dimension(), 0));
            
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
        
        
        Index remove(Index i, bool swap) {
            this->check_can_modify();
            Index r = -1;
            Index last = matrix->cols() - 1;
            if (swap) {
                if (i != last) {
                    matrix->col(i) = matrix->col(last);
                    r = last;
                }
            } else {
                for(Index j = i + 1; j <= last; j++)
                    matrix->col(i-1) = matrix->col(i);
            }
            matrix->conservativeResize(matrix->rows(), last);
            _size--;
            return r;
        }
        
        /**
         * Swap with another matrix
         */
        template<class Derived>
        void swap(Eigen::MatrixBase<Derived> &m) {
            if (!matrix.get()) 
                matrix.reset(new ScalarMatrix());
            matrix->swap(m.derived());
            view_mode = false;
            column_start = 0;
            _size = matrix->cols();
            this->gramMatrix.resize(0,0);
        }
        
        void swap(typename AltDense<Scalar>::type &m) {
            if (!matrix.get()) 
                matrix.reset(new ScalarMatrix());
            m.swap(*this->matrix);
            view_mode = false;
            column_start = 0;
            _size = matrix->cols();
            this->gramMatrix.resize(0,0);            
        }

        
        //! Get a reference to the matrix
        const Eigen::Block<ScalarMatrix> get_matrix() const {
            return this->matrix->block(0, column_start, matrix->rows(), _size);
        }

        //! Get a non-const reference to the matrix
        Eigen::Block<ScalarMatrix> get_matrix() {
            return this->matrix->block(0, column_start, matrix->rows(), _size);
        }
        
        // Computes the Gram matrix
        const ScalarMatrix & inner() const {
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
        Self _linear_combination(const ScalarAltMatrix & mA, Scalar alpha, const Self *mY, const ScalarAltMatrix *mB, Scalar beta) const {
            if (mY == 0) 
                return Self(ScalarMatrix(alpha * (get_matrix() * mA)));
            else {
                ScalarMatrix m(alpha * get_matrix() * mA);
                m += beta * mY->get_matrix() * *mB;
             return Self(m);   
            }
        }
        


        /// Makes a subset
        void _subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Self &into) const {
            if (&into == this) 
                into.check_can_modify();
            
            boost::shared_ptr<ScalarMatrix> m(new ScalarMatrix());
            select_columns(begin, end, *this->matrix, *m);
            into = Self(m);
        }

        

        /// Copy from another dense matrix
        void _set(const Self &f) {            
            view_mode = false;
            column_start = 0;
            
            this->get_matrix() = f.get_matrix();            
            this->gramMatrix.resize(0,0);
        }
        
        
    private:   
        friend class FeatureMatrix<Self>;
        
        // New featuer matrix from shared pointer
        explicit DenseMatrix(const boost::shared_ptr<ScalarMatrix> &m) : view_mode(false), column_start(0), _size(m->cols()), matrix(m) {
        }
        
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
        mutable ScalarMatrix gramMatrix;
        
        //! Our matrix
        boost::shared_ptr<ScalarMatrix> matrix;
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