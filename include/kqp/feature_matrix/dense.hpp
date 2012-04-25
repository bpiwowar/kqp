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

#include <numeric>
#include <kqp/subset.hpp>
#include <kqp/feature_matrix.hpp>

namespace kqp {
    
    //! An interval iterator
    struct IntervalsIterator {
        const std::vector<bool> &which;
        std::pair<size_t, size_t> current;
        
        IntervalsIterator &operator++(int) { 
            current.first  = std::find(current.second + 1 + which.begin(), which.end(), true) - which.begin();
            current.second = std::find(which.begin() + current.first, which.end(), false) - which.begin();
            return *this;
        }
        
        IntervalsIterator(const std::vector<bool> &which) : which(which), current(0,0) {
            (*this)++;
        }
        IntervalsIterator(const std::vector<bool> &which, size_t begin, size_t end) : which(which), current(begin,end) {
        }
        const std::pair<size_t, size_t> & operator*() const {
            return current;
        }
        const std::pair<size_t, size_t> *operator->() const {
            return &current;
        }
        bool operator!=(const IntervalsIterator &other) {
            return &which != &other.which || current != other.current;
        }
    };

    
    struct Intervals {
        std::vector<bool> which;
        size_t _selected;
        
        typedef IntervalsIterator Iterator;
        
        const Iterator _end;
        
        Intervals(const std::vector<bool> &which) : which(which), _end(which, which.size(), which.size()) {             
            _selected = std::accumulate(which.begin(), which.end(), 0);
        }
        size_t size() const { return which.size(); }
        size_t selected() const { return _selected; }
        
        Iterator begin() { 
            return Iterator(which);
        }
        const Iterator &end() { 
            return _end;
        }
    };
    

    
    template <typename Scalar> class DenseMatrix;
    
    /**
     * @brief A feature matrix where vectors are dense vectors in a fixed dimension.
     * @ingroup FeatureMatrix
     */
    template <typename _Scalar> 
    class DenseMatrix : public FeatureMatrix< DenseMatrix<_Scalar> > {
    public:       
        KQP_FMATRIX_COMMON_DEFS(DenseMatrix<_Scalar>);
   
        //! Null constructor: will set the dimension with the first feature vector
        DenseMatrix() {}
        
        //! Construct an empty feature matrix of a given dimension
        DenseMatrix(Index dimension) : matrix(dimension, 0) {
        }

#ifndef SWIG
        //! Construction by moving a dense matrix
        DenseMatrix(ScalarMatrix &&m) : matrix(m) {}
#endif
        //! Construction by copying a dense matrix
        DenseMatrix(const ScalarMatrix &m) : matrix(m) {}

   
        /**
         * Add a vector (from a template expression)
         */
        template<typename Derived>
        void add(const Eigen::DenseBase<Derived> &m) {
            if (matrix.cols() == 0) 
                matrix.resize(m.rows(), 0);
            else if (m.rows() != dimension())
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, 
                                      "Cannot add a vector of dimension %d (dimension is %d)", % m.rows() % dimension());

            Index n = matrix.cols();
            matrix.conservativeResize(matrix.rows(), n + m.cols());
            this->matrix.block(0, n, dimension(), m.cols()) = m; 
        }
        
          /**
         * Add a vector (from a template expression)
         */
        template<typename Derived>
        void add(const Eigen::DenseBase<Derived> &m, const std::vector<bool> &which) {
            if (matrix.cols() == 0) 
                matrix.resize(m.rows(), 0);
            if (m.rows() != dimension())
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, 
                                      "Cannot add a vector of dimension %d (dimension is %d)", % m.rows() % dimension());
            
            Index s = std::accumulate(which.begin(), which.end(), 0);
            matrix.conservativeResize(matrix.rows(), s + m.cols());

            Intervals intervals(which);
            Index offset = size();
            for(auto i = intervals.begin(); i != intervals.end(); i++) {
                Index cols = i->second - i->first;
                this->matrix.block(0, offset, dimension(), cols) = m.block(0, i->first, dimension(), cols);
                offset += cols;
            }
                                
        }
        
        //! Get a const reference to the matrix
        const ScalarMatrix& getMatrix() const {
            return this->matrix;
        }
        
        const ScalarMatrix &toDense() const {
            return this->matrix;
        }

    protected:
        
        void _add(const Self &other, const std::vector<bool> *which = NULL)  {
            if (which) this->add(other.getMatrix(), *which);
            else this->add(other.getMatrix());
        }
        
                       
        Index _size() const { 
            return matrix.cols();
        }
        
        Index _dimension() const {
            return matrix.rows();
        }

        const ScalarMatrix &_inner() const {
            if (size() == 0) return gramMatrix;
            
            // We lose space here, could be used otherwise???
            Index current = gramMatrix.rows();
            if (current < size()) 
                gramMatrix.conservativeResize(size(), size());
            
            Index tofill = size() - current;
            
            // Compute the remaining inner products
            gramMatrix.bottomRightCorner(tofill, tofill).noalias() = this->getMatrix().rightCols(tofill).adjoint() * this->getMatrix().rightCols(tofill);
            gramMatrix.topRightCorner(current, tofill).noalias() = this->getMatrix().leftCols(current).adjoint() * this->getMatrix().rightCols(tofill);
            gramMatrix.bottomLeftCorner(tofill, current) = gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return gramMatrix;
        }
        
        //! Computes the inner product with another matrix
        template<class DerivedMatrix>
        void _inner(const Self &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            const_cast<Eigen::MatrixBase<DerivedMatrix>&>(result)= this->getMatrix().adjoint() * other.getMatrix();
        }
        
        
        Self _linear_combination(const ScalarAltMatrix & mA, Scalar alpha, const Self *mY, const ScalarAltMatrix *mB, Scalar beta) const {
            if (mY == 0) 
                return Self(ScalarMatrix(alpha * (getMatrix() * mA)));
            ScalarMatrix m(alpha * getMatrix() * mA);
            m += beta * mY->getMatrix() * *mB;
            return Self(std::move(m));   
        }
        

        void _subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Self &into) const {
            ScalarMatrix m;
            select_columns(begin, end, this->matrix, m);
            into = Self(std::move(m));
        }

               
    private:        
        //! Cache of the gram matrix
        mutable ScalarMatrix gramMatrix;
        
        //! Our matrix
        ScalarMatrix matrix;
    };
    
    
    
    template<typename Scalar>
    std::ostream& operator<<(std::ostream &out, const DenseMatrix<Scalar> &f) {
        return out << "[Dense Matrix with scalar " << KQP_DEMANGLE((Scalar)0) << "]" << std::endl << f.getMatrix();
    }
    
    
    // The scalar for dense feature matrices
    template <typename _Scalar> struct FeatureMatrixTypes<DenseMatrix<_Scalar> > {
        typedef _Scalar Scalar;
        enum {
            can_linearly_combine = 1
        };
    };
    
    
# // Extern templates
# ifndef SWIG
# define KQP_SCALAR_GEN(scalar) extern template class DenseMatrix<scalar>;
# include <kqp/for_all_scalar_gen>
# endif
    
} // end namespace kqp

#endif