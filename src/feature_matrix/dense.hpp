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
#ifndef __KQP__H__
#define __KQP__H__

#include "feature_matrix.hpp"

namespace kqp {
    template <typename Scalar> class ScalarMatrix;
    
    
    
    //! A scalar vector
    template <typename _Scalar>
    class DenseVector {
    public:       
        typedef _Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        
        DenseVector() {
        }
        
        DenseVector(const ScalarMatrix<Scalar> &m, Index i) : fMatrix(&m), index(i) {
        }
        
        template<typename Derived>
        DenseVector(const Eigen::DenseBase<Derived> &v) : vector(new Vector(v)) {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        }
        
        /// Copy our value into vector v
        template<class Derived>
        void assignTo(const Eigen::DenseBase<Derived> &_v) const {
            Eigen::DenseBase<Derived> &v = const_cast<Eigen::DenseBase<Derived>&>(_v);
            if (fMatrix.get()) {
                v = fMatrix->getMatrix().col(index);
            } else {
                v = *vector;
            }
        }
        
        // Update the matrix with a rank-one update based on ourselves
        template<class Derived, unsigned int UpLo>
        void rankUpdateOf(const Eigen::SelfAdjointView<Derived, UpLo> &_m, Scalar alpha) const {
            Eigen::SelfAdjointView<Derived, UpLo> &m = const_cast<Eigen::SelfAdjointView<Derived, UpLo> &>(_m);
            
            if (fMatrix.get())
                m.rankUpdate(fMatrix->getMatrix().col(index), alpha);
            else
                m.rankUpdate(*vector, alpha);
        }
        
        friend struct Inner<DenseVector<Scalar> >;
        
        void print(std::ostream &os) {
            if (this->fMatrix.get())
                 os << this->fMatrix->getMatrix().col(index);
            else
                os << (*(this->vector));
        }
    private:
        boost::intrusive_ptr<const ScalarMatrix<Scalar> > fMatrix;
        Index index;
        
        boost::shared_ptr<Vector> vector; 
    };
    
    
        
    // Defines a scalar product between two dense vectors
    template <typename Scalar>
    struct Inner<DenseVector<Scalar> > {
        static Scalar compute(const DenseVector<Scalar> &a, const DenseVector<Scalar> &b) {
            switch((a.fMatrix.get() ? 2 : 0) + (b.fMatrix.get() ? 1 : 0)) {
                case 0: return a.vector->dot(*b.vector);
                case 1: return a.vector->dot(b.fMatrix->getMatrix().col(b.index));
                case 2: return a.fMatrix->getMatrix().col(a.index).dot(*b.vector);
                case 3: return a.fMatrix->getMatrix().col(a.index).dot(b.fMatrix->getMatrix().col(b.index));
            }
            KQP_THROW_EXCEPTION(assertion_exception, "Unknown vector types");  
        }
    };
    
    /**
     * @brief A feature matrix where vectors are dense vectors in a fixed dimension.
     * @ingroup FeatureMatrix
     */
    template <typename Scalar> 
    class ScalarMatrix : public FeatureMatrix<DenseVector<Scalar> > {
    public:
        //! Our parent
        typedef FeatureMatrix<DenseVector<Scalar> > Parent;
        //! Our feature vector
        typedef DenseVector<Scalar> FVector;
        //! The type of the matrix
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;


        ScalarMatrix(Index dimension) {
            matrix = boost::shared_ptr<Matrix>(new Matrix(dimension, 0));
        }
        
        template<class Derived>
        ScalarMatrix(const Eigen::EigenBase<Derived> &m) : matrix(new Matrix(m)) {
        }
        
		virtual ~ScalarMatrix() {}
        long size() const { return this->matrix->cols();  }
        
        FVector get(Index i) const { return FVector(*this, i); }
        
        virtual void add(const FVector &f)  {
            Index n = matrix->cols();
            matrix->resize(matrix->rows(), n + 1);
           f.assignTo(matrix->col(n));
        }
        
        /**
         * Add a vector (from a template expression)
         */
        template<typename Derived>
        void add(const Eigen::DenseBase<Derived> &vector) {
            if (vector.rows() != matrix->rows())
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, "Cannot add a vector of dimension %d (dimension is %d)", % vector.rows() % matrix->rows());
            if (vector.cols() != 1)
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, "Expected a vector got a matrix with %d columns", % vector.cols());
            
            Index n = matrix->cols();
            matrix->resize(matrix->rows(), n + 1);
            matrix->col(n) = vector;
        }
        
        virtual void set(Index i, const FVector &f) {
            f.assignTo(matrix->col(i));
        }
        
        virtual void remove(Index i, bool swap = false) {
            Index last = matrix->cols() - 1;
            if (swap) {
                if (i != last) 
                    matrix->col(i) = matrix->col(last);
            } else {
                for(Index j = i + 1; j <= last; j++)
                    matrix->col(i-1) = matrix->col(i);
            }
            matrix->resize(matrix->rows(), last);
            
        }
        
        void swap(Matrix &m) {
            if (m.rows() != matrix->rows())
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Dimension of matrices is different (%d vs %d)", % m.rows() % matrix->rows());
            matrix->swap(m);
        }
        
        
        const Matrix& getMatrix() const {
            return *matrix;
        }
        
    private:
        //! Our matrix
        boost::shared_ptr<Matrix> matrix;
    };
    
    
    
} // end namespace kqp

#endif