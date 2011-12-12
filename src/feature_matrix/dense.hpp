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
    template <typename Scalar> class DenseMatrix;
    
    //! A scalar vector
    template <typename _Scalar>
    class DenseVector {
    public:       
        typedef _Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        
        enum Type {
            UNKNOWN,
            MATRIX_COL,
            VECTOR
        };
        
        DenseVector() : type(UNKNOWN) {
        }
        
        DenseVector(const DenseMatrix<Scalar> &m, Index i) : fMatrix(&m), index(i), type(MATRIX_COL) {
        }
        
        template<typename Derived>
        DenseVector(const Eigen::DenseBase<Derived> &v) : vector(new Vector(v)), type(VECTOR) {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        }
        
        /// Copy our value into vector v
        template<class Derived>
        void assignTo(const Eigen::DenseBase<Derived> &_v) const {
            Eigen::DenseBase<Derived> &v = const_cast<Eigen::DenseBase<Derived>&>(_v);
            switch(type) {
                case MATRIX_COL: v = fMatrix->getMatrix().col(index); break;
                case VECTOR: v = *vector; break;
                default: KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Unknown dense vector type %d", %type);
            }
        }
        

        // Update the matrix with a rank-one update based on ourselves
        template<class Derived, unsigned int UpLo>
        void rankUpdateOf(const Eigen::SelfAdjointView<Derived, UpLo> &_m, Scalar alpha) const {
            Eigen::SelfAdjointView<Derived, UpLo> &m = const_cast<Eigen::SelfAdjointView<Derived, UpLo> &>(_m);
            switch(type) {
                case MATRIX_COL:  m.rankUpdate(fMatrix->getMatrix().col(index), alpha); break;
                case VECTOR: m.rankUpdate(*vector, alpha); break;
                default: KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Unknown dense vector type %d", %type);                
            }
        }
        
        friend struct Inner<DenseVector<Scalar> >;
        
        void print(std::ostream &os) {
            switch(type) {
                case MATRIX_COL:  os << this->fMatrix->getMatrix().col(index); break;
                case VECTOR: os << (*(this->vector)); break;
                default: os << "Unknown dense vector of type " << type;                
            }
        }
        
        Index rows() const {
            switch(type) {
                case MATRIX_COL: return this->fMatrix->getMatrix().rows();
                case VECTOR: return this->vector->rows();
                default: KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Unknown dense vector type %d", %type);                
            }   
        }
        
        void axpy(const Scalar &b, const DenseVector<Scalar> &y) {
            switch(type) {
                case MATRIX_COL:
                    KQP_THROW_EXCEPTION(illegal_operation_exception, "Unknown dense vector type ");

                case VECTOR: 
                    switch(y.type) {
                        case MATRIX_COL:  *vector += b * y.fMatrix->getMatrix().col(index); break;
                        case VECTOR: *vector += b * *y.vector; break;
                        default: KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Unknown dense vector type %d", %type);                 
                    }
                    break;
                    
                case UNKNOWN:
                    type = VECTOR;
                    vector.reset(new Vector(y.rows()));
                    switch(y.type) {
                        case MATRIX_COL:  *vector = b * y.fMatrix->getMatrix().col(index); break;
                        case VECTOR: *vector = b * *y.vector; break;
                        default: KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Unknown dense vector type %d", %type);                 
                    }
                    break;
                    
                default: 
                    KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Unknown dense vector type %d", %type);                 
            }
            
        }
        

        
        boost::intrusive_ptr<const DenseMatrix<Scalar> > fMatrix;
        Index index;
        
        boost::shared_ptr<Vector> vector; 
        
        Type type;
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
    
    /** By default, vectors cannot be combined */
    template<typename Scalar> 
    struct linear_combination<DenseVector<Scalar> > {         
        //! Cannot combine by default
        static const bool can_combine = true; 
        
        /**
         * Computes x <- x + b * y
         */
        static void axpy(DenseVector<Scalar> & x, const Scalar &b, const DenseVector<Scalar>  &y) {
            x.axpy(b, y);
        };
    };
    
    
    /**
     * @brief A feature matrix where vectors are dense vectors in a fixed dimension.
     * @ingroup FeatureMatrix
     */
    template <typename Scalar> 
    class DenseMatrix : public FeatureMatrix<DenseVector<Scalar> > {
    public:
        //! Our parent
        typedef FeatureMatrix<DenseVector<Scalar> > Parent;
        //! Our feature vector
        typedef DenseVector<Scalar> FVector;
        //! The type of the matrix
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        //! The type of inner product matrices
        typedef typename Parent::InnerMatrix InnerMatrix;
        
        //! Null constructor: will set the dimension with the first feature vector
        DenseMatrix() {}
        
        DenseMatrix(Index dimension) {
            matrix = boost::shared_ptr<Matrix>(new Matrix(dimension, 0));
        }
        
        template<class Derived>
        DenseMatrix(const Eigen::EigenBase<Derived> &m) : matrix(new Matrix(m)) {
        }
        
		virtual ~DenseMatrix() {}
        
        Index size() const { 
            return this->matrix->cols();  
        }
        
        FVector get(Index i) const { 
            return FVector(*this, i); 
        }
        
        virtual void add(const FVector &f)  {
            if (!matrix.get()) 
                matrix.reset(new Matrix(f.rows(), 0));
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
    
    
    // Extern templates
    extern template class DenseVector<double>;
    extern template class DenseVector<float>;
    extern template class DenseVector<std::complex<double> >;
    extern template class DenseVector<std::complex<float> >;

    extern template class DenseMatrix<double>;
    extern template class DenseMatrix<float>;
    extern template class DenseMatrix<std::complex<double> >;
    extern template class DenseMatrix<std::complex<float> >;

} // end namespace kqp

#endif