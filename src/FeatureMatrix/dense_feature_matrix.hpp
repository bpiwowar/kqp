//
//  generic_feature_matrix.h
//  kqp
// 
//  This file contains all the virtual definitions for the classes needed
//
//  Copyright 2011 Benjamin Piwowarski. All rights reserved.
//

#ifndef __KQP__H__
#define __KQP__H__

#include "kernel_evd.hpp"

namespace kqp {
    template <typename Scalar> class ScalarMatrix;
    
    
    
    //! A scalar vector
    template <typename _Scalar>
    class DenseVector {
    public:       
        typedef _Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> Vector;
        
        DenseVector() {
        }
        
        DenseVector(const ScalarMatrix<Scalar> &m, Index i) : fMatrix(&m), index(i) {
        }
        
        template<typename Derived>
        DenseVector(const Eigen::DenseBase<Derived> &v) : vector(new Vector()) {
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

    private:
        boost::intrusive_ptr<const ScalarMatrix<Scalar> > fMatrix;
        Index index;
        
        boost::shared_ptr<Vector> vector; 
    };
    
    template <typename Scalar>
    Scalar inner(const DenseVector<Scalar> &a, const DenseVector<Scalar> &b) {
        BOOST_THROW_EXCEPTION(not_implemented_exception());
    }
    
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
        typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> Matrix;

        ScalarMatrix(const Matrix &matrix) : matrix(new Matrix(matrix)) {
        }

        ScalarMatrix(Index dimension) {
            matrix = boost::shared_ptr<Matrix>(new Matrix(dimension, 0));
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
        
        
        const Matrix& getMatrix() const {
            return *matrix;
        }
        
    private:
        //! Our matrix
        boost::shared_ptr<Matrix> matrix;
    };
    
    
    
} // end namespace kqp

#endif