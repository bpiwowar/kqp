//
//  generic_feature_matrix.h
//  kqp
// 
//  This file contains all the virtual definitions for the classes needed
//
//  Copyright 2011 Benjamin Piwowarski. All rights reserved.
//

#ifndef __KQP_DENSE_DIRECT_BUILDER_H__
#define __KQP_DENSE_DIRECT_BUILDER_H__

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include "kernel_evd.hpp"
#include "FeatureMatrix/dense_feature_matrix.hpp"

namespace kqp {
    /**
     * @brief Direct computation of the density (i.e. matrix representation) for dense vectors.
     * 
     * @ingroup OperatorBuilder
     */
    template <class Scalar> class DenseDirectBuilder : public OperatorBuilder<DenseVector<Scalar> > {
    public:
        typedef OperatorBuilder<DenseVector<Scalar> > Ancestor;
        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        
        DenseDirectBuilder(int dimension) : matrix(new typename Ancestor::Matrix(dimension, dimension)) {
            matrix->setConstant(0);
        }
        
        virtual void add(typename Ancestor::Real alpha, const typename Ancestor::FVector &v) {
            v.rankUpdateOf(this->matrix->template selfadjointView<Eigen::Lower>(), alpha);
        }
        
        virtual void add(const typename Ancestor::FMatrix &fMatrix, const typename Ancestor::Matrix &coefficients) {
            // Invalidate the cache
            mX = typename Ancestor::FMatrixPtr();
            mD = typename Ancestor::RealVectorPtr();

            if (const ScalarMatrix<Scalar> *mf = dynamic_cast<const ScalarMatrix<Scalar> *> (&fMatrix)) {
                matrix->template selfadjointView<Eigen::Lower>().rankUpdate(mf->getMatrix() * coefficients);
            } else 
                BOOST_THROW_EXCEPTION(not_implemented_exception());
        }
        
        virtual typename Ancestor::FMatrixCPtr getX() const {
            compute();
            return mX;
        }
        
        virtual typename Ancestor::MatrixCPtr getY() const {
            return typename Ancestor::MatrixCPtr();
        }
        
        virtual typename Ancestor::RealVectorPtr getD() const {
            compute();
            return mD;
        }
        
        void compute() const {
            if (!mX.get()) {
                Eigen::SelfAdjointEigenSolver<typename Ancestor::Matrix> evd(*matrix);
                mD = typename Ancestor::RealVectorPtr(new typename Ancestor::RealVector(evd.eigenvalues()));
                mX = typename Ancestor::FMatrixPtr(new ScalarMatrix<Scalar>(evd.eigenvectors()));
            }
        }
        
        
        
    public:
        mutable typename Ancestor::RealVectorPtr mD;
        mutable typename Ancestor::FMatrixPtr mX;
        
        boost::shared_ptr<Matrix> matrix;
    };
    
} // end namespace kqp

#endif
