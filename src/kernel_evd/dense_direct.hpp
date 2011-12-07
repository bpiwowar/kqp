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

#ifndef __KQP_DENSE_DIRECT_BUILDER_H__
#define __KQP_DENSE_DIRECT_BUILDER_H__

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include "kernel_evd.hpp"
#include "feature_matrix/dense.hpp"

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
        typedef DenseVector<Scalar> FVector;
        typedef typename Ancestor::Real Real;
        
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
                typedef Eigen::SelfAdjointEigenSolver<typename Ancestor::Matrix> EigenSolver;
                
                // Swap vectors for efficiency
                EigenSolver evd(matrix->template selfadjointView<Eigen::Lower>());
                
                mD = typename Ancestor::RealVectorPtr(new typename Ancestor::RealVector());
                const_cast<typename Ancestor::RealVector&>(evd.eigenvalues()).swap(*mD);
                // Take the square root, and set to zero eigenvalues which are negative (should 
                // be very small negatives)
                for(Index i = 0; i < mD->rows(); i++)
                    if ((*mD)[i] < 0) (*mD)[i] = 0;
                    else (*mD)[i] = Eigen::internal::sqrt((*mD)[i]);
                
                ScalarMatrix<Scalar> *_mX = new ScalarMatrix<Scalar>(matrix->rows());
                mX = typename Ancestor::FMatrixPtr(_mX);
                _mX->swap(const_cast<typename Ancestor::Matrix&>(evd.eigenvectors()));
                
            }
        }
        
        
        
    public:
        mutable typename Ancestor::RealVectorPtr mD;
        mutable typename Ancestor::FMatrixPtr mX;
        
        boost::shared_ptr<Matrix> matrix;
    };
    
} // end namespace kqp

#endif
