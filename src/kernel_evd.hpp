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

#ifndef __KQP_OPERATOR_BUILDER_H__
#define __KQP_OPERATOR_BUILDER_H__

#include "feature_matrix.hpp"

namespace kqp {
    /**
     * @brief Builds a compact representation of an hermitian operator. 
     *
     * Computes \f$ \mathfrak{U} = \sum_{i} \alpha_i \mathcal U A A^\top \mathcal U ^ \top   \f$
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     *
     * @param <Scalar>
     *            The Scalar type
     * @param <F>
     *            The type of the base vectors in the original space
     * @ingroup OperatorBuilder
     */
    template <class _FVector> class OperatorBuilder  {        
    public:
        typedef _FVector FVector;        
        typedef typename FVector::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        typedef typename Eigen::Matrix<Real, Eigen::Dynamic, 1> RealVector;
        typedef boost::shared_ptr<RealVector> RealVectorPtr;
        typedef boost::shared_ptr<const RealVector> RealVectorCPtr;
        
        typedef FeatureMatrix<FVector> FMatrix;
        typedef typename FMatrix::Ptr FMatrixPtr;
        typedef typename FMatrix::CPtr FMatrixCPtr;
        
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef boost::shared_ptr<Matrix> MatrixPtr;
        typedef boost::shared_ptr<const Matrix> MatrixCPtr;
        
        //! Virtual destructor to build the vtable
        virtual ~OperatorBuilder() {}
        
        //! Returns the feature matrix
        virtual FMatrixCPtr getX() const = 0;
        
        /** @brief Get the feature pre-image combination matrix.
         * @return a reference to the matrix (a null matrix pointer is returned instead of identity)
         */
        virtual MatrixCPtr getY() const = 0;
        
        //! Returns the diagonal matrix as a vector
        virtual RealVectorPtr getD() const = 0;
        
        /**
         * Add a new vector to the density
         *
         * Computes \f$A^\prime \approx A + \alpha   X A A^T  X^\top\f$
         * 
         * @param alpha
         *            The coefficient for the update
         * @param mX  The feature matrix X with n feature vectors
         * @param mA  The mixture matrix (of dimensions n x k)
         */
        virtual void add(const FMatrix &mX, const Matrix &mA) = 0;
        
        
        virtual void add(Real alpha, const FVector &mX) = 0;
        
    };
    
}

#endif
