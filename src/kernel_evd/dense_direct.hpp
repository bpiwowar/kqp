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

#include <Eigen/Eigenvalues>

#include <Eigen/Core>

#include "kernel_evd.hpp"
#include "feature_matrix/dense.hpp"
#include "kernel_evd/utils.hpp"

namespace kqp {
    /**
     * @brief Direct computation of the density (i.e. matrix representation) for dense vectors.
     * 
     * @ingroup KernelEVD
     */
    template <class Scalar> class DenseDirectBuilder : public KernelEVD<DenseMatrix<Scalar> > {
    public:
        typedef ftraits<DenseMatrix<Scalar> > FTraits;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef DenseVector<Scalar> FVector;
        typedef typename FTraits::Real Real;
        
        DenseDirectBuilder(int dimension) : matrix(dimension, dimension) {
            matrix.setConstant(0);
        }
        
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrixView &mX, const typename FTraits::Matrix &mA) {
            if (is_empty(mA))
                if (const DenseMatrix<Scalar> * _mX = dynamic_cast<const DenseMatrix<Scalar>*>(&mX))
                    matrix.template selfadjointView<Eigen::Lower>().rankUpdate(_mX->get_matrix(), alpha);
                else if (const DenseVector<Scalar> * _mX = dynamic_cast<const DenseVector<Scalar>*>(&mX))
                    matrix.template selfadjointView<Eigen::Lower>().rankUpdate(_mX->get(), alpha);
                else 
                    KQP_THROW_EXCEPTION_F(assertion_exception, "Expected a DenseMatrix or DenseVector, got %s", %KQP_DEMANGLE(mX));
            else  {
                typename FTraits::FMatrix mX_mA;
                mX.linear_combination(mA, mX_mA);
                matrix.template selfadjointView<Eigen::Lower>().rankUpdate(mX_mA.get_matrix(), alpha);
            }
        }
        
        virtual void get_decomposition(typename FTraits::FMatrix& mX, typename FTraits::Matrix &mY, typename FTraits::RealVector& mD) {
            Eigen::SelfAdjointEigenSolver<typename FTraits::Matrix> evd(matrix.template selfadjointView<Eigen::Lower>());
            kqp::thinEVD(evd, mY, mD);  
            mX.swap(mY);
            mY.resize(0,0);
        }
        
    public:
        
        Matrix matrix;
    };
    
    KQP_FOR_ALL_SCALAR_TYPES(extern template class DenseDirectBuilder<, >;);
    
    
} // end namespace kqp

#endif
