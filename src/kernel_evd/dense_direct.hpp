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
        typedef typename FTraits::Real Real;
        
        DenseDirectBuilder(int dimension) : matrix(dimension, dimension) {
            matrix.setConstant(0);
        }
        
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrix &mX, const typename FTraits::AltMatrix &mA) {
            matrix.template selfadjointView<Eigen::Lower>().rankUpdate(mX.get_matrix() * mA, alpha);
        }
        
        virtual void get_decomposition(typename FTraits::FMatrix& mX, typename FTraits::AltMatrix &mY, typename FTraits::RealVector& mD) {
            Eigen::SelfAdjointEigenSolver<typename FTraits::Matrix> evd(matrix.template selfadjointView<Eigen::Lower>());
            
            typename FTraits::Matrix _mX;
            kqp::thinEVD(evd, _mX, mD);  
            mX.swap(_mX);
            
            mY = AltMatrix<Scalar>::Identity(mX.size());
        }
        
    public:
        
        Matrix matrix;
    };
    
    KQP_FOR_ALL_SCALAR_TYPES(extern template class DenseDirectBuilder<, >;);
    
    
} // end namespace kqp

#endif
