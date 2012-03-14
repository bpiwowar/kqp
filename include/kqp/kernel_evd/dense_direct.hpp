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

#include <kqp/kernel_evd.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include <kqp/kernel_evd/utils.hpp>

namespace kqp {
    /**
     * @brief Direct computation of the density (i.e. matrix representation) for dense vectors.
     * 
     * @ingroup KernelEVD
     */
    template <class _Scalar> class DenseDirectBuilder : public KernelEVD< DenseMatrix<_Scalar> > {
    public:
        typedef DenseMatrix<_Scalar> FMatrix;
        KQP_FMATRIX_TYPES(DenseMatrix<_Scalar>);        
        
        DenseDirectBuilder(int dimension) : matrix(dimension, dimension) {
            matrix.setConstant(0);
        }
        
        virtual void _add(Real alpha, const FMatrix &mX, const ScalarAltMatrix &mA) override {
            matrix.template selfadjointView<Eigen::Lower>().rankUpdate(ScalarMatrix(mX.get_matrix() * mA), alpha);
        }
        
        virtual Decomposition<FMatrix> getDecomposition() const override {
            Decomposition<FMatrix> d;
            Eigen::SelfAdjointEigenSolver<typename FTraits::ScalarMatrix> evd(matrix.template selfadjointView<Eigen::Lower>());
            
            ScalarAltMatrix _mX;
            kqp::thinEVD(evd, _mX, d.mD);              
            
            d.mY = ScalarMatrix::Identity(d.mX.size(), d.mX.size());
            d.mX.swap(_mX);
            return d;
        }
        
    public:
        
        ScalarMatrix matrix;
    };
    
    KQP_FOR_ALL_SCALAR_TYPES(extern template class DenseDirectBuilder<, >;);
    
    
} // end namespace kqp

#endif
