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

/** \file
 *  This file contains common matrix operations for given types, and thus is useful to reduce compilation times
 */

#ifndef __KQP_MATRIX_COMPUTATIONS_H__
#define __KQP_MATRIX_COMPUTATIONS_H__

#include <kqp/feature_matrix.hpp>

namespace kqp {
    //! Computations related to a scalar
    template<typename Scalar> 
    struct Compute {
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        typedef Matrix<Scalar,Dynamic,Dynamic> ScalarMatrix;
        typedef typename AltDense<Scalar>::type ScalarAltMatrix;
        typedef typename AltVector<Real>::type RealAltVector;
        
        static ScalarMatrix inner(const ScalarMatrix &preImageInners, 
                                  const ScalarAltMatrix &mY1, 
                                  const ScalarAltMatrix &mY2) {
            return mY1.transpose() * preImageInners * mY2;
        }
        
        static ScalarMatrix inner(const ScalarMatrix &preImageInners, 
                                  const ScalarAltMatrix &mY1, const RealAltVector &mD1,
                                  const ScalarAltMatrix &mY2, const RealAltVector &mD2) {
            return mD1.asDiagonal() * inner(preImageInners, mY1, mY2) * mD2.asDiagonal();
        }
    };
    
    //! Computations related to Feature matrix
    template<typename FMatrix>
    struct FCompute {
        KQP_FMATRIX_TYPES(FMatrix);
        
        static ScalarMatrix inner(const FMatrix &mX1, const ScalarAltMatrix &mY1, 
                                  const FMatrix &mX2, const ScalarAltMatrix &mY2) {
            return  Compute<Scalar>::inner(kqp::inner(mX1, mX2), mY1, mY2);
        }
        
        static ScalarMatrix inner(const FMatrix &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1, 
                                  const FMatrix &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) {
            return  Compute<Scalar>::inner(kqp::inner(mX1, mX2), mY1, mD1, mY2, mD2);
        }
    };
    
}

#ifndef SWIG

#define KQP_SCALAR_GEN(Scalar) extern template struct kqp::Compute<Scalar>;
#include <kqp/for_all_scalar_gen>

#define KQP_FMATRIX_GEN_EXTERN(type) extern template struct kqp::FCompute<type>;
#include <kqp/for_all_fmatrix_gen>

#endif

#endif