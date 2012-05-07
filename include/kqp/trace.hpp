//! Trace related functions

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

#ifndef __KQP_TRACE_H__
#define __KQP_TRACE_H__

#include <kqp/feature_matrix.hpp>

namespace kqp {
//    
//    /**
//     * @brief Computes \f$ \vert X Y D^2 Y^\dagger X^\dagger \f$.
//     *
//     * Computes \f$ tr( X Y D^2 Y^\dagger X^\dagger X Y D^2 Y^\dagger X^\dagger) = tr( [D Y^\dagger X^\dagger X Y D] [D Y^\dagger X^\dagger X Y D] ) = \vert D Y^\dagger X^\dagger X Y D \vert^2 \f$
//     */
//    template<class Scalar>
//    typename Real squaredNorm(const FeatureMatrix<Scalar> &mX, 
//                          const typename ftraits<Derived>::ScalarAltMatrix  &mY,
//                          const typename ftraits<Derived>::RealVector &mD) {
//        typename ftraits<Derived>::ScalarMatrix m;
//        return (mD.asDiagonal() * mY.transpose() * mX.inner() * mY * mD.asDiagonal()).squaredNorm();
//    }
//    
//    /**
//     * @brief Computes the trace of an operator.
//     *
//     * Computes \f$ tr( X Y D^2 Y^\dagger X^\dagger) \f$
//     *
//     */
//    template<typename Derived, typename OtherDerived>
//    typename ftraits<Derived>::Scalar traceAAT(const FeatureMatrix<Derived> &mX, 
//                 const typename ftraits<Derived>::ScalarAltMatrix  &mY,
//                 const Eigen::MatrixBase<OtherDerived> &mD) {
//        typename ftraits<Derived>::ScalarMatrix m;
//        return (mY.transpose() * mX.inner() * mY * mD.cwiseAbs2().asDiagonal()).trace();
//    }
//    
//    
//    /**
//     * Computes \f$ tr( X_1 Y_1 D_1 Y_1^\dagger X1^\dagger  X_2 Y_2 D_2 Y_2^\dagger X_2^\dagger) \f$
//     *
//     */
//    template<class Scalar>
//    typename Scalar trace_function(const Space<Scalar>
//                          const FeatureMatrix<Scalar> &mX1, 
//                          const typename ScalarAltMatrix  &mY1,
//                          const typename RealVector &mD1,
//                          
//                          const FeatureMatrix<Derived> &mX2, 
//                          const typename ftraits<Derived>::ScalarAltMatrix  &mY2,
//                          const typename ftraits<Derived>::RealVector &mD2) {
//        
//        typename ftraits<Derived>::ScalarMatrix m = mY1.transpose() * inner<Derived>(mX1.derived(), mX2.derived()) * mY2;
//        
//        return (m.adjoint() * mD1.asDiagonal() * m * mD2.asDiagonal()).trace();
//    }
//    
//    /**
//     * Computes the difference between two operators using trace functions
//     */
//    template<class Derived>
//    typename ftraits<Derived>::Scalar difference(const FeatureMatrix<Derived> &mX1, 
//                          const typename ftraits<Derived>::ScalarAltMatrix  &mY1,
//                          const typename ftraits<Derived>::RealVector &mD1,
//                          
//                          const FeatureMatrix<Derived> &mX2, 
//                          const typename ftraits<Derived>::ScalarAltMatrix  &mY2,
//                          const typename ftraits<Derived>::RealVector &mD2) {
//        
//        double tr1 = trace_function(mX1, mY1, mD1, mX1, mY1, mD1);       
//        double tr2 = trace_function(mX2, mY2, mD2, mX2, mY2, mD2);
//        double tr12 = trace_function(mX1, mY1, mD1, mX2, mY2, mD2);
//        
//        return tr1 + tr2 - 2. * tr12;
//    }
}

#endif
