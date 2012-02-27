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
    /**
     * @brief Computes || X Y D^2 Y^T X^T ||^2.
     *
     * Computes tr( X Y D^2 Y^T X^T X Y D^2 Y^T X^T) = tr( [D Y^T X^T X Y D] [D Y^T X^T X Y D] ) = || D Y^T X^T X Y D ||^2
     */
    template<class Derived>
    double squaredNorm(const FeatureMatrix<Derived> &mX, 
                          const typename ftraits<Derived>::ScalarAltMatrix  &mY,
                          const typename ftraits<Derived>::RealVector &mD) {
        typename ftraits<Derived>::ScalarMatrix m;
        noalias(m) = mD.asDiagonal() * mY.transpose() * mX.inner() * mY * mD.asDiagonal();
        return m.squaredNorm();
    }
    
    /**
     * Computes tr( X1 Y1 D1 Y1^T X1^T  X2 Y2 D2 Y2^T X2^T)
     * 
     * as
     *  tr( D1 Y1^T X1^T  X2 Y2 D2 Y2^T X2^T Y1^T X1^T )
     */
    template<class Derived>
    double trace_function(const FeatureMatrix<Derived> &mX1, 
                          const typename ftraits<Derived>::ScalarAltMatrix  &mY1,
                          const typename ftraits<Derived>::RealVector &mD1,
                          
                          const FeatureMatrix<Derived> &mX2, 
                          const typename ftraits<Derived>::ScalarAltMatrix  &mY2,
                          const typename ftraits<Derived>::RealVector &mD2) {
        typedef typename ftraits<Derived>::ScalarVector Vector;
        typename ftraits<Derived>::Matrix m;
        inner<Derived>(mX1.derived(), mX2.derived(),m);
        
        m = mY1.transpose() * m * mY2;
        
        double trace = 0;
        for(Index i = 0; i < m.rows(); i++) {
            Vector x = m.row(i).adjoint().cwiseProduct(mD2);
            Vector y = m.row(i).adjoint();
            
            trace += mD1[i] * x.dot(y);
        }
        
        return trace;
    }
    
    /**
     * Computes the difference between two operators
     */
    template<class Derived>
    double difference(const FeatureMatrix<Derived> &mX1, 
                          const typename ftraits<Derived>::ScalarAltMatrix  &mY1,
                          const typename ftraits<Derived>::RealVector &mD1,
                          
                          const FeatureMatrix<Derived> &mX2, 
                          const typename ftraits<Derived>::ScalarAltMatrix  &mY2,
                          const typename ftraits<Derived>::RealVector &mD2) {
        
        double tr1 = trace_function(mX1, mY1, mD1, mX1, mY1, mD1);       
        double tr2 = trace_function(mX2, mY2, mD2, mX2, mY2, mD2);
        double tr12 = trace_function(mX1, mY1, mD1, mX2, mY2, mD2);
        
        return tr1 + tr2 - 2. * tr12;
    }
}

#endif
