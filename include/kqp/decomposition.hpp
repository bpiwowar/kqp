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

#ifndef _KQP_DECOMPOSITION_H
#define _KQP_DECOMPOSITION_H

#include <utility>
#include <kqp/feature_matrix.hpp>
#include <kqp/matrix_computations.hpp>

namespace kqp {
    
    //! An "EVD" decomposition
    template<typename FMatrix>
    struct Decomposition {
        KQP_FMATRIX_TYPES(FMatrix);        
        
        //! The feature matrix
        FMatrix mX;
        
        //! The linear combination matrix
        ScalarAltMatrix mY;
        
        //! The diagonal matrix
        RealAltVector mD;
        
        //! If this is a real decomposition
        bool orthonormal;
        
        //! Number of rank updates
        Index updateCount;
        
        //! Default constructor (sets orthonormal to true)
        Decomposition() : orthonormal(true) {}
        
        //! Full constructor
        Decomposition(const FMatrix &mX, const ScalarAltMatrix &mY, const RealAltVector &mD, bool orthonormal) 
        : mX(mX), mY(mY), mD(mD), orthonormal(orthonormal), updateCount(0) {}

#ifndef SWIG
        //! Full constructor
        Decomposition(const FMatrix &&mX, const ScalarAltMatrix &&mY, const RealAltVector &&mD, bool orthonormal) 
        : mX(mX), mY(mY), mD(mD), orthonormal(orthonormal), updateCount(0) {}

        //! Move constructor
        Decomposition(Decomposition &&other) {
            *this = std::move(other);
        }

        //! Move assignement
        Decomposition &operator=(Decomposition &&other) {
            swap(other);
            return *this;
        }
#endif
        void swap(Decomposition &other) {
            mX = std::move(other.mX);
            mY.swap(other.mY);
            mD.swap(other.mD);
            std::swap(orthonormal, other.orthonormal);
            std::swap(updateCount, other.updateCount);
        }

        
        //! Copy constructor
        Decomposition(const Decomposition &other) {
            *this = other;
        }
                
        //! Copy assignement
        Decomposition &operator=(const Decomposition &other) {
            mX = other.mX;
            mY = other.mY;
            mD = other.mD;
            orthonormal = other.orthonormal;
            updateCount = other.updateCount;
            return *this;
        }
        
        
        ScalarMatrix innerXY(const Decomposition<FMatrix>& that) const {
            return FCompute<FMatrix>::inner(mX, mY, that.mX, that.mY);
        }
        
        ScalarMatrix innerXYD(const Decomposition<FMatrix>& that) const {
            return FCompute<FMatrix>::inner(mX, mY, mD, that.mX, that.mY, that.mD);
        }

        
    };   
    
        
}

#ifndef SWIG
#define KQP_FMATRIX_GEN_EXTERN(type) extern template struct kqp::Decomposition<type>;
#include <kqp/for_all_fmatrix_gen>
#endif

#endif

