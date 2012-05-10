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

namespace kqp {
    
    //! An "EVD" decomposition
    template<typename Scalar>
    struct Decomposition {
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        //! Feature space
        FSpace fs;
        
        //! The feature matrix
        FMatrix  mX;
        
        //! The linear combination matrix
        ScalarAltMatrix mY;
        
        //! The diagonal matrix
        RealAltVector mD;
        
        //! If this is a real decomposition
        bool orthonormal;
        
        //! Number of rank updates
        Index updateCount;
        
        //! Default constructor with an undefined feature space
        Decomposition() : orthonormal(true) {}
        
        //! Default constructor with a feature space
        Decomposition(const FSpace &fs) : fs(fs), mX(fs.newMatrix()), orthonormal(true) {}
        
        //! Full constructor
        Decomposition(const FSpace &fs, const FMatrix &mX, const ScalarAltMatrix &mY, const RealAltVector &mD, bool orthonormal) 
            : fs(fs), mX(fs.newMatrix(mX)), mY(mY), mD(mD), orthonormal(orthonormal), updateCount(0) {}

#ifndef SWIG
        //! Move constructor
        Decomposition(FSpace &&fs, FMatrix &&mX, const ScalarAltMatrix &&mY, const RealAltVector &&mD, bool orthonormal) 
        : fs(fs), mX(mX), mY(mY), mD(mD), orthonormal(orthonormal), updateCount(0) {
        }

        //! Move constructor
        Decomposition(Decomposition &&other)  {
            *this = std::move(other);
        }

        //! Move assignement
        Decomposition &operator=(Decomposition &&other) {
            swap(other);
            return *this;
        }
#endif
        void swap(Decomposition &other) {
            fs = std::move(other.fs);
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
            fs = other.fs;
            mX = other.mX;
            mY = other.mY;
            mD = other.mD;
            orthonormal = other.orthonormal;
            updateCount = other.updateCount;
            return *this;
        }
        
        
        /**
         * Computes \f$ D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2 \f$
         */
        ScalarMatrix k(const Decomposition &other) const {
            return fs.k(mX, mY, mD, other.mX, other.mY, other.mD);
        }
        
        
    };   
    
        
}

#ifndef SWIG
#define KQP_SCALAR_GEN(type) extern template struct kqp::Decomposition<type>;
#include <kqp/for_all_scalar_gen.h.inc>
#endif

#endif

