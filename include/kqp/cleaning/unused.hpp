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

#ifndef __KQP_REDUCED_SET_UNUSED_H__
#define __KQP_REDUCED_SET_UNUSED_H__

#include <kqp/cleanup.hpp>
#include <kqp/feature_matrix.hpp>
#include <kqp/subset.hpp>

namespace kqp {
    template<typename Scalar>
    struct CleanerUnused : public Cleaner<Scalar> {
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        virtual void cleanup(Decomposition<Scalar> &d) const override {
            run(d.mX, d.mY);
        }
        
        /**
         * @brief Removes unused pre-images
         */
        static void run(const FMatrixPtr &mF, ScalarMatrix &mY) {
            // Dimension of the problem
            Index N = mY.rows();
            assert(N == mF->size());
            
            std::vector<bool> to_keep(N, true);
            
            // Removes unused pre-images
            for(Index i = 0; i < N; i++) 
                if (mY.row(i).norm() < Eigen::NumTraits<Scalar>::epsilon()) 
                    to_keep[i] = false;
            
            select_rows(to_keep, mY, mY);
            const_cast<FMatrixPtr &>(mF) = mF->subset(to_keep);
        }
        
        static void run(const FMatrixPtr &mF, ScalarAltMatrix &mY) {
            // Dimension of the problem
            Index N = mY.rows();
            assert(N == mF->size());
            
            
            // Removes unused pre-images
            RealVector v = mY.rowwise().squaredNorm();
            
            bool change = false;
            std::vector<bool> to_keep(N, true);
            for(Index i = 0; i < N; i++) 
                if (v(i) < Eigen::NumTraits<Scalar>::epsilon()) {
                    change = true;
                    to_keep[i] = false;
                }
            
            if (!change) return;
            
            select_rows(to_keep, mY, mY);
            
            const_cast<FMatrixPtr &>(mF) = mF->subset(to_keep);
        }
    };
    
# ifndef SWIG
# define KQP_SCALAR_GEN(Scalar) extern template struct CleanerUnused<Scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
# endif 
}

#endif
