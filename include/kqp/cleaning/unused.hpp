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
    class CleanerUnused : public Cleaner<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
    private:
         template <typename T> static void _run(const FMatrixPtr &mF, T &mY) {
            // Dimension of the problem
            Index N = mY.rows();
            assert(N == mF->size());
            
            
            // Removes unused pre-images
            RealVector v = mY.rowwise().squaredNorm();
            Real threshold = v.sum() * Eigen::NumTraits<Scalar>::epsilon();

            bool change = false;
            std::vector<bool> to_keep(N, true);
            for(Index i = 0; i < N; i++) 
                if (v(i) < threshold) {
                    change = true;
                    to_keep[i] = false;
                }
            
            if (!change) return;
            
            select_rows(to_keep, mY, mY);
            
            *mF = *mF->subset(to_keep);
        }       
    public:
        
        virtual void cleanup(Decomposition<Scalar> &d) const override {
            run(d.mX, d.mY);
        }
        
        /**
         * @brief Removes unused pre-images
         */
        static void run(const FMatrixPtr &mF, ScalarMatrix &mY) {
            CleanerUnused<Scalar>::_run(mF, mY);
        }
        
        static void run(const FMatrixPtr &mF, ScalarAltMatrix &mY) {
            CleanerUnused<Scalar>::_run(mF, mY);
        }
    };
    
}

#endif
