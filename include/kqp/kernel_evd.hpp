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

#ifndef __KQP_KERNEL_EVD_H__
#define __KQP_KERNEL_EVD_H__

#include <utility>


#include <kqp/decomposition.hpp>
#include <kqp/feature_matrix.hpp>
#include <kqp/rank_selector.hpp>



namespace kqp {

    /**
     * @brief Builds a compact representation of an hermitian operator. 
     *
     * Computes \f$ \mathfrak{U} = \sum_{i} \alpha_i \mathcal U A A^\top \mathcal U ^ \top   \f$
     * as \f$ {\mathcal X}^\dagger Y^\dagger D Y \mathcal X \f$ where \f$D\f$ is a diagonal matrix
     * and \f$ \mathcal X Y \f$ is an orthonormal matrix, i.e. \f$  \mathcal Y^\dagger {\mathcal X}^\dagger \mathcal X Y \f$
     * is the identity.
     *
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     *
     * @param <FMatrix>
     *            The type of the base vectors in the original space
     * @ingroup OperatorBuilder
     */
    template <class Scalar> class KernelEVD {        
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        KernelEVD(const FSpace &featureSpace) : nbUpdates(0), m_featureSpace(featureSpace)  {
        }
        

        virtual ~KernelEVD() {}
               

        /**
         * @brief Rank-n update.
         *
         * Updates the current decomposition to \f$A^\prime \approx A + \alpha   X A A^T  X^\top\f$
         * 
         * @param alpha
         *            The coefficient for the update
         * @param mX  The feature matrix X with n feature vectors.
         * @param mA  The mixture matrix (of dimensions n x k).
         */
        virtual void add(Real alpha, const FMatrix &mX, const ScalarAltMatrix &mA) {
            // Check consistency
            if (mX->size() != mA.rows())
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot combine %d pre-images with a %d rows matrix", %mX->size() %mA.rows());

            _add(alpha, mX, mA);
            nbUpdates += mA.cols();
        }

        /** @brief Rank-n update.
         * 
         * Updates the current decomposition to \f$A^\prime \approx A + X  X^\top\f$
         */
        inline void add(const FMatrix &mU) {
            add(1., mU, Eigen::Identity<Scalar>(mU->size(),mU->size()));
            nbUpdates += mU->size();
        }

        /**
         * Get the current decomposition
         */
        Decomposition<Scalar> getDecomposition() const {
            Decomposition<Scalar> d(_getDecomposition());
            d.updateCount = nbUpdates;
            d.fs = this->m_featureSpace;
            // Creates a new empty feature matrix when nothing is returned
            if (!d.mX)
                d.mX = d.fs->newMatrix();
            if (!d.check())
                KQP_THROW_EXCEPTION_F(assertion_exception, "Decomposition in an invalid state (%d, %dx%d, %d) for KEVD %s", 
                        %d.mX->size() %d.mY.rows() %d.mY.cols() %d.mD.rows() %KQP_DEMANGLE(*this))

            return d;
        }

        //! Resets the state of the builder
        virtual void reset() {
            nbUpdates = 0;
        }
        
        //! Return the number of updates
        Index getUpdateCount() const {
            return nbUpdates;
        }
    
        const FSpace &getFSpace() const { return m_featureSpace; }
        
    protected:
        /**
         * @brief Rank-n update.
         *
         * Updates the current decomposition to \f$A^\prime \approx A + \alpha   X A A^T  X^\top\f$
         * 
         * @param alpha
         *            The coefficient for the update
         * @param mX  The feature matrix X with n feature vectors.
         * @param mA  The mixture matrix (of dimensions n x k).
         */
        virtual void _add(Real alpha, const FMatrix &mU, const ScalarAltMatrix &mA) = 0;

        /** Get the decomposition */
        virtual Decomposition<Scalar> _getDecomposition() const = 0;
    
    private:
        Index nbUpdates;
        FSpace m_featureSpace;
    };

    
        
}


#endif
