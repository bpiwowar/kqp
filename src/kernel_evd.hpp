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


#include "feature_matrix.hpp"
#include "rank_selector.hpp"

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
    template <class FMatrix> class KernelEVD  {        
    public:
        typedef ftraits<FMatrix> FTraits;
        
        
        KernelEVD() : pre_images_per_rank(std::numeric_limits<float>::infinity()) {
            
        }
        
        void set_pre_images_per_rank(float pre_images_per_rank) {
            this->pre_images_per_rank = pre_images_per_rank;
        }

        void set_selector(const boost::shared_ptr<const Selector> &selector) {
            this->selector = selector;
        }
        
        //! Virtual destructor to build the vtable
        ~KernelEVD() {}
               
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
        virtual void add(typename FTraits::Real alpha, const typename FTraits::FMatrix &mU, const typename FTraits::AltMatrix &mA) = 0;

        /**
         * Get the current decomposition
         * @param mX the pre-images
         * @param mY is used to get a basis from pre-images
         * @param mD is a diagonal matrix
         */
        virtual void get_decomposition(typename FTraits::FMatrix& mX, typename FTraits::AltMatrix &mY, typename FTraits::RealVector& mD) {
            // Get the decomposition from the instance of Kernel EVD
            _get_decomposition(mX, mY, mD);
            
            // Reduce the rank
        }

        
    protected:
        /**
         * Get the current decomposition
         * @param mX the pre-images
         * @param mY is used to get a basis from pre-images
         * @param mD is a diagonal matrix
         */
        virtual void _get_decomposition(typename FTraits::FMatrix& mX, typename FTraits::AltMatrix &mY, typename FTraits::RealVector& mD) = 0;

        /**
         * Maximum number of pre-images per rank
         */
        float pre_images_per_rank;
        
        /**
         * Eigen value selector
         */
        boost::shared_ptr<const Selector> selector;

    };

    
    
#define KQP_KERNEL_EVD_INSTANCIATION(qualifier, type)\
    qualifier template class type<DenseMatrix<double> >
//    KQP_FOR_ALL_SCALAR_TYPES(qualifier template class type<DenseMatrix<, > >)

        
}

#endif
