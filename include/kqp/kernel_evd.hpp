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

#include <kqp/feature_matrix.hpp>
#include <kqp/rank_selector.hpp>

#include <kqp/reduced_set/unused.hpp>
#include <kqp/reduced_set/null_space.hpp>
#include <kqp/reduced_set/qp_approach.hpp>


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
        KQP_FMATRIX_TYPES(FMatrix);        
        
        KernelEVD() : preImageRatios(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()) {
            
        }
        
        /**
         * @brief Set constraints on the number of pre-images
         */
        void set_pre_images_per_rank(float minimum, float maximum) {
            this->preImageRatios = std::make_pair(minimum, maximum);
        }

        void set_selector(const boost::shared_ptr< const Selector<Scalar> > &selector) {
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
        virtual void add(Real alpha, const FMatrix &mU, const ScalarAltMatrix &mA) {
            _add(alpha, mU, mA);
        }

        /** @brief Rank-n update.
         * 
         * Updates the current decomposition to \f$A^\prime \approx A + X  X^\top\f$
         */
        inline void add(const FMatrix &mU) {
            add(1., mU, ScalarMatrix::Identity(mU.size(),mU.size()));
        }

        /**
         * Get the current decomposition
         * @param mX the pre-images
         * @param mY is used to get a basis from pre-images
         * @param mD is a diagonal matrix
         * @param cleanup Try to reduce the size of the decomposition
         */
        virtual void get_decomposition(FMatrix& mX, ScalarAltMatrix &mY, RealVector& mD, bool do_cleanup = true) const {
            // Get the decomposition from the instance of Kernel EVD
            _get_decomposition(mX, mY, mD);

            if (do_cleanup) 
                cleanup(mX, mY, mD);
        }

        
    protected:
        /**
         * Get the current decomposition
         * @param mX the pre-images
         * @param mY is used to get a basis from pre-images
         * @param mD is a diagonal matrix
         */
        virtual void _get_decomposition(FMatrix& mX, ScalarAltMatrix &mY, RealVector& mD) const = 0;

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

        
        /**
         * @brief Ensures the decomposition has the right rank and number of pre-images 
         */
        void cleanup(FMatrix& mX, ScalarAltMatrix &mY, RealVector& mD) const {
            // --- Rank selection   
            
            DecompositionList<Real> list(mD);
            selector->selection(list);
            
            // Remove corresponding entries
            select_rows(list.getSelected(), mD, mD);
            select_columns(list.getSelected(), mY, mY);
            
            // --- Remove null space
            removePreImagesWithNullSpace(mX, mY);
            
            // --- Ensure we have a small enough number of pre-images
            if (mX.size() > (preImageRatios.second * mD.rows())) {
                if (mX.can_linearly_combine()) {
                    // Easy case: we can linearly combine pre-images
                    mX = mX.linear_combination(mY);
                    mY = ScalarMatrix::Identity(mX.size(), mX.size());
                } else {
                    // Use QP approach
                    ReducedSetWithQP<FMatrix> qp_rs;
//                    qp_rs.run(preImageRatios.first * mD.rows(), mX, mY, mD);
                    mX = qp_rs.getFeatureMatrix();
                    mY = qp_rs.getMixtureMatrix();
                    mD = qp_rs.getEigenValues();
                }
                
            }
        }
        
        void cleanup(FMatrix& mX, ScalarMatrix &mY, RealVector& mD) const {
            ScalarAltMatrix _mY;
            _mY.swap(mY);
            cleanup(mX, _mY, mD);
            _mY.swap(mY);
        }
        
        /**
         * Minimum/Maximum number of pre-images per rank
         */
        std::pair<float,float> preImageRatios;

        //! Eigen value selector
        boost::shared_ptr< const Selector<Real> > selector;

    };

    
    
#define KQP_KERNEL_EVD_INSTANCIATION(qualifier, type)\
   KQP_FOR_ALL_SCALAR_TYPES(qualifier template class type<DenseMatrix<, > >)

        
}

#endif
