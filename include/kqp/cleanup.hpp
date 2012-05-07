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
#ifndef _KQP_CLEANUP_H_
#define _KQP_CLEANUP_H_

#include <boost/shared_ptr.hpp>
#include <kqp/decomposition.hpp>

#include <kqp/rank_selector.hpp>
#include <kqp/reduced_set/unused.hpp>
#include <kqp/reduced_set/null_space.hpp>
#include <kqp/reduced_set/qp_approach.hpp>

namespace kqp {
    
    template<typename Scalar> class Cleaner {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        //! Default constructor
        Cleaner() : preImageRatios(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()), useLinearCombination(true) {}
        
        virtual ~Cleaner() {}
        
        //! Sets the flag for linear combination use (debug)
        void setUseLinearCombination(bool flag) {
            useLinearCombination = flag; 
        }
        
        //! Set constraints on the number of pre-images
        void setPreImagesPerRank(float minimum, float maximum) {
            this->preImageRatios = std::make_pair(minimum, maximum);
        }
        
        //! Set the rank selector
        void setSelector(const boost::shared_ptr< const Selector<Real> > &selector) {
            this->selector = selector;
        }
        
        //! Cleanup
        virtual void cleanup(Decomposition<Scalar> &) {}
        
    protected:
        /**
         * Minimum/Maximum number of pre-images per rank
         */
        std::pair<float,float> preImageRatios;
        
        //! Eigen value selector
        boost::shared_ptr< const Selector<Real> > selector;

        //! Flag to use linear combination (debug)
        bool useLinearCombination;

    };
    
    
    template<typename Scalar>
    class StandardCleaner : public Cleaner<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
	
        /**
         * @brief Ensures the decomposition has the right rank and number of pre-images 
         */
        void cleanup(Decomposition<Scalar> &d) override {
            // --- Rank selection   
            DecompositionList<Real> list(d.mD);
            if (this->selector) {
                
                this->selector->selection(list);
                
                // Remove corresponding entries
                select_rows(list.getSelected(), d.mD, d.mD);
                
                // Case where mY is the identity matrix
                if (d.mY.getTypeId() == typeid(typename AltDense<Scalar>::IdentityType)) {
                    d.mX = d.mX.subset(list.getSelected());
                    d.mY.conservativeResize(list.getRank(), list.getRank());
                } else {
                    select_columns(list.getSelected(), d.mY, d.mY);
                }
            }
            
            // --- Remove unused pre-images
            RemoveUnusedPreImages<Scalar>::run(d.mX, d.mY);
            
            // --- Remove null space
            ReducedSetNullSpace<Scalar>::run(d.fs, d.mX, d.mY);
            
            // --- Ensure we have a small enough number of pre-images
            if (d.mX.size() > (this->preImageRatios.second * d.mD.rows())) {
                if (d.fs.canLinearlyCombine() && this->useLinearCombination) {
                    // Easy case: we can linearly combine pre-images
                    d.mX = d.fs.linearCombination(d.mX, d.mY);
                    d.mY = ScalarMatrix::Identity(d.mX.size(), d.mX.size());
                } else {
                    // Use QP approach
                    ReducedSetWithQP<Scalar> qp_rs;
                    qp_rs.run(this->preImageRatios.first * d.mD.rows(), d.fs, d.mX, d.mY, d.mD);
                    
                    // Get the decomposition
                    d.mX = qp_rs.getFeatureMatrix();
                    d.mY = qp_rs.getMixtureMatrix();
                    d.mD = qp_rs.getEigenValues();
                    
                    // The decomposition is not orthonormal anymore
                    d.orthonormal = false;
                }
                
            }
        }
        
    };
}

# ifndef SWIG
# define KQP_SCALAR_GEN(type)  \
         extern template class kqp::Cleaner<type>; \
         extern template class kqp::StandardCleaner<type>; 
# include <kqp/for_all_scalar_gen.h.inc>
# endif

#endif


