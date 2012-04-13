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

#ifndef __KQP_DIVIDE_AND_CONQUER_BUILDER_H__
#define __KQP_DIVIDE_AND_CONQUER_BUILDER_H__

#include <kqp/kernel_evd.hpp>

namespace kqp {
    
    template<typename FMatrix> 
    class KernelEVDFactory {
    public:
        virtual boost::shared_ptr<KernelEVD<FMatrix>> create() = 0;
    };
    
    /**
     * @brief Meta builder: Uses other operator builders and combine them at regular intervals.
     * @ingroup KernelEVD
     */
    template <class FMatrix> class DivideAndConquerBuilder : public KernelEVD<FMatrix> {
    public:
        KQP_FMATRIX_TYPES(FMatrix);        

        //! Set the maximum number of rank updates before using a combiner
        void setBatchSize(Index batchSize) { 
            this->batchSize = batchSize;
        }
        
        //! Sets the builder factory
        void setBuilderFactory(const boost::shared_ptr<KernelEVDFactory<FMatrix>> &builderFactory) {
            this->builderFactory = builderFactory;
        }
        
        //! Sets the merger factory
        void setMergerFactory(const boost::shared_ptr<KernelEVDFactory<FMatrix>> &mergerFactory) {
            this->mergerFactory = mergerFactory;
        }
        
        
        virtual Decomposition<FMatrix> getDecomposition() const override {
        }
        
    protected:
        virtual void _add(Real alpha, const FMatrix &mU, const ScalarAltMatrix &mA) override {
            // Get a builder
            
            
        }
        
    private:
        //! Counts of rank updates for each builder/merger
        std::vector<Index> counts; 
        
        //! Vector of kernel-EVD builder/merger
        std::vector<boost::unique_ptr<KernelEVD<FMatrix>> builders;
        
        Index batchSize = 100;
        
        boost::shared_ptr<KernelEVDFactory<FMatrix>> builderFactory;
        
        boost::shared_ptr<KernelEVDFactory<FMatrix>> mergerFactory;        
    };
}

#endif

