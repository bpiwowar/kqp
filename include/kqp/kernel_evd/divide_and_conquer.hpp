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

#include <kqp/cleanup.hpp>
#include <kqp/kernel_evd.hpp>

namespace kqp {
    
    /**
     * @brief Meta builder: Uses other operator builders and combine them at regular intervals.
     * @ingroup KernelEVD
     */
    template <typename Scalar> class DivideAndConquerBuilder : public KernelEVD<Scalar>  {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);        
        
        DivideAndConquerBuilder(const FSpace &fs) : KernelEVD<Scalar>(fs) {}
        virtual ~DivideAndConquerBuilder() {}
        
        //! Set the maximum number of rank updates before using a combiner
        void setBatchSize(Index batchSize) { 
            this->batchSize = batchSize;
        }
        
        //! Sets the builder
        void setBuilder(const boost::shared_ptr< KernelEVD<Scalar>  > &builder) {
            this->builder = builder;
        }
        
        //! Sets the builder cleaner
        void setBuilderCleaner(const boost::shared_ptr< Cleaner<Scalar> > &builderCleaner) {
            this->builderCleaner = builderCleaner;
        }
        
        //! Sets the merger
        void setMerger(const boost::shared_ptr<KernelEVD<Scalar>  > &merger) {
            this->merger = merger;
        }
        
        //! Sets the merger cleaner
        void setMergerCleaner(const boost::shared_ptr<Cleaner<Scalar> > &mergerCleaner) {
            this->mergerCleaner = mergerCleaner;
        }
        
        
    protected:
        void reset() {
            *this = DivideAndConquerBuilder(this->getFSpace());
        }

        virtual Decomposition<Scalar> _getDecomposition() const override {
            // Flush last merger
            const_cast<DivideAndConquerBuilder&>(*this).flushBuilder();
            
            // Empty decomposition if we had no data
            if (decompositions.size() == 0) 
                return Decomposition<Scalar>();
            
            
            // Merge everything
            const_cast<DivideAndConquerBuilder&>(*this).merge(true);
            return decompositions[0];
        }
        
        // Rank update
        virtual void _add(Real alpha, const FMatrix &mU, const ScalarAltMatrix &mA) override {
            // Prepare
            if (builder->getUpdateCount() > batchSize)  {
                flushBuilder();
                merge(false);
            }
            // Update the decomposition
            builder->add(alpha, mU, mA);
            
        }
        
    private:
        // Add the current decomposition to the stack
        void flushBuilder() {
            if (builder->getUpdateCount() == 0) return;
            
            // Get & clean
            decompositions.push_back(builder->getDecomposition());
            if (builderCleaner.get())
                builderCleaner->cleanup(decompositions.back());
            
            // Resets the builder
            builder->reset();
        }
        
        template<typename _Scalar, class enable = void> struct Merge;
        
        template<typename _Scalar> struct Merge<_Scalar, typename boost::enable_if<boost::is_complex<_Scalar> >::type> {
            static void merge(KernelEVD<_Scalar>  &merger, const Decomposition<Scalar> &d) {
                ScalarMatrix mY = d.mY * d.mD.asDiagonal();
                merger.add(1, d.mX, d.mY.cwiseSqrt());
            }
        };
        
        template<typename _Scalar> struct Merge<_Scalar, typename boost::disable_if<boost::is_complex<_Scalar> >::type> {
            static void merge(KernelEVD<_Scalar>  &merger, const Decomposition<Scalar> &d) {

                Index posCount = 0;

                for(Index j = 0; j <  d.mD.size(); j++) 
                    if (d.mD(j,0) >= 0) posCount++;
                Index negCount = d.mD.size() - posCount;
                
                // FIXME: block expression for Alt expression
                ScalarMatrix mY = d.mY * d.mD.cwiseAbs().cwiseSqrt().asDiagonal();
                
                Index jPos = 0;                
                ScalarMatrix _mYPos(mY.rows(), posCount);
                for(Index j = 0; j <  d.mD.size(); j++) 
                     if (d.mD(j,0) >= 0)
                         _mYPos.col(jPos++) = mY.col(j);
                ScalarMatrix mYPos;
                mYPos.swap(_mYPos);
                
                jPos = 0;
                ScalarMatrix _mYNeg(mY.rows(), negCount);
                for(Index j = 0; j <  d.mD.size(); j++) 
                    if (d.mD(j,0) < 0)
                        _mYPos.col(jPos++) = mY.col(j);
                ScalarAltMatrix mYNeg;
                mYNeg.swap(_mYNeg);
                
                if (posCount > 0) 
                    merger.add(1, d.mX, mYPos);
                if (negCount > 0) 
                    merger.add(-1, d.mX, mYNeg);
            }
        };

        
        /**
         * Merge the decompositions
         * @param force true if we want to merge all decompositions (i.e. not ensuring that mergings are balanced)
         */
        void merge(bool force) {
            // Merge while the number of merged decompositions is the same for the two last decompositions
            // (or less, to handle the case of previous unbalanced merges)
            while (decompositions.size() >= 2 && (force || (decompositions.back().updateCount >= (decompositions.end()-1)->updateCount))) {
                merger->reset();
                for(int i = 0; i < 2; i++) {
                    const Decomposition<Scalar> &d = decompositions.back();
                    Merge<Scalar>::merge(*merger, d);
                    decompositions.pop_back();
                } 
                
                // Push back new decomposition
                decompositions.push_back(merger->getDecomposition());
                if (mergerCleaner.get())
                    mergerCleaner->cleanup(decompositions.back());
            }
        }
        
        
    private:
        
        //! Vector of decompositions
        std::vector<Decomposition<Scalar> > decompositions;
        
        //! Number of rank updates
        Index batchSize = 100;
        
        boost::shared_ptr<KernelEVD<Scalar>  > builder;
        boost::shared_ptr<Cleaner<Scalar> > builderCleaner;
        
        boost::shared_ptr<KernelEVD<Scalar>  > merger;        
        boost::shared_ptr<Cleaner<Scalar> > mergerCleaner;
    };
}

#ifndef SWIG
#define KQP_SCALAR_GEN(type) extern template class kqp::DivideAndConquerBuilder<type>;
#include <kqp/for_all_scalar_gen.h.inc>
#endif

#endif

