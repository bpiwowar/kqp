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

#include <boost/type_traits/is_complex.hpp>
#include <kqp/cleanup.hpp>
#include <kqp/kernel_evd.hpp>

namespace kqp {

#   include <kqp/define_header_logger.hpp>
    DEFINE_KQP_HLOGGER("kqp.kevd.dc");

    /**
     * @brief Meta builder: Uses other operator builders and combine them at regular intervals.
     * @ingroup KernelEVD
     */
    template <typename Scalar> class DivideAndConquerBuilder : public KernelEVD<Scalar>  {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);        
        
        DivideAndConquerBuilder(const FSpace &fs) : KernelEVD<Scalar>(fs), batchSize(100) {}
        
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
            decompositions.clear();
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
            Decomposition<Scalar> &d = decompositions.back();
            if (builderCleaner.get())
                builderCleaner->cleanup(d);
            assert(!kqp::isNaN(d.fs->k(d.mX, d.mY, d.mD).squaredNorm()));
            
            // Resets the builder
            builder->reset();
        }
        
        //! Add a decomposition to a merger
        static void merge(KernelEVD<Scalar> &merger, const Decomposition<Scalar> &d) {
            Index posCount = 0;
            
            for(Index j = 0; j <  d.mD.size(); j++) 
                if (d.mD(j,0) >= 0) posCount++;
            Index negCount = d.mD.size() - posCount;
            
            // FIXME: block expression for Alt expression
            ScalarMatrix mY = d.mY * d.mD.cwiseAbs().cwiseSqrt().asDiagonal();
            
            Index jPos = 0;                
            Index jNeg = 0;
            
            ScalarMatrix mYPos(mY.rows(), posCount);
            ScalarMatrix mYNeg(mY.rows(), negCount);
            
            for(Index j = 0; j <  d.mD.size(); j++) 
                if (d.mD(j,0) >= 0)
                    mYPos.col(jPos++) = mY.col(j);
                else
                    mYNeg.col(jNeg++) = mY.col(j);
            
            assert(jPos == posCount);
            assert(jNeg == negCount);
            
            Real posNorm = d.fs->k(d.mX, mYPos).norm();
            Real negNorm = d.fs->k(d.mX, mYNeg).norm();
            if (posCount > 0 && posNorm / negNorm > Eigen::NumTraits<Scalar>::epsilon()) 
                merger.add(1, d.mX, mYPos);
            if (negCount > 0 && negNorm / posNorm > Eigen::NumTraits<Scalar>::epsilon()) 
                merger.add(-1, d.mX, mYNeg);
        }

        
        /**
         * Merge the decompositions
         * @param force true if we want to merge all decompositions (i.e. not ensuring that mergings are balanced)
         */
        void merge(bool force) {
            // Merge while the number of merged decompositions is the same for the two last decompositions
            // (or less, to handle the case of previous unbalanced merges)
            while (decompositions.size() >= 2 && (force || (decompositions.back().updateCount >= (decompositions.end()-2)->updateCount))) {
                
                Decomposition<Scalar> d1 = std::move(decompositions.back());
                decompositions.pop_back();
                Decomposition<Scalar> d2 = std::move(decompositions.back());
                decompositions.pop_back();

                merger->reset();
                merge(*merger, d1);
                merge(*merger, d2);
                
                // Push back new decomposition
                decompositions.push_back(merger->getDecomposition());
                auto &d = decompositions.back();
                assert(!kqp::isNaN(d.fs->k(d.mX, d.mY, d.mD).squaredNorm()));
                if (mergerCleaner.get())
                    mergerCleaner->cleanup(d);
                d.updateCount = d1.updateCount + d2.updateCount;
                assert(!kqp::isNaN(d.fs->k(d.mX, d.mY, d.mD).squaredNorm()));

                KQP_HLOG_INFO_F("Merged two decompositions [%d/%d;%d] and [%d/%d;%d] into [rank= %d, pre-images=%d; updates=%d]", 
                                 %d1.mD.rows() %d1.mX->size() %d1.updateCount 
                                 %d2.mD.rows() %d2.mX->size() %d2.updateCount 
                                 %d.mD.rows()  %d.mX->size()  %d.updateCount);
            }
        }
        
        
    private:
        
        //! Vector of decompositions
        std::vector<Decomposition<Scalar> > decompositions;
        
        //! Number of rank updates
        Index batchSize;
        
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

