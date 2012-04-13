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

#ifndef __KQP_RANK_SELECTOR_H__
#define __KQP_RANK_SELECTOR_H__

#include <Eigen/Core>
#include <kqp/kqp.hpp>

namespace kqp {
    /**
     * A list of eigenvalues that can be edited
     */
    template<typename Scalar>
    class EigenList {
    public:
        virtual ~EigenList();
        
        /**
         * Select an eigenvalue
         */
        virtual Scalar get(Index i) const = 0;
        
        /**
         * Remove this eigenvalue from the selection
         */
        virtual void remove(Index i) = 0;
        
        /**
         * The original number of eigenvalues
         */
        virtual Index size() const = 0;
        
        /**
         * The current number of selected
         */
        virtual Index getRank() const = 0;
        
        /**
         * Check if an eigenvalue is currently selected or not
         */
        virtual bool isSelected(size_t i) const = 0;
    };

    
    template<typename Scalar, bool absolute>
    struct EigenListComparator {
        const EigenList<Scalar> &list;
        EigenListComparator(const EigenList<Scalar> &list) : list(list) {}
        bool operator() (int i1, int i2) {
            if (absolute) return std::abs(list.get(i1)) < std::abs(list.get(i2));
            return list.get(i1) < list.get(i2);
        }
    };


    
    template<typename Scalar>
    class DecompositionList : public EigenList<Scalar> {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        
        
        DecompositionList(const Vector &eigenvalues) 
        : eigenvalues(eigenvalues), selected(eigenvalues.size(), true), rank(eigenvalues.size()) {
            
        }
        
        /**
         * Select an eigenvalue
         */
        virtual Scalar get(Index i) const override {
            return eigenvalues[i];
        }
        
        /**
         * Remove this eigenvalue from the selection
         */
        virtual void remove(Index i) override {
            if (selected[i]) {
                selected[i] = false;
                rank--;
            }
        }
        
        /**
         * The original number of eigenvalues
         */
        virtual Index size() const override {
            return eigenvalues.size();
        }
        
        /**
         * The current number of selected
         */
        virtual Index getRank() const override {
            return rank;
        }
        
        /**
         * Check if an eigenvalue is currently selected or not
         */
        virtual bool isSelected(size_t i) const override { 
            return selected[i];
        }
        
        const std::vector<bool> &getSelected() const override {
            return selected;
        }
        
    private:
        
        Vector eigenvalues;
        std::vector<bool> selected;
        Index rank;
    };
    
        
    /**
     * Gets an eigenlist and removes whatever eigenvalues it does not like
     */
    template<typename Scalar>
    class Selector {
    public:
        /**
         * @param eigenValues
         *            The ordered list of eigenvalues
         */
        virtual void selection(EigenList<Scalar>& eigenvalues) const = 0;
    };
    
    /**
     * Chain selection
     */
    template<typename Scalar>
    class ChainSelector : public Selector<Scalar> {
        std::vector<boost::shared_ptr<Selector<Scalar>>> selectors;
    public:
        ChainSelector() {}
        void add(const boost::shared_ptr<Selector<Scalar>> &selector) {
            selectors.push_back(selector);
        }
        
        virtual void selection(EigenList<Scalar>& eigenvalues) const override {
            for(auto i = selectors.begin(), end = selectors.end(); i != end; i++)
                (*i)->selection(eigenvalues);
        }
    };
    
    /**
     * Minimum relative eigenvalue
     */
    template<typename Scalar>
    class MinimumSelector : public Selector<Scalar> {
        Scalar minRatio;
    public:
        MinimumSelector() : minRatio(EPSILON) {}
        virtual void selection(EigenList<Scalar>& eigenvalues) const override {
            // Computes the maximum of eigenvalues
            Scalar maxLambda = 0;
            for(Index i = 0; i < eigenvalues.size(); i++) 
                if (eigenvalues.isSelected(i)) 
                    maxLambda = std::max(maxLambda, eigenvalues.get(i));
            
            // Remove those above the maximum * ratio
            Scalar threshold = maxLambda * minRatio;
            for(Index i = 0; i < eigenvalues.size(); i++) 
                if (eigenvalues.isSelected(i)) 
                    if (eigenvalues.get(i) < threshold)
                        eigenvalues.remove(i);
   
        }
    };
    
    /**
     * Select the highest eigenvalues (with a possible "reset" rank)
     */
    template<typename Scalar, bool byMagnitude>
    class RankSelector : public Selector<Scalar> {
        //! Maximum rank
        Index maxRank;
        
        //! Rank to select when the rank is above maxRank
        Index resetRank;
    public:
        /**
         * Selects the highest eigenvalues (either magnitude or values)
         * @param rank The maximum and selected rank
         * @param byMagnitude If true, then eigenvalues will be sorted by absolute value
         */
        RankSelector(Index maxRank) : maxRank(maxRank), resetRank(maxRank) {}
        
        /**
         * Construct a selector that uses a reset rank
         */
        RankSelector(Index maxRank, Index resetRank) : maxRank(maxRank), resetRank(resetRank) {
            if (resetRank > maxRank)
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "The maximum rank (%d) should be greater or equal to the reset rank (%d)", %maxRank %resetRank);
        }
        
        void selection(EigenList<Scalar>& eigenvalues) const override {
            // exit if we have nothing to do
            if (eigenvalues.getRank() <= maxRank) return;
            
            // Copy the values
            std::vector<Scalar> values;
            values.reserve(eigenvalues.getRank());
            for(Index i = 0; i < eigenvalues.size(); i++) 
                if (eigenvalues.isSelected(i)) 
                    values.push_back(i);
            
            // Sort and select
            std::sort(values.begin(), values.end(), EigenListComparator<Scalar, byMagnitude>(eigenvalues)); 

            // Select the rank highest eigenvalues
            for(Index i = 0; i < eigenvalues.size() - resetRank; i++) 
                eigenvalues.remove(values[i]);
        }
    };
}

#endif
