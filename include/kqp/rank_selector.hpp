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
    template<class Scalar>
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
        virtual void selection(EigenList<Scalar>& eigenValues) const = 0;
        
    };
    
    /**
     * Chain selection
     */
    template<typename Scalar>
    class ChainSelector : public Selector<Scalar> {
    public:
        ChainSelector();
        void add(const ChainSelector<Scalar> &);
        virtual void selection(EigenList<Scalar>& eigenValues) const;
    };
    
    /**
     * Minimum relative eigenvalue
     */
    template<typename Scalar>
    class MinimumSelector {
    public:
        MinimumSelector();
        virtual void selection(EigenList<Scalar>& eigenValues) const;
    };
}

#endif
