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

#ifndef __KQP_SUBSET_H__
#define __KQP_SUBSET_H__

#include <kqp/kqp.hpp>
#include <kqp/alt_matrix.hpp>


namespace kqp {
    struct Mover {
        bool same;
        Mover(bool same) : same(same) {}
        
        virtual void prepare(Index /*new_size*/) {}
        virtual void cleanup() {}
        
        virtual void disjoint_assign(Index from, Index to, Index size) = 0;   
        
        virtual void assign(Index from, Index to, Index size) {
            if (!same || to + size <= from)
                disjoint_assign(from, to, size);
            else 
                if (from != to) {
                    // Cases where we are the same and from + size > to:
                    // only consider the case where from is different from to.
                    
                    // Move by block
                    Index maxStep = from - to;
                    while (size > 0) {
                        // Choose the step size
                        Index step = size <= maxStep ? size : maxStep;
                        
                        // Move data
                        disjoint_assign(from, to, step);
                        
                        // Update
                        from += step; 
                        to += step;
                        size -= step;
                    }
                    
                }
        }
        
    };
    
    //! Reduces a set of indexed things by block
    inline void selection(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Mover &mover) {
        // Current column for insertions
        size_t current_index = 0;
        
        // Resize
        size_t rank = 0;
        for(std::vector<bool>::const_iterator i = begin; i != end; i++)
            if (*i) rank++;
        
        // Prepare the destination
        mover.prepare(rank);
        
        // Set i on the first eigenvalue
        std::vector<bool>::const_iterator i = std::find(begin, end, true);
        
        while (i != end) {
            // find the next false value
            std::vector<bool>::const_iterator j = std::find(i, end, false);
            
            // copy
            mover.assign(i - begin, current_index, j - i);
            
            // Update the current column index
            current_index += j - i;
            // Position is just after the last false
            i = std::find(j, end, true);
        }
        
        mover.cleanup();
    }    

    
    //! Mover for columns
    template<typename Matrix>
    struct columns : public Mover {
        const Matrix &source;
        Matrix &dest;
        Index new_size;
        
        columns(const Matrix &source, Matrix &dest) : Mover(&source == &dest), source(source), dest(dest), new_size(0) {}
        
        void prepare(Index new_size) {
            this->new_size = new_size;
            if (!same) 
                resize<Matrix>(dest, false, source.rows(), new_size);
        }
        
        void disjoint_assign(Index from, Index to, Index size) {
            copy(source.block(0, from, source.rows(), size), dest.block(0, to, dest.rows(), size));
        }        
        
        void cleanup() {
            if (&source == &dest) 
                resize<Matrix>(dest, true, source.rows(), new_size);
        }
    };
    
    //! Mover for rows
    template<typename Matrix>
    struct rows : public Mover {
        const Matrix &source;
        Matrix &dest;
        Index new_size;
        
        rows(const Matrix &source, Matrix &dest) : Mover(&source == &dest), source(source), dest(dest), new_size(0) {}
        
        void prepare(Index new_size) {
            this->new_size = new_size;
            if (&source != &dest) 
                resize<Matrix>(dest, false, new_size, source.cols());
        }
        
        void disjoint_assign(Index from, Index to, Index size) {
            kqp::copy(source.block(from, 0, size, source.cols()), dest.block(to, 0, size, dest.cols()));
        }       
        
        void cleanup() {
            if (&source == &dest) 
                resize<Matrix>(dest, true, new_size, source.cols());
        }
    };
    
    
    
    //! Select columns of a matrix
    template <typename Derived>
    void select_columns(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end,
                        const Derived &values, Derived &new_values)  {
        kqp::columns<Derived> mover(values, new_values);
        selection(begin, end, mover);
    }
    template <typename Derived>
    void select_columns(const std::vector<bool> &list,  const Derived& values, Derived &new_values) {
        select_columns(list.begin(), list.end(), values, new_values);
    }
    
    //! Select rows of a matrix
    template <typename Derived>
    void select_rows(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end,
                     const Derived& values, Derived &new_values)  {
        kqp::rows<Derived> mover(values, new_values);
        selection(begin, end, mover);
    }
    
    template <typename Derived>
    void select_rows(const std::vector<bool> &list,  const Derived& values, Derived &new_values) {
        select_rows(list.begin(), list.end(), values, new_values);
    }

    
    
    
    //! Reduces the eigenpair given the current selection pattern
    template<typename Scalar>
    void value_selection(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, 
                         const Eigen::Matrix<Scalar,Dynamic,1> &values, const Eigen::Matrix<Scalar,Dynamic,1> &new_values)  {
        selection(begin, end, values, new_values);
    }
    
  }
#endif
