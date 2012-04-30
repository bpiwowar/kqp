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
 
#ifndef __KQP_INTERVALS_H__
#define __KQP_INTERVALS_H__

#include <vector>

//! An interval iterator
struct IntervalsIterator {
    const std::vector<bool> &which;
    std::pair<size_t, size_t> current;
    
    IntervalsIterator &operator++(int) { 
        current.first  = std::find(current.second + 1 + which.begin(), which.end(), true) - which.begin();
        current.second = std::find(which.begin() + current.first, which.end(), false) - which.begin();
        return *this;
    }
    
    IntervalsIterator(const std::vector<bool> &which) : which(which), current(0,0) {
        (*this)++;
    }
    IntervalsIterator(const std::vector<bool> &which, size_t begin, size_t end) : which(which), current(begin,end) {
    }
    const std::pair<size_t, size_t> & operator*() const {
        return current;
    }
    const std::pair<size_t, size_t> *operator->() const {
        return &current;
    }
    bool operator!=(const IntervalsIterator &other) {
        return &which != &other.which || current != other.current;
    }
};


struct Intervals {
    std::vector<bool> which;
    size_t _selected;
    
    typedef IntervalsIterator Iterator;
    
    const Iterator _end;
    
    Intervals(const std::vector<bool> &which) : which(which), _end(which, which.size(), which.size()) {             
        _selected = std::accumulate(which.begin(), which.end(), 0);
    }
    size_t size() const { return which.size(); }
    size_t selected() const { return _selected; }
    
    Iterator begin() { 
        return Iterator(which);
    }
    const Iterator &end() { 
        return _end;
    }
};

#endif
