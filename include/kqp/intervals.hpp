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
namespace kqp {
  #   include <kqp/define_header_logger.hpp>
      DEFINE_KQP_HLOGGER("kqp.intervals");
  
//! An interval iterator
struct IntervalsIterator {
    const std::vector<bool> *which;
    size_t size;
    std::pair<size_t, size_t> current;

    void find(size_t fromInclusive) {
      if (which) {
        // Find the first "true" bit from current position
        current.first  = std::find(which->begin() + fromInclusive, which->end(), true) - which->begin();
        
        // Find the first "false" bit from start
        current.second = current.first == size ? size : std::find(which->begin() + current.first + 1, which->end(), false) - which->begin() - 1;
      } else {
        if (fromInclusive > 0) 
          current.first = current.second = size;
        else {
          current.first = 0;
          current.second = size - 1;
        }
      }
      
    }
    
    IntervalsIterator &operator++(int) {
        find(current.second+1);
        return *this;
    }
    
    IntervalsIterator(const std::vector<bool> *which, size_t size) : which(which), size(size) {
      find(0);
    }
    
    IntervalsIterator(const std::vector<bool> *which, size_t size, size_t begin, size_t end) : which(which),  size(size), current(begin,end) {
    }
    
    const std::pair<size_t, size_t> & operator*() const {
        return current;
    }
    const std::pair<size_t, size_t> *operator->() const {
        return &current;
    }
    bool operator!=(const IntervalsIterator &other) {
        return which != other.which || current != other.current;
    }
};


class Intervals {
public:
    typedef IntervalsIterator Iterator;

    Intervals(const std::vector<bool> &which) : m_which(&which), m_size(which.size()), m_end(m_which, m_size, m_size, m_size) {             
      m_selected = std::accumulate(which.begin(), which.end(), 0);
    }
    
    Intervals(const std::vector<bool> *which, size_t size) : m_which(which), m_size(size), m_end(m_which, m_size, m_size, m_size) {             
        assert(!which || which->size() == m_size);
        if (which) m_selected = std::accumulate(which->begin(), which->end(), 0);
        else m_selected = m_size;
    }
    
    size_t size() const { return m_size;}
    size_t selected() const { return m_selected; }
    
    Iterator begin() { 
        return Iterator(m_which, m_size);
    }
    const Iterator &end() { 
        return m_end;
    }

  private:          
      // Vector of selected entries
      const std::vector<bool> *m_which;

      // Number of entries
      size_t m_size;

      // End for the iterator
      const Iterator m_end;

      // Number of selected entries
      size_t m_selected;

      
};
}
#endif
