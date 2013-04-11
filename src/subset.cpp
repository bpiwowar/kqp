#include <kqp/subset.hpp>

DEFINE_LOGGER(logger, "kqp.subset")

namespace kqp {
    void selection(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Mover &mover) {
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
}
