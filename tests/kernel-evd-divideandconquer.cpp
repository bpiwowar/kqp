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
#include "kernel-evd-tests.hpp"

#include <kqp/kernel_evd/divide_and_conquer.hpp>
#include <kqp/kernel_evd/dense_direct.hpp>
DEFINE_LOGGER(logger, "kqp.test.kernel_evd.divide-and-conquer")

namespace kqp {
    namespace kevd_tests {  
        int DivideAndConquer::run(const Dense_evd_test &test) const {
            // We want at least 3 merges
            DivideAndConquerBuilder<double> builder(DenseFeatureSpace<double>::create(test.n));
            
            builder.setBatchSize(test.nb_add * test.min_preimages / 3);
            
            builder.setBuilder(boost::shared_ptr<DenseDirectBuilder<double>>(new DenseDirectBuilder<double>(test.n)));
            builder.setMerger(boost::shared_ptr<DenseDirectBuilder<double>>(new DenseDirectBuilder<double>(test.n)));
            
            return test.run(logger, builder);
        }
    }
}