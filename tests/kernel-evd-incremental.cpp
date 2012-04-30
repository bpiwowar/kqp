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
#include <kqp/kernel_evd/incremental.hpp>

DEFINE_LOGGER(logger, "kqp.test.kernel_evd.incremental")

namespace kqp {
    namespace kevd_tests {        
        int Incremental::run(const Dense_evd_test &test) const {
            IncrementalKernelEVD< double > builder(DenseFeatureSpace<double>::create(test.n));
            return test.run(logger, builder);
        }
    }
}