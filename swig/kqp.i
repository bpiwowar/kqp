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

%module kqp


%{
    #include <kqp/cleanup.hpp>

    #include <kqp/feature_matrix/dense.hpp>
    
    #include <kqp/kernel_evd/dense_direct.hpp>
    #include <kqp/kernel_evd/accumulator.hpp>
    #include <kqp/kernel_evd/incremental.hpp>
    #include <kqp/kernel_evd/divide_and_conquer.hpp>
    
    #include <kqp/probabilities.hpp>

    #include "alt_matrix_swig.hpp"
%}

// --- Defines some macros

#define Index long
#define override 
#define DEFINE_KQP_HLOGGER(x)

// Preserve the arguments
#define %kqparg(X...) X

// --- Language dependent includes

%include "boost_shared_ptr.i"

#if SWIGJAVA
%include "java/kqp.i"
#endif

#ifdef SWIGPYTHON
%include <pycontainer.swg>
%import "python/std_vector.i"
#endif

// --- STL related types

%template(BoolArrayList) std::vector<bool>;


// ---- Command renaming

%rename operator++ next;
%rename operator!= notEqual;

namespace kqp {}
namespace Eigen {}

%ignore kqp::FeatureMatrix;
%ignore kqp::Intervals;
%ignore kqp::IntervalsIterator;
%ignore kqp::DenseMatrix::getMatrix;
%ignore kqp::LinearCombination;

namespace Eigen {
    template<typename Scalar, int Rows, int Cols>
    class Eigen::Matrix {
    public:
        Matrix(Index rows, Index cols);
    };
}

%include "kqp_all.i"