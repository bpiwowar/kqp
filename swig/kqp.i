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

    #include "eigenmatrix.hpp"
    #include "alt_matrix_swig.hpp"
%}

// --- Language dependent includes
#if SWIGJAVA
%import "java/std_vector.i"
%ignore operator=;
#endif

#ifdef SWIGPYTHON
%include <pycontainer.swg>
%import "python/std_vector.i"
#endif

// --- STL related types

%template(BoolArrayList) std::vector<bool>;

// --- Defines some macros

#define Index long
#define override 
#define DEFINE_KQP_HLOGGER(x)

// ---- Command renaming

%rename operator++ next;
%rename operator!= notEqual;
// Namespaces 

namespace kqp {}
namespace Eigen {}

// ---- Macros to define for all types (scalars and feature matrix)

%define FOR_ALL_SCALARS_2(prefix, suffix, type, _scalar, _Real, _real)
#define _SWIG_SCALAR_GEN_AUX(__scalar__) _SWIG_SCALAR_GEN_(prefix ## suffix, type, _Real, _real)
_SWIG_SCALAR_GEN_AUX(_scalar)
#undef _SWIG_SCALAR_GEN_AUX
%enddef

%define FOR_ALL_SCALARS_1(prefix, type)
FOR_ALL_SCALARS_2(prefix, Double, type, double, Double, double)
%enddef


%define FOR_ALL_SCALARS(expression)
#define _SWIG_SCALAR_GEN_(SNAME,STYPE,RNAME,RTYPE) expression
FOR_ALL_SCALARS_1(/**/, __scalar__)
#undef _SWIG_SCALAR_GEN_
%enddef

%define FOR_ALL_FMATRIX(expression)
#define _SWIG_SCALAR_GEN_(FNAME,FTYPE,RNAME,RTYPE) expression
FOR_ALL_SCALARS_1(Dense, kqp::DenseMatrix<__scalar__>)
#undef _SWIG_SCALAR_GEN_
%enddef

// --- Eigen 
namespace Eigen {
    
    template<typename T> struct NumTraits;
    template<> struct NumTraits<double> {
      typedef double Real;  
    };
    template<> struct NumTraits< kqp::ftraits< kqp::DenseMatrix<double> > > {
      typedef double Real;  
    };
    
}
FOR_ALL_SCALARS(%template() Eigen::NumTraits< STYPE >;)


// --- Eigen Dense Matrix

%include "eigenmatrix.hpp"
%ignore Eigen::Matrix;
FOR_ALL_SCALARS(%template(SNAME ## EigenMatrix) kqp::swig::EigenMatrix< STYPE >);

// --- Alt vectors

%include "alt_matrix_swig.hpp"

FOR_ALL_SCALARS(%template(SNAME ## AltMatrix) kqp::swig::AltMatrix<STYPE>;)
FOR_ALL_SCALARS(%template(SNAME ## AltVector) kqp::swig::AltVector<STYPE>;)

// --- Feature matrices

%ignore kqp::FeatureMatrix;
%ignore kqp::Intervals;
%ignore kqp::IntervalsIterator;
%ignore kqp::DenseMatrix::getMatrix;
%ignore kqp::LinearCombination;
%include <kqp/feature_matrix.hpp>
%include <kqp/feature_matrix/dense.hpp>


FOR_ALL_SCALARS(namespace kqp { \
    template<> struct ftraits< DenseMatrix<STYPE> > { \
        typedef STYPE Scalar; typedef RTYPE Real; \
        typedef kqp::swig::AltMatrix<STYPE> ScalarAltMatrix;  \
        typedef kqp::swig::AltVector<RTYPE> RealAltVector; \
        }; \
    } \
    %template() kqp::ftraits< kqp::DenseMatrix< STYPE > >;
)

FOR_ALL_SCALARS(%template() kqp::FeatureMatrix< kqp::DenseMatrix< STYPE > >;)
FOR_ALL_SCALARS(%template(SNAME ## DenseFeatureMatrix) kqp::DenseMatrix< STYPE >;)

// ---- Kernel EVD

%include "kqp/rank_selector.hpp"
%include "kqp/decomposition.hpp"
%include "kqp/cleanup.hpp"
FOR_ALL_SCALARS(%template(SNAME ## RankSelector) kqp::Selector< STYPE >;)
FOR_ALL_FMATRIX(%template(FNAME ## Cleaner) kqp::Cleaner< FTYPE >;)
FOR_ALL_FMATRIX(%template(FNAME ## StandardCleaner) kqp::StandardCleaner< FTYPE >;)

FOR_ALL_FMATRIX(%template(FNAME ## Decomposition) kqp::Decomposition< FTYPE >;)

%include "kqp/kernel_evd.hpp"
FOR_ALL_FMATRIX(%template(FNAME ## KEVD) kqp::KernelEVD< FTYPE >;)

%include "kqp/kernel_evd/dense_direct.hpp"
FOR_ALL_SCALARS(%template(Dense ## SNAME ## KEVDDirect) kqp::DenseDirectBuilder< STYPE >;)

%include "kqp/kernel_evd/accumulator.hpp"
#define %kqparg(X...) X
FOR_ALL_FMATRIX(%kqparg(%template(FNAME ## KEVDAccumulator) kqp::AccumulatorKernelEVD< FTYPE, true >;))

%include "kqp/kernel_evd/incremental.hpp"
FOR_ALL_FMATRIX(%template(FNAME ## KEVDIncremental) kqp::IncrementalKernelEVD< FTYPE >;)

%include "kqp/kernel_evd/divide_and_conquer.hpp"
FOR_ALL_FMATRIX(%template(FNAME ## KEVDDivideAndConquer) kqp::DivideAndConquerBuilder< FTYPE >;)

// ---- Probabilities

%include "kqp/probabilities.hpp"
FOR_ALL_FMATRIX(%template(FNAME ## KernelOperator) kqp::KernelOperator< FTYPE >;)
FOR_ALL_FMATRIX(%template(FNAME ## Event) kqp::Event< FTYPE >;)
FOR_ALL_FMATRIX(%template(FNAME ## Density) kqp::Density< FTYPE >;)
