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
    #include <kqp/feature_matrix/sparse.hpp>
    #include <kqp/feature_matrix/sparse_dense.hpp>
    #include <kqp/feature_matrix/unary_kernel.hpp>
    
    #include <kqp/kernel_evd/dense_direct.hpp>
    #include <kqp/kernel_evd/accumulator.hpp>
    #include <kqp/kernel_evd/incremental.hpp>
    #include <kqp/kernel_evd/divide_and_conquer.hpp>
    
    #include <kqp/probabilities.hpp>

    namespace kqp {
        namespace _AltMatrix { enum AltMatrixType { DENSE, IDENTITY }; }
        namespace _AltVector { enum AltVectorType { DENSE, CONSTANT }; }
    }
    
    using Eigen::Dynamic;
    using Eigen::Matrix;
%}

// --- Defines some macros

#define Index long
#define override 
#define DEFINE_KQP_HLOGGER(x)

// Preserve the arguments
#define %kqparg(X...) X

// --- Language dependent includes

%include "boost_shared_ptr.i"
%include "std_vector.i"

#if SWIGJAVA
%include "java_kqp.i"
#endif

#ifdef SWIGPYTHON
%include <pycontainer.swg>
#endif

// --- STL related types

%template(BoolArrayList) std::vector<bool>;
%template(IndexArrayList) std::vector<Index>;

// ---- Command renaming

%rename operator++ next;
%rename operator!= notEqual;

%ignore kqp::Intervals;
%ignore kqp::IntervalsIterator;
%ignore kqp::DenseMatrix::getMatrix;
%ignore kqp::LinearCombination;

// ---- Some basic declarations

namespace kqp {
    namespace _AltMatrix { enum AltMatrixType { DENSE, IDENTITY }; }
    namespace _AltVector { enum AltVectorType { DENSE, CONSTANT }; }
    
    template<typename Scalar> struct ScalarDefinitions;
        
    //! A dense vector or a constant vector
     template<typename Scalar> struct AltVector {
         typedef Eigen::Matrix<Scalar,Dynamic,1>  VectorType;
         typedef typename Eigen::Matrix<Scalar,Dynamic,1>::ConstantReturnType ConstantVectorType;
     };


     //! Dense or Identity matrix
     template<typename Scalar> struct AltDense {
         typedef Eigen::Matrix<Scalar,Dynamic,Dynamic> DenseType;
         typedef typename Eigen::MatrixBase< Eigen::Matrix<Scalar,Dynamic,Dynamic> >::IdentityReturnType IdentityType;
     };
         
    template<typename T1,typename T2>
    class AltMatrix {
    };
}

namespace Eigen {
    template<typename Scalar> struct NumTraits;
    
    template<typename Scalar, int Rows, int Cols>
    class Matrix {
    public:
        Matrix(Index rows, Index cols);
    };
    
    template<typename Scalar, int Major>
    class SparseMatrix {
    public:
        SparseMatrix(Index rows, Index cols);
    };
    
    
}


%include "kqp_all.i"
