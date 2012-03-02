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
  <kqp/kqp.hpp>feature_matrix/dense.hpp"
  <kqp/kqp.hpp>kernel_evd/dense_direct.hpp"
%}

#define Index int
#define KQP_FOR_ALL_SCALAR_TYPES(prefix, suffix)

%define ALL_SCALARS_DECLARATION(javaname, classname)
%template(Double ## javaname) classname <double >;
%enddef

%define FOR_ALL_SCALAR(prefix, suffix)
prefix ## double ## suffix;
%enddef


%ignore kqp::DenseMatrix::get_matrix;
%include "feature_matrix/dense.hpp"
ALL_SCALARS_DECLARATION(DenseFeatureMatrix, kqp::DenseMatrix)


%include "kernel_evd/dense_direct.hpp"
ALL_SCALARS_DECLARATION(DenseDirectBuilder, kqp::DenseMatrix)
