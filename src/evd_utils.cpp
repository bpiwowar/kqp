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

#include <kqp/evd_utils.hpp>

#define KQP_SCALAR_GEN(scalar) \
    template struct kqp::Orthonormalize<scalar>; \
    template struct kqp::ThinEVD<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>>
#include <kqp/for_all_scalar_gen.h.inc>
