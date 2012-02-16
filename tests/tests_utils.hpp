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

#ifndef _KQP_KERNEL_EVD_TESTS_UTILS_H_
#define _KQP_KERNEL_EVD_TESTS_UTILS_H_

#include <cassert>

#include <Eigen/Core>
#include <Eigen/QR>

#include "kqp.hpp"

namespace kqp {

    /** 
     * Generate an hermitian matrix of a given rank in given space dimension
     * 
     * @param dim The dimension of the space
     * @param rank The rank of the matrix
     */
    template<typename Scalar> Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> generateMatrix(Index dim, Index rank, Scalar min = (Scalar)1, Scalar max = (Scalar)10) {
        // Initialisation
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        assert(dim >= rank);
        
        // First, get a unary matrix through QR decomposition
        Eigen::FullPivHouseholderQR<Matrix> qr(Matrix::Random(dim, dim));
        Matrix matrix = qr.matrixQ();

        // Creates a diagonal matrix of the specified dimension
        Vector diagonal = Vector::Zero(dim);
        for(Index i = 0; i < rank; i++)
            diagonal(i) = Eigen::internal::random(min, max);
        
        // Returns the matrix
        return matrix * diagonal.asDiagonal() * matrix.transpose();
        
    }
}

#endif