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

#ifndef __KQP_KERNEL_EVD_UTILS_H__
#define __KQP_KERNEL_EVD_UTILS_H__

#include <cassert>

#include <Eigen/Eigenvalues>

#include <Eigen/Core>

#include <kqp/kernel_evd.hpp>
#include <kqp/feature_matrix/dense.hpp>

namespace kqp {
    template <class Derived> 
    void thinEVD(const Eigen::SelfAdjointEigenSolver<Derived> &evd, Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors, 
                 Eigen::Matrix<typename Eigen::NumTraits<typename Derived::Scalar>::Real, Eigen::Dynamic, 1> &eigenvalues,
                 Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> *nullEigenvectors = 0
                 ) {
        typedef typename Derived::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        // We expect eigenvalues to be sorted by increasing order
        Index dimension = evd.eigenvectors().rows();
        
        const Eigen::Matrix<Real, Eigen::Dynamic, 1> &d = evd.eigenvalues();
        Real threshold = EPSILON * (Real)d.size();
        
        Index n = d.rows();
        Index negatives = 0, zeros = 0;
        
        for(Index i = 0; i < n; i++) {
            assert(i==0 || d[i-1] <= d[i]);
            if (-d[i] > threshold) negatives++;
            else if (d[i] < threshold) zeros++; 
            else break;
        }
        
        Index positives = n - negatives - zeros;
        
        eigenvalues.resize(positives+negatives);
        eigenvalues.head(negatives) = d.head(negatives);
        eigenvalues.tail(positives) = d.tail(positives);
        
        eigenvectors.resize(dimension, positives + negatives);
        eigenvectors.leftCols(negatives) = evd.eigenvectors().leftCols(negatives);
        eigenvectors.rightCols(positives) = evd.eigenvectors().rightCols(positives);
        
        if (nullEigenvectors) {
            *nullEigenvectors = evd.eigenvectors().block(0, negatives, dimension, zeros);
        }
    }
    
}

#endif