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

#ifndef __KQP_REDUCED_SET_LDL_APPROACH_H__
#define __KQP_REDUCED_SET_LDL_APPROACH_H__

#include <algorithm> // sort

#include <Eigen/LU> // LU decomposition

#include <kqp/feature_matrix.hpp>
#include <kqp/subset.hpp>
#include <kqp/reduced_set/unused.hpp>

namespace kqp {
    
    template <class ComparableArray, typename Index = int>
    struct IndirectSort {
        const ComparableArray &array;
        IndirectSort(const ComparableArray &array) : array(array) {}
        bool operator() (int i,int j) { return (array[i] < array[j]);}
    };
    
    /**
     * @brief Removes pre-images with the null space method
     * 
     * Removes pre-images using the null space method.
     * We have \f$ X Z  = 0\f$ and want to find \f$X^\prime\f$ and \f$A\f$ such that
     * \f$ X = (\begin{array}{cc} X^\prime & X^\prime A \end{array} )
     * 
     * @param mF (\b in) the feature matrix \f$X\f$ (\b out) the reduced feature matrix \f$X^\prime\f$
     * @param kernel (in) the null space basis \f$Z\f$ for the matrix \ref mF (out) the matrix \f$A\f$ such that \f$ X^\prime A = X^{\prime\prime} \f$
     * @param mP (out) Permutation matrix so that \f$X P = (X^\prime X^{\prime\prime})\f$
     * @param weights give an order to the different pre-images
     * @param delta
     */
    template <class FMatrix>
    void removeNullSpacePreImages(FMatrix &mF, typename ftraits<FMatrix>::ScalarMatrix &kernel, 
                                  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index>& mP, const typename ftraits<FMatrix>::RealVector &weights, double delta = 1e-4) {
        typedef typename ftraits<FMatrix>::ScalarMatrix ScalarMatrix;
        typedef typename ftraits<FMatrix>::ScalarVector ScalarVector;
        typedef typename ftraits<FMatrix>::RealVector RealVector;
        typedef typename ftraits<FMatrix>::Real Real;
        typedef typename Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> Permutation;

        // --- Look up at the indices to remove
        
        // Get the pre-images available for removal (one per vector of the null space, i.e. by columns in kernel)
        // Summarize the result per pre-image (number of vectors where it appears)
        Eigen::Matrix<int, Eigen::Dynamic, 1> num_vectors = (kernel.array().abs() > kernel.array().abs().colwise().maxCoeff().replicate(kernel.rows(), 1) * delta).template cast<int>().rowwise().sum();
        
        std::vector<Index> to_remove;
        for(Index i = 0; i < num_vectors.rows(); i++)
            if (num_vectors[i] > 0) 
                to_remove.push_back(i);
        
        // Sort        
        const Index pre_images_count = kernel.rows();
        const Index remove_size = kernel.cols();
        const Index keep_size = mF.size() - remove_size;
        
        mP = Permutation(mF.size());
        mP.setIdentity();

        std::sort(to_remove.begin(), to_remove.end(), IndirectSort<RealVector>(weights));
        std::vector<bool> selection(mF.size(), true);
        std::vector<bool> used(remove_size, false);

        ScalarVector v;
        
        // --- Remove the vectors one by one (Pivoted Gauss)

        for(size_t index = 0;  index < remove_size; index++) {
            // remove the ith pre-image
            size_t i = to_remove[index];
            
            // Searching for the highest magnitude
            Index j = -1;
            Real max = 0;
            for(Index k = 0; k < kernel.cols(); k++) {
                Real x = kernel.array().abs()(i, k);
                if (!used[k] && x > max) {
                    max = x;
                    j = k;
                }
            }
            assert(j != -1);
            used[j] = true;
            
            // Update permutation
            selection[i] = false;            
            mP.indices()(i) = j + keep_size;

            // Update the matrix
            v = kernel.col(j) / kernel(i,j);
            kernel.col(j) /= -kernel(i,j);
            kernel(i,j) = 0;
            
            kernel = ((ScalarMatrix::Identity(pre_images_count, pre_images_count) 
                      - v * ScalarVector::Unit(pre_images_count, i).transpose()) * kernel).eval();
        }
        
        
        // --- Remove the vectors from mF and set the permutation matrix

        // Remove unuseful vectors
        mF.subset(selection.begin(), selection.end());
        select_rows(selection.begin(), selection.end(), kernel, kernel);

        // Complete the permutation matrix
        Index count = 0;
        for(size_t index = 0; index < selection.size(); index++) 
            if (selection[index]) {
                mP.indices()(index) = count++;
            }

    }
    
    /**
     * @brief Removes unuseful pre-images 
     *
     * 1. Removes unused pre-images 
     * 2. Computes a \f$LDL^\dagger\f$ decomposition of the Gram matrix to find redundant pre-images
     * 3. Removes newly unused pre-images
     */
    template <class FMatrix>
    void removePreImagesWithNullSpace(FMatrix &mF, typename ftraits<FMatrix>::ScalarAltMatrix &mY) {
        typedef typename ftraits<FMatrix>::Scalar Scalar;
        typedef typename ftraits<FMatrix>::Real Real;
        typedef typename ftraits<FMatrix>::ScalarMatrix ScalarMatrix;
        typedef typename ftraits<FMatrix>::RealVector RealVector;
        
        // Removes unused pre-images
        removeUnusedPreImages(mF, mY);
        
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        // LDL decomposition (stores the L^T matrix)
        Eigen::FullPivLU<ScalarMatrix> lu_decomposition(mF.inner());
        
        // Stop if full rank
        if (lu_decomposition.rank() == N) 
            return;
        
        // Remove pre-images using the kernel
        ScalarMatrix kernel = lu_decomposition.kernel();
        RealVector weights = mY.rowwise().squaredNorm().array() * mF.inner().diagonal().array().abs();
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> mP;
        removeNullSpacePreImages(mF, kernel, mP, weights);
        
        // Y <- (Id A) P Y
        ScalarMatrix mY2(mY);
        mY2= (mP * mY2).topRows(mF.size()) + kernel * (mP * mY2).bottomRows(mY.rows() - mF.size());
        mY.swap(mY2);
        
        // Removes unused pre-images
        removeUnusedPreImages<FMatrix>(mF, mY);
    }
    
    template <class FMatrix>
    void removePreImagesWithNullSpace(FMatrix &mF, typename ftraits<FMatrix>::ScalarMatrix &mY) {
        typename ftraits<FMatrix>::ScalarAltMatrix _mY;
        _mY.swap(mY);
        removePreImagesWithNullSpace(mF, _mY);
        _mY.swap(mY);
    }
    
    
}

#endif
