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

#ifndef __KQP_NULL_SPACE_H__
#define __KQP_NULL_SPACE_H__

#include <Eigen/Dense>

#include "feature_matrix.hpp"
#include "coneprog.hpp"

namespace kqp {
    
    /**
     * @brief Removes pre-images with the null space method
     */
    template <class FVector> 
    void removeUnusedPreImages(FeatureMatrix<FVector> &mF, Eigen::Matrix<typename FVector::Scalar, Eigen::Dynamic, Eigen::Dynamic> &mY) {
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        // Removes unused pre-images
        for(Index i = 0; i < N; i++) 
            while (N > 0 && mY.row(i).norm() < EPSILON) {
                mF.remove(i, true);
                if (i != N - 1) 
                    mY.row(i) = mY.row(N-1);
                N = N - 1;
            }
    }
    
    
    /**
     * @brief Removes pre-images with the null space method
     * 
     * Removes pre-images using the null space method
     * 
     * @param mF the feature matrix
     * @param nullSpace the null space vectors of the gram matrix of the feature matrix
    */
    template <class FVector, typename Derived>
    void removeNullSpacePreImages(FeatureMatrix<FVector> &mF, const Eigen::MatrixBase<Derived> &nullSpace) {
        
    }
    
    /**
     * @brief Removes unuseful pre-images 
     *
     * 1. Removes unused pre-images 
     * 2. Computes a \f$LDL^\dagger\f$ decomposition of the Gram matrix to find redundant pre-images
     * 3. Removes newly unused pre-images
     */
    template <class FVector, typename Derived>
    void removeUnusefulPreImages(FeatureMatrix<FVector> &mF, const Eigen::MatrixBase<Derived> &mY) {
        typedef typename FVector::Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        
        // Removes unused pre-images
        removeUnusedPreImages(mF, mY);
        
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());

        // LDL decomposition (stores the L^T matrix)
        Eigen::LDLT<Eigen::MatrixBase<Derived>, Eigen::Upper> ldlt(mF.getGramMatrix());
        Eigen::MatrixBase<Derived> &mLDLT = ldlt.matrixLDLT();
        
        // Get the rank
        Index rank = 0;
        for(Index i = 0; i < N; i++)
            if (mLDLT.get(i,i) < EPSILON) rank++;
        
       
        Eigen::Block<Matrix> mL1 = mLDLT.block(0,0,rank,rank);
        Eigen::Block<Matrix> mL2 = mLDLT.block(0,rank+1,rank,N-rank);
        
        if (rank != N) {
            // Gets the null space vectors in mL2
            mL1.template triangularView<Derived, Eigen::Upper>().solveInPlace(mL2);
            mL2 *= ldlt.transpositionsP().adjoint();
            
            // TODO: Remove the vectors
            removeNullSpacePreImages(mF, mL2);
                
            // Removes unused pre-images
            removeUnusedPreImages(mF, mY);
        }
    }
    
    
    
    /**
     * Reduces the set of images using the quadratic optimisation approach.
     * 
     * It is advisable to use the removeUnusefulPreImages technique first.
     *
     * @param target The number of pre-images that we should get at the end
     * @param F
     */
    template <class FVector, typename Derived>
    typename boost::enable_if_c<!Eigen::NumTraits<FVector>::IsComplex, void>::type
    removePreImagesWithQP(Index target, FeatureMatrix<FVector> &mF, const Eigen::MatrixBase<Derived> &mY) {
    }

    
    /**
     * Reduces the set of images using the quadratic optimisation approach.
     * 
     * It is advisable to use the removeUnusefulPreImages technique first.
     *
     * @param target The number of pre-images that we should get at the end
     * @param F
     */    
    template <class FVector, typename Derived>
    typename boost::enable_if_c<Eigen::NumTraits<FVector>::IsComplex, void>::type
    removePreImagesWithQP(Index target, FeatureMatrix<FVector> &mF, const Eigen::MatrixBase<Derived> &mY) {
    }

    
    /// Solve a QP system
    template<typename Scalar>
    void solve_qp(int r, Scalar lambda, const KQP_MATRIX(Scalar) &gramMatrix, const KQP_MATRIX(Scalar) &alpha, kqp::cvxopt::ConeQPReturn<Scalar> &result);
    
    /**
     * The KKT pre-solver to solver the QP problem
     * @ingroup coneqp
     */
    template<typename Scalar>
    class KQP_KKTPreSolver : public cvxopt::KKTPreSolver<Scalar> {
        Eigen::LLT<KQP_MATRIX(Scalar)> lltOfK;
        KQP_MATRIX(Scalar) B, BBT;
        
    public:
        KQP_KKTPreSolver(const KQP_MATRIX(Scalar)& gramMatrix);
        
        cvxopt::KKTSolver<Scalar> *get(const cvxopt::ScalingMatrix<Scalar> &w);
    };
    

    
} // end namespace

#endif
