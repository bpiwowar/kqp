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

#include <numeric>

#include <Eigen/Dense>

#include "feature_matrix.hpp"
#include "coneprog.hpp"
#include "rank_selector.hpp"

namespace kqp {
    
    /**
     * @brief Removes pre-images with the null space method
     */
    template <class FMatrix> 
    void removeUnusedPreImages(FMatrix &mF, typename ftraits<FMatrix>::ScalarMatrix &mY) {
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        std::vector<bool> to_keep(N, true);
        
        // Removes unused pre-images
        for(Index i = 0; i < N; i++) 
            if (mY.row(i).norm() < EPSILON) 
                to_keep[i] = false;

        select_rows(to_keep.begin(), to_keep.end(), mY, mY);
        mF.subset(to_keep.begin(), to_keep.end());
    }

    /// Solve a QP system
    template<typename Scalar>
    void solve_qp(int r, Scalar lambda, const KQP_MATRIX(Scalar) &gramMatrix, const KQP_MATRIX(Scalar) &alpha, kqp::cvxopt::ConeQPReturn<Scalar> &result);
    

    // Estimate the lambda
    template<typename Scalar>
    struct LambdaError {
        Scalar delta;
        Scalar maxa;
        Index j;
        LambdaError(Scalar delta, Scalar maxa, Index j) : delta(delta), maxa(maxa), j(j) {}
        
        LambdaError operator+(const LambdaError &other) {
            LambdaError<Scalar> r;
            r.delta = this->delta + other.delta;
            r.maxa = this->maxa + other.maxa;
        }
        struct Comparator {
            bool operator() (const LambdaError &e1, const LambdaError &e2) { return e1.delta < e2.delta; }
        };
        
    };
    
    /// Compares using indices that refers to an array of comparable elements
    template<class IndexedComparable>
    struct IndirectComparator {
        const IndexedComparable &x;
        IndirectComparator(const IndexedComparable &x) : x(x) {}
        bool operator() (int i1, int i2) {
            return x[i1] < x[i2];
        }
    };
    
    
    /**
     * Reduces the set of images using the quadratic optimisation approach.
     * 
     * It is advisable to use the removeUnusefulPreImages technique first.
     * The decomposition <em>should</em> be orthonormal in order to work better.
     * The returned decomposition is not orthonormal
     *
     * @param target The number of pre-images that we should get at the end
     * @param mF The matrix of pre-images
     * @param mY the 
     */
    template <class FMatrix, typename Derived>
    typename boost::enable_if_c<!Eigen::NumTraits<typename ftraits<FMatrix>::Scalar>::IsComplex, void>::type
    removePreImagesWithQP(Index target, FeatureMatrix<FMatrix> &mF, const typename ftraits<FMatrix>::AltMatrix &mY, 
                          const typename ftraits<FMatrix>::RealVector &mD,
                          FeatureMatrix<FMatrix> &new_mF, typename ftraits<FMatrix>::AltMatrix &new_mY) {
        // Real case
        typedef ftraits<FMatrix> FTraits;
        typedef typename FTraits::Scalar Scalar;
        typedef typename FTraits::Real Real;
        typedef typename FTraits::ScalarMatrix ScalarMatrix;
        typedef typename FTraits::ScalarVector ScalarVector;
        typedef typename FTraits::RealVector RealVector;
        
        Index r = mY.cols();
        
        ScalarMatrix gram = mF.inner();
        Index n = gram.rows();
        
    
        // Estimate lambda
        std::vector<LambdaError<Real> > errors;
        for(Index j = 0; j < n; j++) {
            Scalar maxa = 0;
            Scalar delta = 0;
            for(Index i = 0; i < r; i++) {
                delta += mY(i,j);
                maxa = std::max(maxa, std::abs(mY(i,j)));
            }
            errors.push_back(LambdaError<Scalar>(delta*gram(j,j)), maxa, j);
        }
        
        std::sort(errors.begin(), errors.end(), LambdaError<Scalar>::Comparator());
        LambdaError<Scalar> acc_lambda = std::accumulate(errors.begin(), errors.begin() + target, LambdaError<Scalar>());
        
        Real lambda = acc_lambda.delta / acc_lambda.maxa;
        
        
        // Solve
        kqp::cvxopt::ConeQPReturn<Scalar> result;
        solve_qp(r, lambda, gram, mY * mD.asDiagonal(), result);
        
        // Re-orthogonalise: direct EVD on reduced gram matrix
        
        // In result.x, the last n components are the maximum of the norm of the coefficients for each pre-image
        // we use the <target> first
        std::vector<Index> indices(n);
        for(Index i = 0; i < n; i++)
            indices[i] = i;
        
        // Sort by increasing order: we will keep only the target last vectors
        std::sort(indices.begin(), indices.end(), IndirectComparator<Real>(result.x.tail(n)));
        
        // Now sorts so that we minimise the number of swaps
        std::sort(indices.end() - target, indices.end());
        
        // Construct a sub-view of the initial set of indices
        new_mF = mF.subset(indices.begin(), indices.end());
        
        // Compute mY so that X_new Y is orthonormal, ie Y' X_new' X_new Y is the identity
        Eigen::SelfAdjointEigenSolver<typename FTraits::Matrix> evd(new_mF.inner().template selfadjointView<Eigen::Lower>());
        new_mY = evd.eigenvectors() * evd.eigenvalues().cwiseAbs().cwiseSqrt().cwiseInverse().asDiagonal();

        // Project
        new_mY *= new_mY.adjoint() * new_mF.inner(mF) * mY;
    }

    
    /**
     * Reduces the set of images using the quadratic optimisation approach.
     * 
     * It is advisable to use the removeUnusefulPreImages technique first.
     *
     * @param target The number of pre-images that we should get at the end
     * @param F
     */    
    template <class FMatrix, typename Derived>
    typename boost::enable_if_c<Eigen::NumTraits<typename FMatrix::Scalar>::IsComplex, void>::type
    removePreImagesWithQP(Index target, FeatureMatrix<FMatrix> &mF, const Eigen::MatrixBase<Derived> &mY);

    
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
