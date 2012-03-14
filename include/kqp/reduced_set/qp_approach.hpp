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

#ifndef __KQP_REDUCED_QP_APPROACH_H__
#define __KQP_REDUCED_QP_APPROACH_H__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <numeric>

#include <kqp/kqp.hpp>
#include <kqp/feature_matrix.hpp>
#include <kqp/coneprog.hpp>


namespace kqp {
    /** \brief Solve the QP system associated to the reduced set problem.
     * 
     * <p>Mimimise 
     * \f$ \sum_{q=1}^{r}\beta_{q}^{\dagger}K\beta_{q}-2\Re\left(\alpha_{q}^{\dagger}Kb_{q}\right)+\lambda\xi_{q} \f$
     * </p>
     * @param r The rank of the operator
     * @param gramMatrix The gram matrix \f$K\$
     */
    template<typename Scalar>
    void solve_qp(int r, Scalar lambda, const KQP_MATRIX(Scalar) &gramMatrix, const KQP_MATRIX(Scalar) &alpha, kqp::cvxopt::ConeQPReturn<Scalar> &result);
    
    /// Structured used to estimate the lambda
    template<typename Scalar>
    struct LambdaError {
        Scalar delta;
        Scalar maxa;
        Index j;
        
        LambdaError() : delta(0), maxa(0) {}
        LambdaError(Scalar delta, Scalar maxa, Index j) : delta(delta), maxa(maxa), j(j) {}
        
        LambdaError operator+(const LambdaError &other) {
            LambdaError<Scalar> r;
            r.delta = this->delta + other.delta;
            r.maxa = this->maxa + other.maxa;
            return r;
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
            return std::abs(x[i1]) < std::abs(x[i2]);
        }
    };
    
    template<typename Derived> IndirectComparator<Derived> getIndirectComparator(const Derived &x) {
        return IndirectComparator<Derived>(x);
    }
    
 
    template <class FMatrix>
    struct ReducedSetWithQP {
        typedef ftraits<FMatrix> FTraits;
        typedef typename FTraits::Scalar Scalar;
        typedef typename FTraits::Real Real;
        typedef typename FTraits::ScalarMatrix ScalarMatrix;
        typedef typename FTraits::ScalarVector ScalarVector;
        typedef typename FTraits::RealVector RealVector;
        
        
        /// Result of a run
        FMatrix new_mF;
        ScalarMatrix new_mY;
        RealVector new_mD;
        
        
        // 
        FMatrix &getFeatureMatrix() { return new_mF; }
        ScalarMatrix &getMixtureMatrix() { return new_mY; }
        RealVector &getEigenValues() { return new_mD; }
        
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
        //  typename boost::enable_if_c<!Eigen::NumTraits<typename ftraits<FMatrix>::Scalar>::IsComplex, void>::type
        void run(Index target, const FMatrix &mF, const typename ftraits<FMatrix>::ScalarAltMatrix &mY, const RealVector &mD) {
            ScalarMatrix gram = mF.inner();
            
            
            // Dimension of the basis
            Index r = mY.cols();

            // Number of pre-images
            Index n = gram.rows();
            
            KQP_LOG_ASSERT_F(main_logger, mY.rows() == n, "Incompatible dimensions (%d vs %d)", %mY.rows() %n);
            
            //
            // (1) Estimate the regularization coefficient \f$lambda\f$
            //
            
            std::vector< LambdaError<Real> > errors;
            for(Index j = 0; j < r; j++) {
                Real maxa = 0;
                Real delta = 0;
                for(Index i = 0; i < n; i++) {
                    Real x = Eigen::internal::abs2(mY(i,j)); 
                    delta += x;
                    maxa = std::max(maxa, std::sqrt(x));
                }
                errors.push_back(LambdaError<Real>(delta * Eigen::internal::abs2(gram(j,j)), maxa, j));
            }
            
            typedef typename LambdaError<Real>::Comparator LambdaComparator;
            std::sort(errors.begin(), errors.end(), LambdaComparator());
            LambdaError<Real> acc_lambda = std::accumulate(errors.begin(), errors.begin() + target, LambdaError<Real>());
            
            Real lambda =  acc_lambda.delta / acc_lambda.maxa;   
                            
            //
            // (2) Solve the cone quadratic problem
            //
            kqp::cvxopt::ConeQPReturn<Scalar> result;
            solve_qp<Scalar>(r, lambda, gram, (mY * mD.asDiagonal()), result);

            if (result.status == cvxopt::SINGULAR_KKT_MATRIX) 
                KQP_THROW_EXCEPTION_F(arithmetic_exception, "QP approach did not converge (singular KKT matrix)", %result.status);
            
            // FIXME: case not converged
            
            //
            // (3) Get the subset 
            //
            
            // In result.x, the last n components are the maximum of the norm of the coefficients for each pre-image
            // we use the <target> first
            std::vector<Index> indices(n);
            for(Index i = 0; i < n; i++)
                indices[i] = i;
            
            // Sort by increasing order: we will keep only the target last vectors
            std::sort(indices.begin(), indices.end(), getIndirectComparator(result.x.tail(n)));
            
            // Now sorts so that we minimise the number of swaps
            std::sort(indices.end() - target, indices.end());
            
            // Construct a sub-view of the initial set of indices
            std::vector<bool> to_keep(n, false);
            for(Index i = n-target; i < n; i++) {
                to_keep[indices[i]] = true;
            }
            mF.subset(to_keep.begin(), to_keep.end(), new_mF);
            
            
            //
            // (4) Project onto the new subspace
            //
            
            // Compute new_mY so that new_mF Y is orthonormal, ie new_mY' new_mF' new_mF new_mY is the identity
            
            Eigen::SelfAdjointEigenSolver<typename FTraits::ScalarMatrix> evd(new_mF.inner().template selfadjointView<Eigen::Lower>());
            new_mY.swap(evd.eigenvectors());
            new_mY *= evd.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal();
            
            // Project onto new_mF new_mY
            
            KQP_MATRIX(Scalar) inner;
            kqp::inner(new_mF, mF, inner);
            new_mY *= new_mY.adjoint() * inner * mY;

            // Diagonal matrix does not change
            new_mD = mD;

        }
        
    };
    
    
    
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
    
}


#endif

