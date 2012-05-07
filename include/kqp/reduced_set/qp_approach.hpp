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
 
#include <functional>
#include <boost/type_traits/is_complex.hpp>
#include <numeric>

#include <kqp/kqp.hpp>
#include <Eigen/Eigenvalues>

#include <kqp/feature_matrix.hpp>
#include <kqp/coneprog.hpp>



namespace kqp {

#   include <kqp/define_header_logger.hpp>
    DEFINE_KQP_HLOGGER("kqp.qp-approach");
    
    /** \brief 
     * Solve the QP system associated to the reduced set problem.
     * 
     * <p>Mimimise 
     * \f$ \sum_{q=1}^{r}\beta_{q}^{\dagger}K\beta_{q}-2\Re\left(\alpha_{q}^{\dagger}Kb_{q}\right)+\lambda\xi_{q} \f$
     * </p>
     * @param r The rank of the operator
     * @param gramMatrix The gram matrix \f$K\$
     * @param complex If the original problem was in the complex field (changes the constraints)
     */
    template<typename Scalar>
    void solve_qp(int r, KQP_REAL_OF(Scalar) lambda, const KQP_MATRIX(Scalar) &gramMatrix, const KQP_MATRIX(Scalar) &alpha, const KQP_VECTOR(KQP_REAL_OF(Scalar)) &nu, kqp::cvxopt::ConeQPReturn<KQP_REAL_OF(Scalar)> &result);
    
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
//            std::cerr << "[[" << *this << " + " << other << "]] = " << r << std::endl;
            return r;
        }
        
        struct Comparator {
            bool operator() (const LambdaError &e1, const LambdaError &e2) { 
                if (e1.maxa < EPSILON) 
                    return e1.maxa < e2.maxa;
                return e1.delta * e2.maxa < e2.delta * e1.maxa; 
            }
        };
        
    };
    
    template<typename Scalar>
    std::ostream &operator<<(std::ostream &out, const LambdaError<Scalar> &l) {
        return out << boost::format("<delta=%g, maxa=%g, index=%d>") % l.delta %l.maxa %l.j;
    }
                             
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
    
    

 
    template <typename Scalar>
    struct ReducedSetWithQP {
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        
        /// Result of a run
        FMatrix new_mF;
        ScalarMatrix new_mY;
        RealVector new_mD;
        
        
        // 
        const FMatrix &getFeatureMatrix() { return new_mF; }
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
        void run(Index target, const FSpace &fs, const FMatrix &mF, const ScalarAltMatrix &mY, const RealVector &mD) {            
            // Get the Gram matrix
            const ScalarMatrix &gram = fs.k(mF);
            
            // Dimension of the basis
            Index r = mY.cols();

            // Number of pre-images
            Index n = gram.rows();
            
            KQP_LOG_ASSERT_F(main_logger, mY.rows() == n, "Incompatible dimensions (%d vs %d)", %mY.rows() %n);
            
            //
            // (1) Estimate the regularization coefficient \f$lambda\f$
            //

           
            // mAS_.j = sigma_j^1/2 A_.j
            ScalarMatrix mAS = mY * mD.cwiseAbs().cwiseSqrt().asDiagonal();
           
            
            std::vector< LambdaError<Real> > errors;
            for(Index i = 0; i < n; i++) {
                
                Real maxa = mAS.cwiseAbs().row(i).maxCoeff();
                Real delta = Eigen::internal::abs2(gram(i,i)) * mAS.cwiseAbs2().row(i).dot(mD.cwiseAbs().cwiseSqrt());
                KQP_HLOG_DEBUG_F("[%d] We have max = %.3g/ delta = %.3g", %i %maxa %delta);
                errors.push_back(LambdaError<Real>(delta, maxa, i));
            }
            
            
            typedef typename LambdaError<Real>::Comparator LambdaComparator;
            std::sort(errors.begin(), errors.end(), LambdaComparator());
            LambdaError<Real> acc_lambda = std::accumulate(errors.begin(), errors.begin() + n - target, LambdaError<Real>());
            
//            for(Index j = 0; j < n; j++) 
//                std::cerr << boost::format("[%d] delta=%g and maxa=%g\n") %j %errors[j].delta %errors[j].maxa;
//
//            std::cerr << boost::format("delta=%g and maxa=%g\n") %acc_lambda.delta % acc_lambda.maxa;

            Real lambda =  acc_lambda.delta / acc_lambda.maxa;   
            if (acc_lambda.maxa <= EPSILON * acc_lambda.delta) {
                KQP_HLOG_WARN("There are only trivial solutions, a call to cleanUnused would have been better");
                lambda = 1;
            }
                
            KQP_HLOG_INFO_F("Lambda = %g", %lambda);                
            
            //
            // (2) Solve the cone quadratic problem
            //
            kqp::cvxopt::ConeQPReturn<Real> result;
            solve_qp<Scalar>(r, lambda, gram, mAS, mD.cwiseAbs().cwiseInverse().cwiseSqrt(), result);

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
            
            Real lowest   = result.x.tail(n)[indices[0]];
            Real last_nsel = n-target-1 >= 0 ? result.x.tail(n)[indices[n-target-1]] : -1;
            Real first_sel = result.x.tail(n)[indices[n-target]];
            Real highest    = result.x.tail(n)[indices[n-1]];
            KQP_HLOG_INFO_F("Lambda values [%d/%d]: lowest=%g, last non selected=%g, first selected=%g, highest=%g [ratios %g and %g]", 
                            %target %n %lowest %last_nsel %first_sel %highest %(last_nsel/first_sel) %(last_nsel/highest));
            
            if (KQP_IS_DEBUG_ENABLED(KQP_HLOGGER)) {
                for(Index i = 0; i < n; i++) {
                    KQP_HLOG_DEBUG_F(boost::format("[%d] %g"), % indices[i] % result.x[result.x.size() - n + indices[i]]);
                }
            }
            
            // Construct a sub-view of the initial set of indices
            std::vector<bool> to_keep(n, false);
            for(Index i = n-target; i < n; i++) {
                to_keep[indices[i]] = true;
            }
            new_mF = mF.subset(to_keep.begin(), to_keep.end());
            
            
            //
            // (4) Project onto the new subspace
            //
            
            // Compute new_mY so that new_mF Y is orthonormal, ie new_mY' new_mF' new_mF new_mY is the identity
            
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(fs.k(new_mF).template selfadjointView<Eigen::Lower>());
            new_mY.swap(evd.eigenvectors());
            new_mY *= evd.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal();
            
            // Project onto new_mF new_mY
            
            KQP_MATRIX(Scalar) inner;
            inner = fs.k(new_mF, mF);
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
    class KQP_KKTPreSolver : public cvxopt::KKTPreSolver<typename Eigen::NumTraits<Scalar>::Real> {
    public:
        typedef typename Eigen::NumTraits<Scalar>::Real Real;

        KQP_KKTPreSolver(const KQP_MATRIX(Scalar)& gramMatrix, const KQP_VECTOR(Real) &nu);
        
        cvxopt::KKTSolver<Real> *get(const cvxopt::ScalingMatrix<Real> &w);
    private:
        Eigen::LLT<KQP_MATRIX(Real)> lltOfK;
        KQP_MATRIX(Real) B, BBT;
        KQP_VECTOR(Real) nu;

    };
    
#ifndef SWIG
# define KQP_SCALAR_GEN(scalar) extern template class KQP_KKTPreSolver<scalar>; extern template struct LambdaError<scalar>; extern template struct ReducedSetWithQP<scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
#endif
}


#endif

