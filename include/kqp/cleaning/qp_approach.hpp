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

#include <iostream>
#include <functional>
#include <list>
#include <boost/type_traits/is_complex.hpp>
#include <numeric>

#include <kqp/kqp.hpp>
#include <Eigen/Eigenvalues>

#include <kqp/evd_utils.hpp>
#include <kqp/cleanup.hpp>
#include <kqp/cleaning/null_space.hpp>
#include <kqp/feature_matrix.hpp>
#include <kqp/coneprog.hpp>



namespace kqp {
    
#ifndef SWIG
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
    void solve_qp(int r, KQP_REAL_OF(Scalar) lambda, const KQP_MATRIX(Scalar) &gramMatrix, const KQP_MATRIX(Scalar) &alpha, 
        const KQP_VECTOR(KQP_REAL_OF(Scalar)) &nu, kqp::cvxopt::ConeQPReturn<KQP_REAL_OF(Scalar)> &result,
        const cvxopt::ConeQPOptions< KQP_REAL_OF(Scalar) >& options = cvxopt::ConeQPOptions< KQP_REAL_OF(Scalar) >());
    
    /// Structured used to estimate the lambda
    template<typename Scalar>
    struct LambdaError {
        Scalar deltaMin, deltaMax;
        Scalar maxa;
        Index j;
        
        LambdaError() : deltaMin(0), deltaMax(0), maxa(0) {}
        LambdaError(Scalar deltaMin, Scalar deltaMax, Scalar maxa, Index j) : deltaMin(deltaMin), deltaMax(deltaMax), maxa(maxa), j(j) {}
        
        LambdaError operator+(const LambdaError &other) {
            LambdaError<Scalar> r;
            r.deltaMin = this->deltaMin + other.deltaMin;
            r.deltaMax = this->deltaMax + other.deltaMax;
            r.maxa = this->maxa + other.maxa;
            //            std::cerr << "[[" << *this << " + " << other << "]] = " << r << std::endl;
            return r;
        }
        
        inline Scalar delta() const {
            return (deltaMax + deltaMin) / 2.;
        }
        
        struct Comparator {
            bool operator() (const LambdaError &e1, const LambdaError &e2) { 
                if (e1.maxa < EPSILON) 
                    return e1.maxa < e2.maxa;
                return e1.delta() * e2.maxa < e2.delta() * e1.maxa; 
            }
        };
        
    };
    
    template<typename Scalar>
    std::ostream &operator<<(std::ostream &out, const LambdaError<Scalar> &l) {
        return out << boost::format("<delta=%g, maxa=%g, index=%d>") % l.delta %l.maxa %l.j;
    }
    
    template<class IndexedComparable>
    struct QPResultComparator {
        const IndexedComparable &x;
        Index r,n;
        
        bool isComplex;
        Index size;
        
        typedef typename Eigen::internal::remove_all<decltype(x[0])>::type Real;
        
        QPResultComparator(bool isComplex, const IndexedComparable &x, Index r, Index n) : x(x), r(r), n(n), isComplex(isComplex),
        size(isComplex ? 2*n*r : n*r) {}
        
        inline Real getXi(int i) {
            return std::abs(x[size+i]);
        }
        
        inline Real getAbsSum(int i) {
            Real sum = 0;
            for(Index j = 0; j < r; j++) {
                if (isComplex) sum += Eigen::internal::abs2(x[2*i*n + j]) + Eigen::internal::abs2(x[2*i*n + n + j]);
                else sum += Eigen::internal::abs2(x[i*n + j]);
                
            }
            return std::sqrt(sum);
        }
        
        inline Real getAbsMax(int i) {
            Real max = 0;
            for(Index j = 0; j < r; j++) {
                if (isComplex) max = std::max(max, std::max(std::abs(x[2*i*n + j]),std::abs(x[2*i*n + n + j])));
                else max = std::max(max, std::abs(x[i*n + j]));
                
            }
            return max;
        }
        
        
        bool operator() (int i1, int i2) {
            return  getAbsSum(i1) < getAbsSum(i2);
        }
    };
    
    template<typename Derived> QPResultComparator<Derived> getQPResultComparator(bool isComplex, const Derived &x, Index r, Index n) {
        return QPResultComparator<Derived>(isComplex, x, r, n);
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
         * @param target The number of pre-images that we should get at the end (the final number may be lower)
         * @param mF The matrix of pre-images
         * @param mY the 
         */
        void run(Index target, const FSpace &fs, const FMatrix &_mF, const ScalarAltMatrix &_mY, const RealVector &mD) {            
            KQP_LOG_ASSERT_F(main_logger, _mY.rows() == _mF->size(), "Incompatible dimensions (%d vs %d)", %_mY.rows() %_mF->size());
            
            // Diagonal won't change
            new_mD = mD;
            
            // Early stop
            if (target >= _mF->size()) {
                new_mY = _mY;
                new_mF = _mF;
                return;
            }
            
            KQP_HLOG_INFO_F("QP approach : reduce rank from %d to %d", %_mF->size() %target);

            //
            // (0) Compute the EVD (used by lambda computation), and use it to remove the null space 
            //     so that we are sure that we won't have a singular KKT
            //
            
            // Compute the eigenvalue decomposition of the Gram matrix
            // (assumes the eigenvalues are sorted by increasing order)
            
            // Remove using the null space approach if we can
            bool useNew = false;
            ScalarAltMatrix new_alt_mY;
            {
                
                ScalarMatrix eigenvectors;
                RealVector eigenvalues;
                boost::shared_ptr<ScalarMatrix> kernel(new ScalarMatrix());
                Real threshold = fs->k(_mF).squaredNorm() * std::sqrt(Eigen::NumTraits<Real>::epsilon());
                kqp::ThinEVD<ScalarMatrix>::run(Eigen::SelfAdjointEigenSolver<ScalarMatrix>(fs->k(_mF).template selfadjointView<Eigen::Lower>()),
                                                eigenvectors, eigenvalues, kernel.get(), threshold);
                
                
                if (kernel->cols() > 0) {
                    Eigen::PermutationMatrix<Dynamic, Dynamic, Index> mP;
                    RealVector weights = _mY.rowwise().squaredNorm().array() * fs->k(_mF).diagonal().array().abs();
                    
                    // kernel will contain a matrix such that *kernel * mP * mY
                    new_mF = ReducedSetNullSpace<Scalar>::remove(_mF, *kernel, mP, weights);
                    
                    // Y <- (Id A) P Y
                    ScalarMatrix mY2(_mY); // FIXME: .topRows() should be defined in AltMatrix expressions
                    ScalarMatrix mY3 = (mP * mY2).topRows(new_mF->size()) + *kernel * (mP * mY2).bottomRows(_mY.rows() - new_mF->size());
                    
                    KQP_HLOG_INFO_F("Reduced rank to %d [target %d] with null space (kernel size: %d)", %new_mF->size() %target %kernel->cols())
                    if (target >= new_mF->size()) {
                        new_mY = std::move(mY3);
                        return;
                    }
                    
                    useNew = true;
                    new_alt_mY = std::move(mY3);
                }
                
                // --- Set some variables before next operations
                
            }
            const FMatrix &mF = useNew ? new_mF : _mF;
            const ScalarAltMatrix &mY = useNew ? new_alt_mY : _mY;
            const ScalarMatrix &gram = fs->k(mF);
            
            // Dimension of the basis
            Index r = mY.cols();
            
            // Number of pre-images
            Index n = gram.rows();
            
            //
            // (1) Estimate the regularization coefficient \f$lambda\f$
            //
            
            // mAS_.j = sigma_j^1/2 A_.j
            ScalarMatrix mAS = mY * mD.cwiseAbs().cwiseSqrt().asDiagonal();
            
            // We first order the pre-images 
            std::vector< LambdaError<Real> > errors;
            for(Index i = 0; i < n; i++) {
                Real maxa = mAS.cwiseAbs().row(i).maxCoeff();
                Real deltaMax = gram(i,i) * mAS.cwiseAbs2().row(i).dot(mD.cwiseAbs().cwiseSqrt());
                Real deltaMin = deltaMax; //std::min(sum_minDelta[i], deltaMax);
                KQP_HLOG_DEBUG_F("[%d] We have max = %.3g/ delta = (%.3g,%.3g)", %i %maxa %deltaMin %deltaMax);
                errors.push_back(LambdaError<Real>(deltaMin, deltaMax, maxa, i));
            }
            
            
            typedef typename LambdaError<Real>::Comparator LambdaComparator;
            std::sort(errors.begin(), errors.end(), LambdaComparator());
            LambdaError<Real> acc_lambda = std::accumulate(errors.begin(), errors.begin() + n - target, LambdaError<Real>());
            
            // for(Index j = 0; j < n; j++) 
            //     std::cerr << boost::format("[%d] delta=(%g,%g) and maxa=%g\n") %j %errors[j].deltaMin %errors[j].deltaMax %errors[j].maxa;
            // std::cerr << boost::format("delta=(%g,%g)  and maxa=%g\n") %acc_lambda.deltaMin %acc_lambda.deltaMax % acc_lambda.maxa;
            
            // Check for a trivial solution
            if (acc_lambda.maxa <= EPSILON * acc_lambda.delta()) {
                KQP_HLOG_WARN("There are only trivial solutions, calling cleanUnused");
                if (!useNew) {
                    new_mY = mY;
                    new_mF = mF;   
                }
                new_mD = mD;
                CleanerUnused<Scalar>::run(new_mF, new_mY);
                if (target >= new_mF->size()) 
                    return;
                KQP_THROW_EXCEPTION_F(arithmetic_exception, "Trivial solutions were not found (target %d, size %d)", %target %new_mF->size());
            }
            
            // Now, we compute a more accurate lambda
            Real lambda = acc_lambda.delta() / acc_lambda.maxa; 
            
            
            KQP_HLOG_INFO_F("Lambda = %g", %lambda);                
            
            //
            // (2) Solve the cone quadratic problem
            //
            kqp::cvxopt::ConeQPReturn<Real> result;
            
            RealVector nu = mD.cwiseAbs().cwiseInverse().cwiseSqrt();
            
            do {
                solve_qp<Scalar>(r, lambda, gram, mAS, nu, result);
                
                if (result.status == cvxopt::OPTIMAL) break;
                
                if (result.status == cvxopt::SINGULAR_KKT_MATRIX) 
                    KQP_HLOG_INFO("QP approach did not converge (singular KKT matrix)");
                
                Real oldLambda = lambda;
                lambda /= 2.;
                KQP_HLOG_INFO_F("Halving lambda to %g", %lambda);
                if (std::abs(lambda - oldLambda) < Eigen::NumTraits<Real>::epsilon())
                    KQP_THROW_EXCEPTION(arithmetic_exception, "Lambda is too small");
            } while (true);
            
            
            //
            // (3) Get the subset 
            //
            
            // In result.x,
            // - the r mixture vectors (of size n or 2n in the complex case) are stored one after each other
            // - the last n components are the maximum of the norm of the coefficients for each pre-image
            // we use the <target> first
            std::vector<Index> indices(n);
            for(Index i = 0; i < n; i++)
                indices[i] = i;
            
            // Sort by increasing order: we will keep only the target last vectors
            auto comparator = getQPResultComparator(boost::is_complex<Scalar>::value, result.x, n, r);
            std::sort(indices.begin(), indices.end(), comparator);
            
            Real lowest   = result.x.tail(n)[indices[0]];
            Real last_nsel = n-target-1 >= 0 ? result.x.tail(n)[indices[n-target-1]] : -1;
            Real first_sel = result.x.tail(n)[indices[n-target]];
            Real highest    = result.x.tail(n)[indices[n-1]];
            KQP_HLOG_INFO_F("Lambda values [%d/%d]: lowest=%g, last non selected=%g, first selected=%g, highest=%g [ratios %g and %g]", 
                            %target %n %lowest %last_nsel %first_sel %highest %(last_nsel/first_sel) %(last_nsel/highest));
            
            if (KQP_IS_DEBUG_ENABLED(KQP_HLOGGER)) {
                for(Index i = 0; i < n; i++) {
                    Index j = indices[i];
                    KQP_HLOG_DEBUG_F(boost::format("[%d] absXi=%g absSum=%g absMax=%g"), % indices[i] 
                                     % comparator.getXi(j) %comparator.getAbsSum(j) %comparator.getAbsMax(j));
                }
            }
            
            // Construct a sub-view of the initial set of indices
            std::vector<bool> to_keep(n, false);
            for(Index i = n-target; i < n; i++) {
                to_keep[indices[i]] = true;
            }
            FMatrix _new_mF = mF->subset(to_keep.begin(), to_keep.end());
            
            
            //
            // (4) Project onto the new subspace
            //
            
            // Compute new_mY so that new_mF Y is orthonormal, ie new_mY' new_mF' new_mF new_mY is the identity
            
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(fs->k(_new_mF).template selfadjointView<Eigen::Lower>());
            
            new_mY.swap(evd.eigenvectors());
            new_mY *= evd.eigenvalues().cwiseAbs().cwiseSqrt().cwiseInverse().asDiagonal();
            
            // Project onto new_mF new_mY
            
            new_mY *= fs->k(_new_mF, new_mY, mF, mY);
            new_mF = std::move(_new_mF);
        }
    };
#endif

    template <typename Scalar>
    class CleanerQP : public Cleaner<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        CleanerQP() : 
            m_preImageRatios(std::make_pair(0, std::numeric_limits<Real>::infinity())),
            m_preImagesRange(std::make_pair(0, std::numeric_limits<Index>::max()))
        {}

        //! Set constraints on the number of pre-images
        void setPreImagesPerRank(float reset, float maximum) {
            this->m_preImageRatios = std::make_pair(reset, maximum);
        }

        void setRankRange(Index reset, Index maximum) {
            this->m_preImagesRange = std::make_pair(reset, maximum);
        }
        
        virtual void cleanup(Decomposition<Scalar> &d) const override {
            // Sets the target value
            Index target = d.mX->size();

            if (d.mX->size() > this->m_preImageRatios.second * d.mD.rows())
                target = std::min(target, (Index)(this->m_preImageRatios.first * d.mD.rows()));

            if (d.mX->size() > this->m_preImagesRange.second)
                target = std::min(target, this->m_preImagesRange.first);

            // Ensure there is one pre-image per rank at least
            target = std::max(target, d.mD.rows());

            // --- Ensure we have a small enough number of pre-images
            if (d.mX->size() > target) {
                if (d.fs->canLinearlyCombine()) {
                    // Easy case: we can linearly combine pre-images
                    d.mX = d.fs->linearCombination(d.mX, d.mY);
                    d.mY = Eigen::Identity<Scalar>(d.mX->size(), d.mX->size());
                } else {
                    // Use QP approach
                    ReducedSetWithQP<Scalar> qp_rs;
                    qp_rs.run(target, d.fs, d.mX, d.mY, d.mD);
                    
                    // Get the decomposition
                    d.mX = std::move(qp_rs.getFeatureMatrix());
                    d.mY = std::move(qp_rs.getMixtureMatrix());
                    d.mD = std::move(qp_rs.getEigenValues());
                    
                    // The decomposition is not orthonormal anymore
                    d.orthonormal = false;
                }
                
            }
        }
        
    private:
        /**
         * Minimum/Maximum number of pre-images per rank (ratios and absolute)
         * First value is the reset value, second value is the bound
         */
        std::pair<float,float> m_preImageRatios;            
        std::pair<Index,Index> m_preImagesRange;            
    };
    
    
#ifndef SWIG    
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
    
#define KQP_CLEANING__QP_APPROACH_H_GEN(extern,scalar) \
extern template class KQP_KKTPreSolver<scalar>; \
extern template struct LambdaError<scalar>; \
extern template struct ReducedSetWithQP<scalar>; \
extern template class CleanerQP<scalar>;
    
#define KQP_SCALAR_GEN(scalar) KQP_CLEANING__QP_APPROACH_H_GEN(extern, scalar)
#include <kqp/for_all_scalar_gen.h.inc>
#undef KQP_SCALAR_GEN
#endif

}


#endif

