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
#include <boost/type_traits/is_complex.hpp>
#include <numeric>

#include <kqp/kqp.hpp>
#include <Eigen/Eigenvalues>

#include <kqp/kernel_evd/utils.hpp>
#include <kqp/cleanup.hpp>
#include <kqp/cleaning/null_space.hpp>
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
         * @param target The number of pre-images that we should get at the end (the final number may be lower)
         * @param mF The matrix of pre-images
         * @param mY the 
         */
        void run(Index target, const FSpace &fs, const FMatrix &_mF, const ScalarAltMatrix &_mY, const RealVector &mD) {            
            KQP_LOG_ASSERT_F(main_logger, _mY.rows() == _mF.size(), "Incompatible dimensions (%d vs %d)", %_mY.rows() %_mF.size());

            // Diagonal won't change
            new_mD = mD;

            // Early stop
            if (target >= _mF.size())
                return;
                        
            //
            // (0) Compute the EVD (used by lambda computation), and use it to remove the null space 
            //     so that we are sure that we won't have a singular KKT
            //
            
            // Compute the eigenvalue decomposition of the Gram matrix
            // (assumes the eigenvalues are sorted by increasing order)
            
            ScalarMatrix eigenvectors;
            RealVector eigenvalues;
            boost::shared_ptr<ScalarMatrix> kernel(new ScalarMatrix());
            kqp::thinEVD(Eigen::SelfAdjointEigenSolver<ScalarMatrix>(fs.k(_mF).template selfadjointView<Eigen::Lower>()),
                eigenvectors, eigenvalues, kernel.get());
           
           // Remove using the null space approach if we can
           bool useNew = false;
           ScalarAltMatrix new_alt_mY;
           if (kernel->cols() > 0) {
               Eigen::PermutationMatrix<Dynamic, Dynamic, Index> mP;
               RealVector weights = _mY.rowwise().squaredNorm().array() * fs.k(_mF).diagonal().array().abs();
               new_mF = ReducedSetNullSpace<Scalar>::remove(_mF, *kernel, mP, weights);
               ScalarMatrix mY2(_mY);
               new_mY = (mP * mY2).topRows(new_mF.size()) + *kernel * (mP * mY2).bottomRows(_mY.rows() - new_mF.size());
               if (target >= new_mF.size()) 
                   return;
               
               useNew = true;
               new_alt_mY = std::move(new_mY);
               
               // Select the right eigenvectors
               eigenvectors = (mP * eigenvectors).topRows(new_mF.size());
               KQP_HLOG_INFO_F("Reduced rank to %d with null space (%d)", %new_mF.size() %kernel->cols())
           }
           

           // --- Set some variables before next operations
           
           const ScalarAltMatrix &mY = useNew ? new_alt_mY : _mY;
           const FMatrix &mF = useNew ? new_mF : _mF;
           const ScalarMatrix &gram = fs.k(mF);
           
            // Dimension of the basis
            Index r = mY.cols();

            // Number of pre-images
            Index n = gram.rows();

           //
           // (1) Estimate the regularization coefficient \f$lambda\f$
           //
           
          // RealVector sigma_3 = mD.array().abs().pow(3).matrix();
          //  
          //  // FIXME: rowwise should be implemented in AltMatrix
          //  RealVector sum_minDelta = ScalarMatrix(
          //      // Mimimum of |D_k / E_ik|^2
          //       (eigenvalues.rowwise().replicate(eigenvectors.cols()).array() / eigenvectors.array())
          //           .abs2().rowwise().minCoeff().matrix().asDiagonal()
          //   
          //       // A
          //       * mY.cwiseAbs2()
          //       
          //       // Sigma^3
          //       * sigma_3.asDiagonal()
          //   ).rowwise().sum();
            
           
            // mAS_.j = sigma_j^1/2 A_.j
            ScalarMatrix mAS = mY * mD.cwiseAbs().cwiseSqrt().asDiagonal();
           
            
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

            Real lambda =  acc_lambda.delta() / acc_lambda.maxa;   
            if (acc_lambda.maxa <= EPSILON * acc_lambda.delta()) {
                KQP_HLOG_WARN("There are only trivial solutions, calling cleanUnused");
                if (!useNew) new_mY = mY;
                new_mF = mF;
                new_mD = mD;
                CleanerUnused<Scalar>::run(new_mF, new_mY);
                if (target >= new_mF.size()) 
                    return;
                KQP_THROW_EXCEPTION_F(arithmetic_exception, "Trivial solutions were not found (target %d, size %d)", %target %new_mF.size());
            }
                
            KQP_HLOG_INFO_F("Lambda = %g", %lambda);                
            
            //
            // (2) Solve the cone quadratic problem
            //
            kqp::cvxopt::ConeQPReturn<Real> result;
            
            do {
                solve_qp<Scalar>(r, lambda, gram, mAS, mD.cwiseAbs().cwiseInverse().cwiseSqrt(), result);
                
                if (result.status == cvxopt::OPTIMAL) break;

                if (result.status == cvxopt::SINGULAR_KKT_MATRIX) 
                    KQP_HLOG_INFO("QP approach did not converge (singular KKT matrix)");
                
                Real oldLambda = lambda;
                lambda /= 2.;
                KQP_HLOG_INFO_F("Halving lambda to %g", %lambda);
                if (std::abs(lambda - oldLambda) < Eigen::NumTraits<Real>::epsilon())
                    KQP_THROW_EXCEPTION(arithmetic_exception, "Lambda is too small");
            } while (true);
            
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

        }
    };
    
    template <typename Scalar>
    class CleanerQP : public Cleaner<Scalar> {
    public:
          KQP_SCALAR_TYPEDEFS(Scalar);

          //! Set constraints on the number of pre-images
          void setPreImagesPerRank(float minimum, float maximum) {
              this->preImageRatios = std::make_pair(minimum, maximum);
          }

          virtual void cleanup(Decomposition<Scalar> &d) const override {
              // --- Ensure we have a small enough number of pre-images
              if (d.mX.size() > (this->preImageRatios.second * d.mD.rows())) {
                  if (d.fs.canLinearlyCombine()) {
                      // Easy case: we can linearly combine pre-images
                      d.mX = d.fs.linearCombination(d.mX, d.mY);
                      d.mY = ScalarMatrix::Identity(d.mX.size(), d.mX.size());
                  } else {
                      // Use QP approach
                      ReducedSetWithQP<Scalar> qp_rs;
                      qp_rs.run(this->preImageRatios.first * d.mD.rows(), d.fs, d.mX, d.mY, d.mD);

                      // Get the decomposition
                      d.mX = qp_rs.getFeatureMatrix();
                      d.mY = qp_rs.getMixtureMatrix();
                      d.mD = qp_rs.getEigenValues();

                      // The decomposition is not orthonormal anymore
                      d.orthonormal = false;
                  }

              }
          }
        
    private:
        /**
         * Minimum/Maximum number of pre-images per rank
         */
        std::pair<float,float> preImageRatios;            
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

#define KQP_CLEANING__QP_APPROACH_H_GEN(extern,scalar) \
  extern template class KQP_KKTPreSolver<scalar>; \
  extern template struct LambdaError<scalar>; \
  extern template struct ReducedSetWithQP<scalar>; \
  extern template class CleanerQP<scalar>;
  
#define KQP_SCALAR_GEN(scalar) KQP_CLEANING__QP_APPROACH_H_GEN(extern, scalar)
#include <kqp/for_all_scalar_gen.h.inc>
#undef KQP_SCALAR_GEN
}


#endif

