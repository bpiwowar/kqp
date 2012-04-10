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

#ifndef __KQP_PROBABILITIES_H__
#define __KQP_PROBABILITIES_H__

#include <Eigen/Core>

#include <kqp/kqp.hpp>
#include <kqp/alt_matrix.hpp>
#include <kqp/kernel_evd.hpp>
#include <kqp/trace.hpp>
#include <kqp/kernel_evd/utils.hpp>

namespace kqp {
    
    //! Helper class for linear combination
    template<typename FMatrix, class Enable = void> struct LinearCombination;
    
    template<typename FMatrix>
    struct LinearCombination<FMatrix, typename boost::enable_if_c<ftraits<FMatrix>::can_linearly_combine>::type>  {
        typedef typename ftraits<FMatrix>::Scalar Scalar;
        typedef typename ftraits<FMatrix>::ScalarAltMatrix ScalarAltMatrix;
        
        static bool run(FMatrix &fmatrix, const FMatrix &mF, const ScalarAltMatrix &mA, Scalar alpha) {
            fmatrix = mF.linear_combination(mA, alpha);
            return true;
        }
        
        static bool run(FMatrix &fmatrix, const FMatrix &mF, const ScalarAltMatrix &mA, Scalar alpha, const FMatrix &mY, const ScalarAltMatrix &mB, Scalar beta) {
            fmatrix = mF.linear_combination(mA, alpha, mY, mB, beta);
            return true;
        }
    };
    
    template<typename FMatrix>
    struct LinearCombination<FMatrix, typename boost::enable_if_c<!ftraits<FMatrix>::can_linearly_combine>::type>  {        
        typedef typename ftraits<FMatrix>::Scalar Scalar;
        typedef typename ftraits<FMatrix>::ScalarAltMatrix ScalarAltMatrix;

        bool run(FMatrix &, const FMatrix &, const ScalarAltMatrix &, Scalar) {
            return false;
        }
        
        bool run(FMatrix &, const FMatrix &, const ScalarAltMatrix &, Scalar , const FMatrix &, const ScalarAltMatrix &, Scalar) {
            return false;
        }
    };

    
    /**
     * Common class shared by (fuzzy) events and densities
     * 
     * <p>
     * The underlying density/subspace is represented by
     * <ul>
     * <li>A set of base vectors provided by {@linkplain FeatureMatrix} X</li>
     * <li>A linear combination matrix Y</li>
     * <li>A diagonal matrix S</li>
     * </ul>
     * 
     * such as <code>A X</code> is usually (i.e.
     * <code>AXX<sup>T</sup>A<sup>T</sup></code> is the identity), and the density
     * <code>rho</code> is expressed as <div>
     * <code>rho = A U S<sup>2</sup> U<sup>T</sup> A<sup>T</sup></code></div>
     * </p>
     * 
     * @date May 2011
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class FMatrix> class KernelOperator {       
    public:
        KQP_FMATRIX_TYPES(FMatrix);
        
        /**
         * Creates an object given a Kernel EVD
         * 
         * @param evd The kernel EVD 
         */
        KernelOperator(const KernelEVD<FMatrix>& evd)  {
            m_operator = evd.getDecomposition();
            m_operator.mD.unaryExprInPlace(Eigen::internal::scalar_sqrt_op<Real>());
        }
        
        
        /**
         * \brief Creates a new kernel operator
         */
        KernelOperator(const FMatrix &mX, const ScalarAltMatrix &mY, const RealVector &mS, bool orthonormal) : m_operator(mX,mY,mS,orthonormal) {
        }
        
        
        /**
         * Trim the eigenvalue decomposition to a lower rank
         * 
         * @param newRank
         *            The new rank of the subspace
         */
        void trim(size_t newRank);        
        
        /**
         * Get the rank of the operator
         */
        Index getRank() const {
            if (isOrthonormal()) 
                return m_operator.mS.size();
            return -1;
        }
        
        //! Get the feature matrix
        inline const FMatrix &X() const {
            return m_operator.mX;
        }
        
        inline const ScalarAltMatrix &Y() const {
            return m_operator.mY;
        }
        
        inline const RealAltVector &S() const {
            return m_operator.mD;
        }
        
        void orthonormalize() const {
            const_cast<KernelOperator*>(this)->_orthonormalize();
        }
        
        inline bool isOrthonormal() {
            return m_operator.orthonormal;
        }
        
        /**
         * Compute the squared norm of the operator
         */
        Real squaredNorm() {
            if (isOrthonormal()) 
                return m_operator.mD.squaredNorm();
            
            return kqp::squaredNorm(X(), Y(), S());
        }
        
        //! Computes the trace of the operator
        Real trace() const {
            Scalar tr = kqp::traceAAT(X(), Y(), RealVector(S()));
            // TODO: check if really real
            return (Real)tr;
        }
        
        //! Multiply the operator by a positive real
        void multiplyBy(Real alpha) {
            if (alpha < 0) 
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot multiply a kernel operator by a negative value (%g)", %alpha);
                
            m_operator.mD.unaryExprInPlace(Eigen::internal::scalar_multiple_op<double>(std::sqrt(alpha)));
        }
        
        
        /**
         * @brief Compute the inner products of the scaled basis of the operators 
         * @param that The other operator
         * @return A matrix where each row correspond to one dimension of the density, 
         *         and each column to one dimension of the subspace
         */
        ScalarMatrix inners(const KernelOperator<FMatrix>& that) const {
            return that.S().asDiagonal() * that.Y().transpose() * inner(that.X(), this->X()) * this->Y() * this->S().asDiagonal();
        }
        
        /**
         * @brief Get the matrix corresponding to the decomposition.
         * @throws An exception if the feature matrix cannot be linearly combined
         */
        FMatrix matrix() const {
            return X().linear_combination(ScalarMatrix(Y() * S().asDiagonal()));
        }
        
    protected:
        
        //! Orthonormalize the decomposition (non const version)
        void _orthonormalize() {
            // Check if we have something to do
            if (isOrthonormal()) return;
            
            // TODO: Eigen should only the needed part
            Eigen::SelfAdjointView<ScalarMatrix, Eigen::Lower> m = ScalarMatrix(S().asDiagonal() * Y().transpose() * X().inner() * Y() * S().asDiagonal());
            
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(m);
            
            ScalarMatrix _mY;
            RealVector _mS;
            kqp::thinEVD(evd, _mY, _mS);
            
            _mS.array() = _mS.array().cwiseAbs().cwiseSqrt();
            _mY *= _mS.cwiseInverse().asDiagonal();
            
            // If we can linearly combine, use it to reduce the future amount of computation
            if (LinearCombination<FMatrix>::run(m_operator.mX, m_operator.mX, _mY, 1))
                m_operator.mY = ScalarMatrix::Identity(X().size(),X().size());
            else 
                m_operator.mY.swap(_mY);
            
            m_operator.mD.swap(_mS);
            
            m_operator.orthonormal = true;
            
        }

        //! The current decomposition
        Decomposition<FMatrix> m_operator;
    };
        
    
    // Foward declarations
    template <class FMatrix> class Density;
    
    // Projection helper classes
    template<typename FMatrix, class Enable = void> struct ProjectionWithLC;
    template<typename FMatrix> struct Projection;

    /**
     * A document subspace is defined by the basis vectors (matrix {@linkplain #mU}
     * ). A diagonal matrix ({@linkplain #mS}) defines the weights associated to the
     * basis vectors
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     * 
     */
    template <class FMatrix> 
    class Event : public KernelOperator<FMatrix>  {
        bool useLinearCombination;
        void init() {
            useLinearCombination = ftraits<FMatrix>::can_linearly_combine;
        }
    public:
        KQP_FMATRIX_TYPES(FMatrix);

        /**
         * Construct a Event from a kernel EVD. See
         * {@linkplain KernelEigenDecomposition#KernelEigenDecomposition(KernelEVD, bool)}
         * @param evd The kernel EVD decomposition
         */
        Event(const KernelEVD<FMatrix> &evd, bool fuzzy = false) 
            : KernelOperator<FMatrix>(evd), 
              useLinearCombination(ftraits<FMatrix>::can_linearly_combine) {       
                  init();
                  if (!fuzzy) {
                      this->orthonormalize();
                      this->m_operator.mD = RealVector::Ones(this->X().size());
                  }
        }
        
        /** Construct an event from a basis */
        Event(const FMatrix &mX, bool orthonormal) : KernelOperator<FMatrix>(mX, ScalarAltMatrix::Identity(mX.rows()), RealVector::Ones(mX.size()), orthonormal) {
            init();
        }

        /** Construct an event from a basis */
        Event(const FMatrix &mX, const ScalarAltMatrix &mY, bool orthonormal) : KernelOperator<FMatrix>(mX, mY, RealVector::Ones(mX.size()), orthonormal) {
            init();
        }

        //! (debug) Manually sets the linear combination
        void setUseLinearCombination(bool b) {
            this->useLinearCombination = b;
        }
        
        /**
         * @brief Project onto a subspace.
         *
         * The resulting density will <b>not</b> normalized 
         */
        
        Density<FMatrix> project(const Density<FMatrix>& density, bool orthogonal = false) {
            if (orthogonal) 
                return useLinearCombination ? ProjectionWithLC<FMatrix>::projectOrthogonal(density, *this) : Projection<FMatrix>::projectOrthogonal(density, *this);
            
            return  useLinearCombination ? ProjectionWithLC<FMatrix>::project(density, *this) : Projection<FMatrix>::project(density, *this);
        }
                
        friend class Density<FMatrix>;
    };
    
    
    
    
    //! Projection (using linear combination)
    template<typename FMatrix> 
    struct ProjectionWithLC<FMatrix, typename boost::enable_if_c<ftraits<FMatrix>::can_linearly_combine>::type>  {
        KQP_FMATRIX_TYPES(FMatrix);

        static Density<FMatrix> project(const Density<FMatrix>& density, const Event<FMatrix> &event) {        
            event.orthonormalize();
            ScalarMatrix lc;
            noalias(lc) = event.Y() * event.S().asDiagonal() * event.Y().transpose() * inner(event.X(), density.X()) * density.Y() * density.S().asDiagonal();
            FMatrix mX = event.X().linear_combination(lc);
            ScalarAltMatrix mY = ScalarMatrix::Identity(mX.size(),mX.size());
            RealVector mS = RealVector::Ones(mY.cols());
            return Density<FMatrix>(mX, mY, mS, false);
        }
        
        static Density<FMatrix> projectOrthogonal(const Density<FMatrix>& density, const Event<FMatrix> &event) {
            // We need the event to be in an orthonormal form
            event.orthonormalize();
                        
            Index n = event.S().rows();
            
            // FIXME
            RealVector s = event.S();
            ScalarAltMatrix e_mY(event.Y() * (RealVector::Ones(n) - (RealVector::Ones(n) - s.cwiseAbs2()).cwiseSqrt()).asDiagonal() * (event.Y().transpose() * inner(event.X(), density.X()) * density.Y()));
            
            FMatrix mX = density.X().linear_combination(density.Y(), 1., event.X(), e_mY, -1.);
            
            return Density<FMatrix>(mX, ScalarMatrix::Identity(mX.size(),mX.size()), density.S(), false);
        }
    };

    
    //! Projection (error since FMatrix cannot be linearly combined)
    template<typename FMatrix> 
    struct ProjectionWithLC<FMatrix, typename boost::enable_if_c<!ftraits<FMatrix>::can_linearly_combine>::type>  {
        static Density<FMatrix> project(const Density<FMatrix>& , const Event<FMatrix> &) {
            KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot use linear combination for feature matrix of type %s", %KQP_DEMANGLE((FMatrix*)0));
        }
        static Density<FMatrix> projectOrthogonal(const Density<FMatrix>& , const Event<FMatrix> &) {
            KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot use linear combination for feature matrix of type %s", %KQP_DEMANGLE((FMatrix*)0));
        }
    };    
    
    
    //! Projection (without linear combination)
    template<typename FMatrix> 
    struct Projection  {
        KQP_FMATRIX_TYPES(FMatrix);
        
        static Density<FMatrix> project(const Density<FMatrix>& density, const Event<FMatrix> &event) {        
            event.orthonormalize();
            FMatrix mX = event.X();
            ScalarAltMatrix mY(event.Y() * event.S().asDiagonal() * event.Y().transpose() * inner(event.X(), density.X()) * density.Y() * density.S().asDiagonal());
            RealVector mS = RealVector::Ones(mY.cols());
            return Density<FMatrix>(mX, mY, mS, false);
        }
        
        static Density<FMatrix> projectOrthogonal(const Density<FMatrix>& density, const Event<FMatrix> &event) {
            event.orthonormalize();
            
            // Concatenates the vectors
            FMatrix mX = density.X();
            mX.add(event.X());
            
            // Computes the new linear combination matrix
            ScalarMatrix _mY(density.X().size() + event.X().size(), density.S().rows());
            RealVector s = event.S();
            Index n = s.rows();
            
            _mY.topRows(density.X().size()) = density.Y();
            _mY.bottomRows(event.X().size()) = ((Scalar)-1) * event.Y() * (RealVector::Ones(n) - (RealVector::Ones(n) - s.cwiseAbs2()).cwiseSqrt()).asDiagonal() * (event.Y().transpose() * inner(event.X(), density.X()) * density.Y());
            
            ScalarAltMatrix mY; 
            mY.swap(_mY);
            
                        
            // Return
            return Density<FMatrix>(mX, mY, density.S(), false);
        }
    };
    
    /**
     * A probability density
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class FMatrix> class Density: public KernelOperator<FMatrix>  {
    public: 
        KQP_FMATRIX_TYPES(FMatrix);
        
        /**
         * Creates a new density
         * 
         * @param evd
         * @param deepCopy
         */
        Density(const KernelEVD<FMatrix>& evd) : KernelOperator<FMatrix>(evd) {
        }
        
        Density(const FMatrix &mX, const ScalarAltMatrix &mY, const RealVector &mD, bool orthonormal) : KernelOperator<FMatrix>(mX, mY, mD, orthonormal) {
        }

        Density(const FMatrix &mX, bool orthonormal) : KernelOperator<FMatrix>(mX, ScalarMatrix::Identity(mX.size(),mX.size()), RealVector::Ones(mX.size()), orthonormal) {
        }
        
        //! Normalise the density
        Density &normalize() {
            this->multiplyBy((Scalar)1 / this->trace());
            return *this;
        }
        
        /**
         * Compute the probability of an event
         * 
         * @param event The event for which the probability is computed
         *            The event
         * @return The probability
         */
        Real probability(const Event<FMatrix>& event) const {
            return (event.S().asDiagonal() * event.Y() * inner(event.X(),this->X()) * this->Y() * this->S().asDiagonal()).squaredNorm();            
        }

        
        //! Computes the entropy
        Real entropy() const {
            this->orthonormalize();
            RealVector s = this->S();
            return (2 * s.array().log() * s.array().abs2()).sum();
        }
        
        /**
         * @brief Computes the divergence with another density
         *
         * Computes the divergence defined in the paper
         * "Conditional expectation in an operator algebra. IV. Entropy and information" by H. Umegaki (1962),
         * at page 69. The formula is:
         * \f[ J(\rho || \tau) = tr(\rho \log (\rho) - \rho \log(\tau))) \f]
         */
        Real computeDivergence(const Density<FMatrix> &tau, Real epsilon = EPSILON) const {
            const Density<FMatrix> &rho = *this;

            // --- Requires orthonormal decompositions
            rho.orthonormalize();
            tau.orthonormalize();
            
            // --- Notation
            
            ScalarMatrix inners = inner(rho.X(), tau.X());
            noalias(inners) = rho.S().asDiagonal() * rho.Y().transpose() * inners * tau.Y(); 

            // --- Compute tr(p log q)
            Scalar plogq = 0;
        
            Index dimension = rho.X().dimension();
            
			// The background density span the subspace 
			Real alpha = 1. / (Real)(dimension);
			Real alpha_noise = epsilon * alpha;
            
			// Includes the smoothing probability if not too small
            if (epsilon >= EPSILON) 
                plogq = log(alpha_noise) * (1. - inners.squaredNorm());
            
			
			// Main computation
			RealVector mD(tau.S().rows());
            
            RealVector tau_S = tau.S();
            
            plogq -= (inners * (-((1 - epsilon) * tau_S.array().abs2() + alpha_noise).log())  .sqrt().matrix().asDiagonal()).squaredNorm();
            
            
            // --- Compute tr(p log p)
            return this->entropy() - plogq;        
            
		}
        
        
    };
    
    
    
}

#endif