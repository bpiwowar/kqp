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
        KernelOperator(const KernelEVD<FMatrix>& evd) : orthonormal(true) {
            evd.get_decomposition(mX, mY, mS);
            mS = mS.cwiseSqrt();
        }
        
        
        /**
         * \brief Creates a new kernel operator
         */
        KernelOperator(const FMatrix &mX, const ScalarAltMatrix &mY, const RealVector &mS, bool orthonormal) : mX(mX), mY(mY), mS(mS), orthonormal(orthonormal) {
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
        size_t getRank() const {
            if (orthonormal) return mS.size();
        }
        
        // Get the feature matrix
        inline const FMatrix &X() const {
            return mX;
        }
        
        inline const ScalarAltMatrix &Y() const {
            return mY;
        }
        
        inline const RealVector &S() const {
            return mS;
        }
        
        void orthonormalize() const {
            const_cast<KernelOperator*>(this)->_orthonormalize();
        }
        
        /**
         * Compute the squared norm of the operator
         */
        Real squaredNorm() {
            if (orthonormal) return mS.squaredNorm();
            
            return kqp::squaredNorm(X(), Y(), S());
        }
        
        //! Computes the trace of the operator
        Real trace() {
            Scalar tr = kqp::trace(X(), Y(), S().cwiseAbs2());
            // TODO: check if really real
            return (Real)tr;
        }
        
        //! Multiply the operator by a positive real
        void multiplyBy(Real alpha) {
            if (alpha < 0) 
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot multiply a kernel operator by a negative value (%g)", %alpha);
                
            mS *= std::sqrt(alpha);
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
            return X().linear_combination(ScalarMatrix(mY * mS));
        }
        
    protected:
        
        //! Orthonormalize the decomposition (non const version)
        void _orthonormalize() {
            if (!orthonormal) {
                // TODO: Eigen should only the needed part
                Eigen::SelfAdjointView<ScalarMatrix, Eigen::Lower> m = ScalarMatrix(S().asDiagonal() * Y().transpose() * X().inner() * Y() * S().asDiagonal());
                
                Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(m);
                
                ScalarMatrix _mY;
                kqp::thinEVD(evd, _mY, mS);
                
                mS.array() = mS.array().cwiseAbs().cwiseSqrt();
                _mY *= mS.cwiseInverse().asDiagonal();
                
                if (LinearCombination<FMatrix>::run(mX, mX, mY, 1))
                    mY = ScalarMatrix::Identity(mX.size(),mX.size());
                
                else mY.swap(_mY);
                
                orthonormal = true;
            }
        }

        
        /**
         * The base vector list
         */
        FMatrix  mX;
        
        /**
         * The combination matrix.
         * 
         * In case of an EVD decomposition, mX mY is orthonormal
         */
        ScalarAltMatrix mY;
        
        /**
         * The singular values
         *
         * This matrix is used only if the KernelOperator is in a 
         * EVD decomposed form
         */
        RealVector mS;
        
        //! Is the decomposition othonormal, i.e. is Y^T X^T X Y the identity?
        bool orthonormal;
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
                  if (fuzzy) this->mS = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(this->X().size(), this->X().size());
        }
        
        /** Construct an event from a basis */
        Event(const FMatrix &mX, bool orthonormal) : KernelOperator<FMatrix>(mX, ScalarAltMatrix::Identity(mX.rows()), RealVector::Ones(mX.size()), orthonormal) {
            init();
        }

        /** Construct an event from a basis */
        Event(const FMatrix &mX, const ScalarAltMatrix &mY, bool orthonormal) : KernelOperator<FMatrix>(mX, mY, RealVector::Ones(mX.size()), orthonormal) {
            init();
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
            ScalarMatrix lc;
            noalias(lc) = event.Y().transpose() * inner(density.X(), density.X()) * density.Y() * density.S();
            FMatrix mX = event.X().linear_combination(lc);
            ScalarAltMatrix mY = ScalarMatrix::Identity(mX.size(),mX.size());
            RealVector mS = RealVector::Ones(mX.size());
            return Density<FMatrix>(mX, mY, mS, false);
        }
        
        static Density<FMatrix> projectOrthogonal(const Density<FMatrix>& density, const Event<FMatrix> &event) {
            // We need the event to be in an orthonormal form
            event.orthonormalize();
                        
            Index n = event.S().rows();
            
            ScalarAltMatrix e_mY(event.Y() * (RealVector::Ones(n) - (RealVector::Ones(n) - event.S().cwiseAbs2()).cwiseSqrt()).asDiagonal() * (event.Y().transpose() * inner(event.X(), density.X()) * density.Y()));
            
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
            FMatrix mX = event.X();
            ScalarAltMatrix mY(event.Y().transpose() * inner(density.X(), density.X()) * density.Y() * density.S());
            RealVector mS = RealVector::Ones(mX.size());
            return Density<FMatrix>(mX, mY, mS, false);
        }
        
        static Density<FMatrix> projectOrthogonal(const Density<FMatrix>& density, const Event<FMatrix> &event) {
            event.orthonormalize();
            
            // Concatenates the vectors
            FMatrix mX = density.X();
            mX.add(event.X());
            
            // Computes the new linear combination matrix
            ScalarMatrix _mY(density.X().size() + event.Y().size(), density.S().rows());
            Index n = event.S().rows();
            
            _mY.topRows(density.X().size()) = ((Scalar)-1) * density.Y() * density.S();
            _mY.bottomRows(event.X().size()) = ((Scalar)-1) * event.Y() * (RealVector::Ones(n) - (RealVector::Ones(n) - event.S().cwiseAbs2()).cwiseSqrt()).asDiagonal() * (event.Y().transpose() * inner(event.X(), density.X()) * density.Y());
            
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
            
            return (2 * this->S().array().log() * this->S().array().abs2()).sum();
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
            noalias(inners) = rho.Y().transpose() * inners * tau.Y() * tau.S().asDiagonal();

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
            
            plogq -= (inners * (-((1 - epsilon) * tau.S().array().abs2() + alpha_noise).log())  .sqrt().matrix().asDiagonal()).squaredNorm();
            
            
            // --- Compute tr(p log p)
            return this->entropy() - plogq;        
            
		}
        
        
    };
    
    
    
}

#endif