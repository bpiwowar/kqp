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


#include <kqp/kqp.hpp>
#include <kqp/alt_matrix.hpp>
#include <kqp/kernel_evd.hpp>
#include <kqp/evd_utils.hpp>


namespace kqp {
    
    // Foward declarations
    template <typename Scalar> class Density;
    template <typename Scalar> class Event;
    
    
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
    template <typename Scalar> class KernelOperator {       
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        /**
         * Creates an object given a Kernel EVD
         * 
         * @param evd The kernel EVD 
         */
        KernelOperator(const KernelEVD<Scalar> & evd) : m_operator(evd.getDecomposition()) {
            m_operator.squareRoot(true);
        }
        
        /**
         * Creates an object given a decomposition
         * 
         * @param evd The decomposition (in orthonomormal form)
         * @param takeSqrt Takes the square root of the eigen values
        */
        KernelOperator(const Decomposition<Scalar> & d) : m_operator(d) {
            m_operator.squareRoot(true);
        }
        
        
        /**
         * \brief Creates a new kernel operator
         */
        KernelOperator(const FSpaceCPtr &fs, const FMatrixCPtr &mX, const ScalarAltMatrix &mY, const RealVector &mS, bool orthonormal) 
            : m_operator(fs, mX,mY,mS,orthonormal,false) {
                m_operator.squareRoot(true);
        }
        
        
        /**
         * Trim the eigenvalue decomposition to a lower rank. If the rank 
         * is already lower, don't do anything.
         * 
         * @param newRank
         *            The new rank of the subspace
         */
        void trim(Index newRank) {
            this->orthonormalize();

            // Don't do anything
            if (newRank <= getRank()) return;

            m_operator.mY.conservativeResize(m_operator.mY.rows(), newRank);
            m_operator.mD.conservativeResize(newRank, 1);
        }
        
        /**
         * Get the rank of the operator
         */
        Index getRank() const {
            if (isOrthonormal()) 
                return m_operator.mD.size();
            return -1;
        }
        
        //! Get the feature space
        inline const FSpaceBase &fs() const {
            return *m_operator.fs;
        }
        
        //! Get the feature matrix
        inline const FMatrixBase &X() const {
            return *m_operator.mX;
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
        
        inline bool isOrthonormal() const {
            return m_operator.orthonormal;
        }
        
        /**
         * Compute the squared norm of the operator
         */
        Real squaredNorm() {
            if (isOrthonormal()) 
                return m_operator.mD.squaredNorm();
            
            return m_operator.fs->k(X(), Y(), S()).squaredNorm();
        }
        
        /**
         * @brief Compute the inner products of the scaled basis of the operators 
         * @param that The other operator
         * @return A matrix where each row correspond to one dimension of the density, 
         *         and each column to one dimension of the subspace
         */
        ScalarMatrix inners(const KernelOperator<Scalar>& that) const {
            return m_operator.fs->k(m_operator.mX, m_operator.mY, m_operator.mD,
                                   that.m_operator.mX, that.m_operator.mY, that.m_operator.mD);
        }
        
        /**
         * @brief Get the matrix corresponding to the decomposition.
         * @throws An exception if the feature matrix cannot be linearly combined
         */
        FMatrixPtr matrix() const {
            return m_operator.fs->linearCombination(X(), ScalarMatrix(Y() * S().asDiagonal()));
        }
        
       
    protected:
        
        //! Orthonormalize the decomposition (non const version)
        void _orthonormalize() {
            m_operator.orthonormalize();    
        }
        
        //! The current decomposition
        Decomposition<Scalar> m_operator;
        friend class Density<Scalar>;
        friend class Event<Scalar>;
    };
    
    
    
    /**
     * A document subspace is defined by the basis vectors (matrix {@linkplain #mU}
     * ). A diagonal matrix ({@linkplain #mS}) defines the weights associated to the
     * basis vectors
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     * 
     */
    template <typename Scalar> 
    class Event : public KernelOperator<Scalar>  {
        bool useLinearCombination;
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        /**
         * Construct a Event from a kernel EVD. See
         * {@linkplain KernelEigenDecomposition#KernelEigenDecomposition(KernelEVD, bool)}
         * @param evd The kernel EVD decomposition
         */
        Event(const KernelEVD<Scalar>  &evd, bool fuzzy = false) 
        : KernelOperator<Scalar>(evd), 
        useLinearCombination(evd.getFSpace()->canLinearlyCombine()) {       
            if (!fuzzy) {
                this->orthonormalize();
                this->m_operator.mD = RealVector::Ones(this->Y().cols());
            }
        }
        
        /** Construct an event from a basis */
        Event(const FSpaceCPtr &fs, const FMatrixCPtr &mX, bool orthonormal)
        : KernelOperator<Scalar>(fs, mX, Eigen::Identity<Scalar>(mX->size(),mX->size()), RealVector::Ones(mX->size()), orthonormal) {
        }
        
        /** Construct an event from a basis */
        Event(const FSpaceCPtr &fs, const FMatrixCPtr &mX, const ScalarAltMatrix &mY, bool orthonormal) 
        : KernelOperator<Scalar>(fs, mX, mY, RealVector::Ones(mX->size()), orthonormal) {
        }
        
        Event(const Decomposition<Scalar> & d, bool fuzzy) : KernelOperator<Scalar>(d) {
            if (!fuzzy) {
                this->orthonormalize();
                this->m_operator.mD = RealVector::Ones(this->Y().cols());
            }            
        }
        
        //! (debug) Manually sets the linear combination
        void setUseLinearCombination(bool b) {
            this->useLinearCombination = b;
        }
        

        
        friend class Density<Scalar>;
#ifndef SWIG        
        static Density<Scalar> projectWithLC(const Density<Scalar>& density, const Event<Scalar> &event) {        
            event.orthonormalize();
            ScalarMatrix lc;
            noalias(lc) = event.Y() * event.m_operator.k(density.m_operator);
            
            FMatrixPtr mX = event.m_operator.fs->linearCombination(event.X(), lc);
            ScalarAltMatrix mY = Eigen::Identity<Scalar>(mX->size(),mX->size());
            RealVector mS = RealVector::Ones(mY.cols());
            return Density<Scalar>(event.m_operator.fs, mX, mY, mS, false);
        }
        
        static Density<Scalar> projectOrthogonalWithLC(const Density<Scalar>& density, const Event<Scalar> &event) {
            // We need the event to be in an orthonormal form
            event.orthonormalize();
            
            Index n = event.S().rows();
            
            // FIXME
            RealVector s = event.S();
            ScalarAltMatrix e_mY(event.Y() * (RealVector::Ones(n) - (RealVector::Ones(n) - s.cwiseAbs2()).cwiseSqrt()).asDiagonal() 
                                 * event.m_operator.fs->k(event.X(), event.Y(), density.X(), density.Y()));
            
            FMatrixPtr mX = density.m_operator.fs->linearCombination(density.X(), density.Y(), 1., event.X(), e_mY, -1.);
            
            return Density<Scalar>(event.m_operator.fs, mX, Eigen::Identity<Scalar>(mX->size(),mX->size()), density.S(), false);
        }
        
        
        static Density<Scalar> project(const Density<Scalar>& density, const Event<Scalar> &event) {        
            event.orthonormalize();
            FMatrixPtr mX = event.X().copy(); //event.m_operator.fs->newMatrix(event.X());
            ScalarAltMatrix mY(event.Y() * event.m_operator.k(density.m_operator));
            RealVector mS = RealVector::Ones(mY.cols());
            return Density<Scalar>(event.m_operator.fs, mX, mY, mS, false);
        }
        
        static Density<Scalar> projectOrthogonal(const Density<Scalar>& density, const Event<Scalar> &event) {
            event.orthonormalize();
            
            // Concatenates the vectors
            FMatrixPtr mX = density.X().copy(); //event.m_operator.fs->newMatrix(density.X());
            mX->add(event.X());
            
            // Computes the new linear combination matrix
            ScalarMatrix _mY(density.X().size() + event.X().size(), density.S().rows());
            RealVector s = event.S();
            Index n = s.rows();
            
            _mY.topRows(density.X().size()) = density.Y();
            _mY.bottomRows(event.X().size()) = ((Scalar)-1)
                * (event.Y() * (RealVector::Ones(n) - (RealVector::Ones(n) - s.cwiseAbs2()).cwiseSqrt()).asDiagonal() * event.m_operator.fs->k(event.X(), event.Y(), density.X(), density.Y()));
            
            ScalarAltMatrix mY; 
            mY.swap(_mY);
            
            
            // Return
            return Density<Scalar>(event.m_operator.fs, mX, mY, density.S(), false);
        }
#endif
                
        /**
         * @brief Project onto a subspace.
         */
        
        Density<Scalar> project(const Density<Scalar>& density, bool orthogonal = false, bool normalize = true) {
            // Orthogonal projection
            if (orthogonal) {
                Density<Scalar> r = useLinearCombination ? projectOrthogonalWithLC(density, *this) : projectOrthogonal(density, *this);
                if (normalize)
                    r.normalize();
                return r;
            } 

            // Normal projection
            Density<Scalar> r = useLinearCombination ? projectWithLC(density, *this) : project(density, *this);
            if (normalize)
                r.normalize();
            return r;
        }
    };
    
    /**
     * A probability density
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <typename Scalar> class Density: public KernelOperator<Scalar>  {
    public: 
        KQP_SCALAR_TYPEDEFS(Scalar);        
        
        /**
         * Creates a new density
         * 
         * @param evd
         * @param deepCopy
         */
        Density(const KernelEVD<Scalar> & evd) : KernelOperator<Scalar>(evd) {
        }
        
        Density(const FSpaceCPtr &fs, const FMatrixCPtr &mX, const ScalarAltMatrix &mY, const RealVector &mD, bool orthonormal) : KernelOperator<Scalar>(fs, mX, mY, mD, orthonormal) {
        }
        
        Density(const FSpaceCPtr &fs, const FMatrixCPtr &mX, bool orthonormal) 
            : KernelOperator<Scalar>(fs, mX, Eigen::Identity<Scalar>(mX->size(),mX->size()), RealVector::Ones(mX->size()), orthonormal) {
        }
        
        Density(const Decomposition<Scalar> & d) : KernelOperator<Scalar>(d) {
        }
        
        
        //! Normalise the density
        void normalize() {
            this->m_operator.traceNormalize();
        }
        
        /**
         * Compute the probability of an event
         * 
         * @param event The event for which the probability is computed
         *            The event
         * @return The probability
         */
        Real probability(const Event<Scalar>& event) const {
            return this->m_operator.k(event.m_operator).squaredNorm();
        }

        /**
         * Computes the probabilities associated with each eigen vector 
         * @param event The event for which probabilities should be computed
         * @param noEigenValues Do not multiply by the eigen values
         */
        RealVector eigenProbabilities(const Event<Scalar>& event, bool noEigenValues) const {
            this->orthonormalize();

            auto &other = event.m_operator;
            
            if (noEigenValues)
                return this->m_operator.fs->k(
                            this->X(), this->Y(), RealVector::Ones(this->Y().cols()), 
                            *other.mX, other.mY, other.mD)
                        .rowwise().squaredNorm();

            return this->m_operator.k(other).rowwise().squaredNorm();
        }
        
        
        //! Computes the entropy
        Real entropy() const {
            this->orthonormalize();
            RealVector s = this->S();
            return - (2 * s.array().log() * s.array().abs2()).sum();
        }

        Real computeDivergence(const Density<Scalar> &tau) const {
            return computeDivergence(tau, epsilon());
        }
        
        /**
         * @brief Computes the divergence with another density
         *
         * Computes the divergence defined in the paper
         * "Conditional expectation in an operator algebra. IV. Entropy and information" by H. Umegaki (1962),
         * at page 69. The formula is:
         * \f[ J(\rho || \tau) = tr(\rho \log (\rho) - \rho \log(\tau))) \f]
         */
        Real computeDivergence(const Density<Scalar> &tau, Real epsilon) const {
            const Density<Scalar> &rho = *this;
            
            // --- Requires orthonormal decompositions
            rho.orthonormalize();
            tau.orthonormalize();
            
            // --- Notation
            
            ScalarMatrix inners = this->m_operator.fs->k(rho.X(), tau.X());
            noalias(inners) = rho.S().asDiagonal() * rho.Y().adjoint() * inners * tau.Y(); 
            
            // --- Compute tr(p log q)
            Scalar plogq = 0;
            
            
			// The background density span the subspace of rho
            Index rank = rho.S().rows() + tau.S().rows();
            Index dimension = rho.fs().dimension();
            if (rank > dimension && dimension > 0) rank = dimension;
            
			Real alpha = 1. / (Real)(rank);
			Real alpha_noise = epsilon * alpha;
            
			// Includes the smoothing probability if not too small
            if (epsilon >= kqp::epsilon()) 
                plogq = log(alpha_noise) * (1. - inners.squaredNorm());
            
			
			// Main computation
			RealVector mD(tau.S().rows());
            
            RealVector tau_S = tau.S();
            
            plogq -= (inners * (-((1 - epsilon) * tau_S.array().abs2() + alpha_noise).log())  .sqrt().matrix().asDiagonal()).squaredNorm();
            
            
            // --- Compute tr(p log p)
            return - plogq - this->entropy();        
            
		}
        
        
    };
    
    
    
}

#endif