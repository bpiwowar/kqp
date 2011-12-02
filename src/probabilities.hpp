/**
 * This module defines kernel quantum probabilities classes
 */



#ifndef __KQP_PROBABILITIES_H__
#define __KQP_PROBABILITIES_H__

#include "Eigen/Core"
#include "kernel_evd.hpp"

namespace kqp {
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
    template <class FVector> class KernelOperator {       
    public:
        typedef typename FVector::Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> Matrix;
        

        /**
         * Creates an object given a Kernel EVD
         * 
         * @param evd
         * @param copy If the object should be copied (safer, but slower)
         */
        KernelOperator(const OperatorBuilder<FVector>& evd, bool copy) {
            mX = copy ? evd.getX() : evd.getX().copy();
            mY.noalias() = (evd.getY() * evd.getZ()).eval();
            mS = copy ? evd.getEigenValues() : evd.getEigenValues().copy();
        }
        
        /**
         * Creates a one dimensional eigen-decomposition representation
         * 
         * @param list The list of vectors
         *           
         * @param copy If the object should be copied
         */
        KernelOperator(const boost::shared_ptr<FeatureMatrix<FVector> > & list, bool copy) {
            
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
        size_t getRank() const;
    
    protected:
        /**
         * The base vector list
         */
        boost::shared_ptr<FeatureMatrix<FVector> > mX;
        
        /**
         * The combination matrix.
         * 
         * In case of an EVD decomposition, mX mY is orthonormal
         */
        boost::shared_ptr<Matrix> mY;
        
        /**
         * The singular values
         *
         * This matrix is used only if the KernelOperator is in a 
         * EVD decomposed form
         */
        boost::shared_ptr<Eigen::DiagonalMatrix<double, Eigen::Dynamic> > mS;
        
        //! Is the decomposition othonormal, i.e. is Y^T X^T X Y the identity?
        bool orthonormal;
        
        //! Is the decomposition an observable, i.e. all the non null eigenvalues equal 1?
        bool observable;
         
    };
    
    template <class FMatrix> class Density;
    
    /**
     * A document subspace is defined by the basis vectors (matrix {@linkplain #mU}
     * ). A diagonal matrix ({@linkplain #mS}) defines the weights associated to the
     * basis vectors
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     * 
     */
    template <class FMatrix> class Event : public KernelOperator<FMatrix>  {
    public:        
        /**
         * Construct a Event from a kernel EVD. See
         * {@linkplain KernelEigenDecomposition#KernelEigenDecomposition(OperatorBuilder, bool)}
         */
        Event(const OperatorBuilder<FMatrix> &evd, bool deepCopy) {
            
        }
        
        friend class Density<FMatrix>;
    };
    
    /**
     * A probability density
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class FMatrix> class Density: public KernelOperator<FMatrix>  {
    public: 
        typedef typename KernelOperator<FMatrix>::Matrix Matrix;
        typedef typename KernelOperator<FMatrix>::Vector Vector;
        typedef typename KernelOperator<FMatrix>::FVector FVector;
        typedef typename KernelOperator<FMatrix>::scalar scalar;
        typedef typename KernelOperator<FMatrix>::Real real;
        
        /**
         * Creates a new density
         * 
         * @param evd
         * @param deepCopy
         */
        Density(const OperatorBuilder<FMatrix>& evd, bool deepCopy);
        
        /**
         * Compute the probability of an event
         * 
         * @param subspace
         *            The event
         * @param fuzzyEvent
         *            The event should be considered as "fuzzy" -- i.e. each
         *            dimension is weighted by the corresponding sigma
         * @return The probability
         */
        real computeProbability(const Event<FMatrix>& subspace,
                                  bool fuzzyEvent) const {
            Matrix result = getProbabilityMatrix(subspace, fuzzyEvent);
            
            if (result.rows() == 0) return 0;
            
            return result.squared_norm();
        }
        
        /**
         * Pre-computation of a probability. This method is shared by others.
         * 
         * <p>
         * Given the subspace representation \f$ S U \f$ and the density
         * \f$ S^\prime U^\prime \f$, computes \f$ U^\top U^\prime S^\prime \f$ (crisp
         * subspace) or \f$ S U^\top U^\prime S^\prime\f$ (fuzzy subspace). The sum of
         * the squares of the matrix correspond to the probability of observing the
         * subspace given the density.
         * </p>
         * 
         * <p>
         * Each row of the resulting matrix correspond to the probability associated
         * with one of the dimension of the subspace. Similarly, each column is
         * associated to a dimension of the density.
         * </p>
         * 
         * @param subspace The subspace
         * @param fuzzyEvent True if the event should be considered as fuzzy
         * @return A matrix where each row correspond to one dimension of the density, 
         *         and each column to one dimension of the subspace
         */
        Matrix getProbabilityMatrix(const Event<FMatrix>& subspace,
                                    bool fuzzyEvent) const {
            // Compute Y_s^T (P) Y_d S_d
            Matrix mP = subspace.mY.transpose() * subspace.mX.computeInnerProducts(this->mX) * this->mY * this->mS;
            
            // Pre-multiply the result by S_s if using fuzzy subspaces
            if (fuzzyEvent)
                mP = subspace.mS * mP;
            
            return mP;
        }
        /**
         * Get the matrix v^T * (U x S) where U is the basis and S is the square
         * root of the eigenvalues. The Froebenius norm of the resulting matrix is
         * the probability of the event defined by v x v^t
         * 
         * @param vector
         *            The vector v
         * @return
         */
        Vector getProbabilityMatrix(const FVector& vector) const {
            return this->mX.computeInnerProducts(vector) * this->mY * this->mS;
        }
        
        /**
         * @brief Computes the divergence with another density
         *
         * Computes the divergence defined in the paper
         * "Conditional expectation in an operator algebra. IV. Entropy and information" by H. Umegaki (1962),
         * at page 69. The formula is:
         * \f[ J(\rho || \tau) = tr(\rho \log (\rho) - \rho \log(\tau))) \f]
         */
        double computeDivergence(const Density<FMatrix> &tau) {
            BOOST_THROW_EXCEPTION(not_implemented_exception());
        }
        

    };
    
    
    
}

#endif