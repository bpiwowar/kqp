/**
 * This module defines kernel quantum probabilities classes
 */
//
//  probabilities.h
//  kqp
//
//  (c) Benjamin Piwowarski on 26/05/2011.
//



#ifndef __KQP_PROBABILITIES_H__
#define __KQP_PROBABILITIES_H__

#include "Eigen/Core"
#include "kernel_evd.h"

namespace kqp {
    /**
     * Common class shared by fuzzy subspaces and densities
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
    template <typename scalar, class F> class KernelOperator {       
    public:
        typedef typename FeatureMatrix<scalar,F>::Matrix Matrix;
        typedef typename FeatureMatrix<scalar,F>::Vector Vector;

        /**
         * Creates an object given a Kernel EVD
         * 
         * @param evd
         * @param copy If the object should be copied (safer, but slower)
         */
         KernelOperator(const DensityBuilder<scalar, F>& evd, bool copy);
        
        /**
         * Creates a one dimensional eigen-decomposition representation
         * 
         * @param list The list of vectors
         *           
         * @param copy If the object should be copied (safer, but slower)
         */
        KernelOperator(const FeatureMatrix<scalar, F>& list, bool copy);
        
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
        FeatureMatrix<scalar, F> mX;
        
        /**
         * The combination matrix.
         * 
         * In case of an EVD decomposition, mX mY is orthonormal
         */
        Matrix mY;
        
        /**
         * The singular values
         *
         * This matrix is used only if the KernelOperator is in a 
         * EVD decomposed form
         */
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> mS;
    };
    
    template <typename scalar, class F> class Density;
    
    /**
     * A document subspace is defined by the basis vectors (matrix {@linkplain #mU}
     * ). A diagonal matrix ({@linkplain #mS}) defines the weights associated to the
     * basis vectors
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     * 
     */
    template <typename scalar, class F> class Subspace : public KernelOperator<scalar, F>  {
    public:        
        /**
         * Construct a Subspace from a kernel EVD. See
         * {@linkplain KernelEigenDecomposition#KernelEigenDecomposition(DensityBuilder, bool)}
         */
        Subspace(const DensityBuilder<scalar, F> &evd, bool deepCopy);  
        
        friend class Density<scalar, F>;
    };
    
    /**
     * A probability density
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <typename scalar, class F> class Density: public KernelOperator<scalar, F>  {
    public: 
        typedef typename KernelOperator<scalar,F>::Matrix Matrix;
        typedef typename KernelOperator<scalar,F>::Vector Vector;
        
        /**
         * Creates a new density
         * 
         * @param evd
         * @param deepCopy
         */
        Density(const DensityBuilder<scalar, F>& evd, bool deepCopy);
        
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
        double computeProbability(const Subspace<scalar, F>& subspace,
                                  bool fuzzyEvent) const;
        
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
        Matrix getProbabilityMatrix(const Subspace<scalar, F>& subspace,
                                                   bool fuzzyEvent) const;        
        /**
         * Get the matrix v^T * (U x S) where U is the basis and S is the square
         * root of the eigenvalues. The Froebenius norm of the resulting matrix is
         * the probability of the event defined by v x v^t
         * 
         * @param vector
         *            The vector v
         * @return
         */
        Vector getProbabilityMatrix(const F& vector) const {
            return this->mX.computeInnerProducts(vector) * this->mY * this->mS;
        }
        

    };
    
    
    
}

#endif