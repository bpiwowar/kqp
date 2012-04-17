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
#ifndef __KQP_FASTRANKONEUPDATE_H__
#define __KQP_FASTRANKONEUPDATE_H__

#include <boost/intrusive_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <kqp/kqp.hpp>
#include <Eigen/Core>
#include <kqp/rank_selector.hpp>

namespace kqp {

    /**
	 * A result from a rank one update
	 * 
	 * @author B. Piwowarski <benjamin@bpiwowar.net>
	 */
	template <typename scalar> class EvdUpdateResult {
    public:
        typedef typename Eigen::NumTraits<scalar>::Real Real;

        /**
         * The eigenvalues
         */
        Eigen::Matrix<Real,Dynamic,1> mD;
        
        /**
         * The eigenvectors
         */
        Eigen::Matrix<scalar,Dynamic,Dynamic> mQ;
	};
    

    /**
     * Maximum rank
     */
    
    
    /**
     * Fast rank one update of an EVD
     */
    template <typename Scalar>
    class FastRankOneUpdate {
        /**
         * Used for deflation
         */
        double gamma;
        
    public:
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        typedef Eigen::Matrix<Real,Dynamic,1> RealVector;
        typedef Eigen::Matrix<Real,Dynamic,1> ScalarVector;
        typedef Eigen::Matrix<Scalar,Dynamic,Dynamic> ScalarMatrix;
        
        /**
         * Default constructor
         */
        FastRankOneUpdate();
               
        /**
         * @brief Rank one update
         * 
         * Computes the EVD of a rank-one perturbed eigenvalue decomposition
         * \f$ Z(D + \rho z z^\dagger) \f$ where 
         * \f$D\f$ is a diagonal matrix with real values and \f$z\f$ is a vector (complex or real field).
         *
         * @param Z If given, instead of returning the eigenvectors matrix Q, update \f$Z\f$ as \f$ Z Q \f$
         *        This allows to compute the eigenvalue decomposition of \f$ Z (D + \alpha z * z^\dagger) Z^\dagger \f$ 
         * @param D the diagonal matrix (all the values must be real)
         * @param rho the coefficient
         * @param keep Keep all the eigenvectors (even those of the not selected eigenvalues)
         *
         */
        void update(const RealVector & D, 
                    double rho, const Eigen::Matrix<Scalar,Dynamic,1> & z,
                    bool computeEigenvectors, const Selector<Real> *selector, bool keep,
                    EvdUpdateResult<Scalar> &result,
                    ScalarMatrix * Z = 0);
        
        
    };
};

#endif
