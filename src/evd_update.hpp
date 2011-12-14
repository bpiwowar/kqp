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
#include "Eigen/Core"

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
        Eigen::Matrix<Real, Eigen::Dynamic, 1> mD;
        
        /**
         * The eigenvectors
         */
        Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> mQ;
	};
    
    /**
     * A list of eigenvalues that can be edited
     */
    class EigenList {
    public:
        virtual ~EigenList();
        
		/**
		 * Select an eigenvalue
		 */
        virtual double get(size_t i) const = 0;
        
		/**
		 * Remove this eigenvalue from the selection
		 */
		virtual void remove(size_t i) = 0;
        
		/**
		 * The original number of eigenvalues
		 */
		virtual size_t size() const = 0;
        
		/**
		 * The current number of selected
		 */
		virtual size_t getRank() const = 0;
        
		/**
		 * Check if an eigenvalue is currently selected or not
		 */
		virtual bool isSelected(size_t i) const = 0;
	};
    
    /**
     * Gets an eigenlist and removes whatever eigenvalues it does not like
     */
    class Selector {
    public:
        /**
         * @param eigenValues
         *            The ordered list of eigenvalues
         */
        virtual void selection(EigenList& eigenValues) const = 0;

    };
    
    /**
     * Chain selection
     */
    class ChainSelector : public Selector {
    public:
        ChainSelector();
        void add(const ChainSelector &);
        virtual void selection(EigenList& eigenValues) const;
    };

    /**
     * Minimum relative eigenvalue
     */
    template<typename Scalar>
    class MinimumSelector {
    public:
        MinimumSelector();
        virtual void selection(EigenList& eigenValues) const;
    };
    
    /**
     * Maximum rank
     */
    
    
    /**
     * Fast rank one update of an EVD
     */
    template <typename scalar>
    class FastRankOneUpdate {
        /**
         * Used for deflation
         */
        double gamma;
        
    public:
        typedef typename Eigen::NumTraits<scalar>::Real Real;
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RealVector;
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> ScalarVector;
        typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
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
                    double rho, const Eigen::Matrix<scalar, Eigen::Dynamic, 1> & z,
                    bool computeEigenvectors, const Selector *selector, bool keep,
                    EvdUpdateResult<scalar> &result,
                    ScalarMatrix * Z = 0);
        
        
    };
};

#endif
