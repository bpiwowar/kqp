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
	class EvdUpdateResult {
    public:
        /**
         * The eigenvalues
         */
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> mD;
        
        /**
         * The eigenvectors
         */
        Eigen::MatrixXd mQ;
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
     * Fast rank one update of an EVD
     */
    class FastRankOneUpdate {
        /**
         * Used for deflation
         */
        double gamma;
        
    public:
        /**
         * Default constructor
         */
        FastRankOneUpdate();
               
        /**
         * @brief Rank one update
         * 
         * Computes the EVD of a rank-one perturbed eigenvalue decomposition
         * @param Z If given, instead of returning the eigenvectors matrix Q, returns \f$ Z Q \f$
         *        This allows to compute the eigenvalue decomposition of \f$ Z (D + \alpha z * z^\top \f$ 
         * 
         * @param keep Keep all the eigenvectors (even those of the not selected eigenvalues)
         *
         */
        void rankOneUpdate(const boost::shared_ptr<Eigen::MatrixXd>& Z, const Eigen::VectorXd& D, double rho, 
                             const Eigen::VectorXd& z,
                             bool computeEigenvectors, const Selector *selector, bool keep,
                             EvdUpdateResult &result);
        
        
    };
};

#endif
