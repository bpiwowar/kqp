//
//  kernel_evd.h
//  kqp
// 
//  This file contains all the virtual definitions for the classes needed
//
//  Created by Benjamin Piwowarski on 18/05/2011.
//  Copyright 2011 Benjamin Piwowarski. All rights reserved.
//

#ifndef __KQP_KERNEL_EVD_H__
#define __KQP_KERNEL_EVD_H__

#include "Eigen/Core"
#include "kqp.h"
#include <vector>
#include <boost/shared_ptr.hpp>

namespace kqp {
    using Eigen::Dynamic;
    
    /** Numerical zero when approximating */
    extern double EPSILON; 
    
    /** By default, vectors cannot be combined */
    template<typename scalar, typename T> 
    struct linear_combination { 
        static const bool canCombine = false; 
        
        static T combine(const scalar &a, const T&, const scalar &b, const T*) {
            BOOST_THROW_EXCEPTION(illegal_operation_exception());
        };
    };
        
    /**
     * This class holds a list of vectors whose exact representation might not be
     * known. All sub-classes must implement basic list operations (add, remove).
     * 
     * @param scalar A valid scalar (float, double, std::complex)
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <typename scalar, class F> class FeatureMatrix {
        typedef Eigen::Matrix<scalar, Dynamic, Dynamic> Matrix;
        typedef Eigen::Matrix<scalar, Dynamic, 1> Vector;
        
        /**
         * Computes the inner product between a given vector and the i<sup>th</sup>
         * vector of the list
         * 
         * @param i
         *            The index of the vector in the list
         * @param vector
         *            The vector provided
         * @return The inner product between both vectors
         */
        virtual scalar computeInnerProduct(int i, const F& vector) const;
        
        /**
         * Compute the inner product with another kernel matrix.
         *
         *
         */
        Matrix computeInnerProducts(const FeatureMatrix<scalar, F>& other) const;
        
        /**
         * Compute the inner products with each of the vectors. This method might be
         * re-implemented for efficiency reason, and by default calls iteratively
         * {@linkplain #computeInnerProduct(int, Object)}.
         * 
         * @param vector
         *            The vector with which the inner product is computed each time
         * @return A series of inner products, one for each of the base vectors
         */
        virtual Vector computeInnerProducts(const F& vector) const;
        
        /**
         * Computes the Gram matrix of the vectors contained in this list
         * 
         * @return A dense symmetric matrix
         */
        virtual Matrix computeGramMatrix() const;
        
        typedef linear_combination<scalar, F> Combiner;
        
        /**
         * Returns true if the vectors can be linearly combined
         */
        virtual bool canLinearlyCombine() {
            return Combiner::canCombine;
        }
        
        /** 
         * Linear combination of vectors
         *
         * In this implementation, we just use the pairwise combiner if it exists
         */
        virtual F linearCombination(const Vector & lambdas) {
            if (lambdas.size() != this->size()) 
                BOOST_THROW_EXCEPTION(illegal_argument_exception());
            
            // Just return the scaled vector if we have only one
            if (lambdas.size() == 1) {
                return Combiner::combine(lambdas[0], this->get(0), 0, (const F*)0);
            }
            
            F result = this->get(0);
            for(long i = 0; i < lambdas.rows() - 1; i++) 
                result = Combiner::combine(lambdas[i], result, lambdas[i+1], &this->get(i));
            
            return result;
        }
        
        /** Get the number of feature vectors */
        virtual long size() const = 0;
        
        /** Get the i<sup>th</sup> feature vector */
        virtual const F& get(size_t i) const = 0;

        /** Get the i<sup>th</sup> feature vector */
        virtual const F& add(const F &f) const = 0;

        /** Get the i<sup>th</sup> feature vector */
        virtual const F& set(size_t i, const F &f) const = 0;

        /** Get the i<sup>th</sup> feature vector */
        virtual const F& remove(size_t i) const = 0;
};
    
    /**
     * A class where vectors know how to multiply themselves, i.e. the function kqp::k(a,b) returns a scalar
     * @param If there exists a combiner
     */
    template <typename scalar, class F> class FeatureList : public FeatureMatrix<scalar, F> {
        std::vector<F> list;
        
    public:
        size_t size() const { return this->list.size(); }
        const F& get(size_t i) const { return this->list[i]; }
    };
    
    
    /**
     * Case where the feature vectors are dense vectors
     */
    template <typename scalar> class ScalarMatrix : public FeatureMatrix<scalar, Eigen::Matrix<scalar, Dynamic, 1> > {
        typedef Eigen::Matrix<scalar, Dynamic, Dynamic> Matrix;
        typedef Eigen::Matrix<scalar, Dynamic, 1> F;
        
        Matrix matrix;
    public:
		virtual ~ScalarMatrix();
        long size() const { return this->matrix.cols();  }
        const F& get(size_t i) const { return this->matrix.col(i); }
        
        virtual const F& add(const F &f) const {}
        virtual const F& set(size_t i, const F &f) const;
        virtual const F& remove(size_t i) const;

        
    };
    
    /**
     * Builds a compact representation of an hermitian operator. 
     *
     * Computes \f$ \mathfrak{U} = \sum_{i} \alpha_i \mathcal U A A^\top \mathcal U ^ \top   \f$
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     *
     * @param <scalar>
     *            The scalar type
     * @param <F>
     *            The type of the base vectors in the original space
     */
    template <typename scalar, class F> class DensityBuilder {
        
    public:
        typedef FeatureMatrix<scalar, F> List;
        typedef typename List::Matrix Matrix;
        
        /** Get the feature vector combination matrix */
        Matrix getY();
        
        /** Get the a diagonal matrix of eigenvalues (or a 0 x 0 matrix if it does not define one) */
        virtual Eigen::Diagonal<double, Eigen::Dynamic> getEigenvalues();

        /**
         * Get the list of feature vectors
         */
        const List &getX() const { return list; }
        List &getX() { return list; }

        
        /**
         * Add a new vector to the density
         *
         * Computes \f$A^\prime \approx A + \alpha   X A A^T  X^\top\f$
         * 
         * @param alpha
         *            The coefficient for the update
         * @param mX  The feature matrix X
         * @param mA  The 
         */
        virtual void add(double alpha, const FeatureMatrix<scalar, F> &mX, const Matrix &mA) = 0;
        
    protected:
        List list;
    };
    
    /**
     * Direct computation of the density
     */
    template <typename scalar, class F> class DirectBuilder : public DensityBuilder<scalar, F> {
    public:
        virtual void add(double alpha, const F &v);
    };
    
 
} // end namespace

#endif
