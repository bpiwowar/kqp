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
#include "Eigen/Cholesky"
#include "kqp.h"
#include "coneprog.h"
#include <vector>
#include <boost/shared_ptr.hpp>

namespace kqp {
    using Eigen::Dynamic;
    
    /** By default, vectors cannot be combined */
    template<typename scalar, typename T> 
    struct linear_combination { 
        static const bool canCombine = false; 
        
        static T combine(const scalar &a, const T&, const scalar &b, const T*) {
            BOOST_THROW_EXCEPTION(illegal_operation_exception());
        };
    };
        
    /**
     * @brief Base for all feature matrix classes
     * 
     * This class holds a list of vectors whose exact representation might not be
     * known. All sub-classes must implement basic list operations (add, remove).
     * Vectors added to the feature matrix are considered as immutable by default
     * since they might be kept as is.
     * 
     * @ingroup FeatureMatrix
     * @param scalar A valid scalar (float, double, std::complex)
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <typename scalar, class F> class FeatureMatrix {
        typedef Eigen::Matrix<scalar, Dynamic, Dynamic> Matrix;
        typedef Eigen::Matrix<scalar, Dynamic, 1> Vector;
        typedef typename Vector::Index Index;
        
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
         * @brief Linear combination of vectors
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
        virtual const F& get(Index i) const = 0;

        /** Get the i<sup>th</sup> feature vector */
        virtual const F& add(const F &f) const = 0;

        /** Get the i<sup>th</sup> feature vector */
        virtual const F& set(Index i, const F &f) const = 0;

        /** 
          * Remove the i<sup>th</sup> feature vector 
          * @param if swap is true, then the last vector will be swapped with one to remove (faster)
          */
        virtual const Index remove(Index i, bool swap = false) const = 0;
};
    
    /**
     * @brief A class where vectors know how to multiply themselves, i.e. the function kqp::k(a,b) returns a scalar
     * @ingroup FeatureMatrix
     * @param If there exists a combiner
     */
    template <typename scalar, class F> class FeatureList : public FeatureMatrix<scalar, F> {
        std::vector<F> list;
        typedef typename FeatureMatrix<scalar, F>::Index Index;
  
    public:
        Index size() const { return this->list.size(); }
        const F& get(Index i) const { return this->list[i]; }
    };
    
    
    /**
     * @brief A feature matrix where vectors 
     * @ingroup FeatureMatrix
     */
    template <typename scalar> class ScalarMatrix : public FeatureMatrix<scalar, Eigen::Matrix<scalar, Dynamic, 1> > {
        typedef Eigen::Matrix<scalar, Dynamic, Dynamic> Matrix;
        typedef Eigen::Matrix<scalar, Dynamic, 1> F;
        typedef typename Matrix::Index Index;
        
        Matrix matrix;
    public:
		virtual ~ScalarMatrix();
        long size() const { return this->matrix.cols();  }
        const F& get(size_t i) const { return this->matrix.col(i); }
        
        virtual const F& add(const F &f) const {
            Index n = matrix.cols();
            matrix.resize(n + 1);
            matrix.col(n) = f;
        }
        virtual const F& set(Index i, const F &f) const {
            matrix.col(i) = f;
        }
        virtual const F& remove(Index i, bool swap) const {
            Index last = matrix.cols() - 1;
            if (swap) {
                if (i != last) 
                    matrix.col(i) = matrix.col(last);
            } else {
                for(Index j = i + 1; j <= last; j++)
                    matrix.col(i-1) = matrix.col(i);
            }
            matrix.resize(last);
        }

        
    };
    
    /**
     * @brief Builds a compact representation of an hermitian operator. 
     *
     * Computes \f$ \mathfrak{U} = \sum_{i} \alpha_i \mathcal U A A^\top \mathcal U ^ \top   \f$
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     *
     * @param <scalar>
     *            The scalar type
     * @param <F>
     *            The type of the base vectors in the original space
     * @ingroup OperatorBuilder
     */
    template <typename scalar, class F> class OperatorBuilder {
        
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
        
        /**
         * Copy the operator builder
         * @param fullCopy is true if the current data should also be copied
         */
        virtual void copy(bool fullCopy) = 0;       
        
    protected:
        List list;
    };
    
    
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup OperatorBuilder
     */
    template <typename scalar, class F> class DivideAndConquerBuilder : public OperatorBuilder<scalar, F> {
    public:
        virtual void add(double alpha, const F &v);
    };
    
    /**
     * @brief Accumulation based computation of the density.
     *
     * Stores the vectors and performs an EVD at the end
     * 
     * @ingroup OperatorBuilder
     */
    template <typename scalar, class F> class AccumulatorBuilder : public OperatorBuilder<scalar, F> {
    public:
        virtual void add(double alpha, const F &v);
    };
    
    
    /**
     * Direct computation of the density (i.e. matrix representation)
     * 
     * @ingroup OperatorBuilder
     */
    template <typename scalar, class F> class DirectBuilder : public OperatorBuilder<scalar, F> {
    public:
        virtual void add(double alpha, const F &v);
    };
    
    
    
    
    
    /**
     * The KKT pre-solver to solver the QP problem
     * 
     */
    class KQP_KKTPreSolver : public cvxopt::KKTPreSolver {
        Eigen::LLT<Eigen::MatrixXd> lltOfK;
        Eigen::MatrixXd B, BBT;
        
    public:
        KQP_KKTPreSolver(const Eigen::MatrixXd& gramMatrix);
        
        cvxopt::KKTSolver *get(const cvxopt::ScalingMatrix &w);
    };
    

    
} // end namespace

#endif
