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

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#include "Eigen/Core"
#include "Eigen/Cholesky"
#include "Eigen/Eigenvalues"

#include "intrusive_ptr_object.hpp"

#include "kqp.hpp"
#include "coneprog.hpp"

namespace kqp {
    using Eigen::Dynamic;
    
    /** By default, vectors cannot be combined */
    template<typename FVector> 
    struct linear_combination { 
        //! Scalar
        typedef typename FVector::Scalar Scalar;
        
        //! Cannot combine by default
        static const bool canCombine = false; 
        
        /**
         * Computes x <- x + b * y
         */
        static void axpy(const FVector& x, const Scalar &b, const FVector &y) {
            BOOST_THROW_EXCEPTION(illegal_operation_exception());
        };
    };
     

    /**
     * Defines a generic inner product that has to be implemented
     * as a last resort
     */
    template <class FVector>
    typename FVector::Scalar inner(const FVector &a, const FVector &b) {
        BOOST_THROW_EXCEPTION(not_implemented_exception());
    }


    /**
     * @brief Base for all feature matrix classes
     * 
     * This class holds a list of vectors whose exact representation might not be
     * known. All sub-classes must implement basic list operations (add, remove).
     * Vectors added to the feature matrix are considered as immutable by default
     * since they might be kept as is.
     * 
     * @ingroup FeatureMatrix
     * @param _FVector the type of the feature vectors
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class _FVector> 
    class FeatureMatrix : public boost::intrusive_ptr_base {
    public:
        
        //! Own type
        typedef FeatureMatrix<_FVector> Self;
        
        //! Feature vector type
        typedef _FVector FVector;

        //! Scalar type
        typedef typename FVector::Scalar Scalar;
        
        //! Matrix type for inner products
        typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> InnerMatrix;
        typedef Eigen::Matrix<Scalar, Dynamic, 1> Vector;
        
        /**
         * @brief Computes the inner product between the i<sup>th</sup>
         * vector of the list and the given vector.
         *
         * By default, uses the inner product between any two vectors of this type 
         * 
         * @param i
         *            The index of the vector in the list
         * @param vector
         *            The vector provided
         * @return The inner product between both vectors
         */
        virtual Scalar inner(Index i, const FVector& vector) const {
            return kqp::inner<FVector>(get(i), vector);
        }
                
        /**
         * Compute the inner products with each of the vectors. This method might be
         * re-implemented for efficiency reason, and by default calls iteratively
         * {@linkplain #computeInnerProduct(int, Object)}.
         * 
         * @param vector
         *            The vector with which the inner product is computed each time
         * @return A series of inner products, one for each of the base vectors
         */
        virtual Vector inner(const FVector& vector) const {
            typename FeatureMatrix::Vector innerProducts(size());
            for (int i = size(); --i >= 0;)
                innerProducts[i] = kqp::inner<FVector>(get(i), vector);
            return innerProducts;
        }
                
        
        
        /**
         * @brief Compute the inner product with another feature matrix.
         *
         * @warning m will be modified (Eigen const trick)
         */
        template <typename DerivedMatrix>
        void inner(const FeatureMatrix<FVector>& other, const Eigen::MatrixBase<DerivedMatrix> &m) const {
            // Eigen-const-cast-trick
            Eigen::MatrixBase<DerivedMatrix> &r = const_cast< Eigen::MatrixBase<DerivedMatrix> >(m);
            
            // Resize if this is possible
            r.derived().resize(size(), other.size());
            
            for (int i = 0; i < size(); i++)
                for (int j = i; j < other.size(); j++)
                    r(i, j) = inner(i, other, j);
            
        }
        

        /**
         * @brief Computes the Gram matrix of this feature matrix
         * 
         * @return A dense symmetric matrix (use only lower part)
         */
        virtual boost::shared_ptr<const InnerMatrix> inner() const {
            // We lose space here, could be used otherwise???
            boost::shared_ptr<InnerMatrix> m(new InnerMatrix(size(), size()));
            
            for (Index i = size(); --i >= 0;)
                for (Index j = 0; j <= i; j++) {
                    Scalar x = kqp::inner<FVector>(get(i), get(j));
                    (*m)(i,j) = x;
                }
            return m;
        }
        
        /**
         * Our combiner
         */
        typedef linear_combination<FVector> Combiner;
        
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
        virtual FVector linearCombination(const Vector & lambdas) {
            if (lambdas.size() != this->size()) 
                BOOST_THROW_EXCEPTION(illegal_argument_exception());
            
            // A null vector
            FVector result;
            
            for(Index i = 0; i < lambdas.rows(); i++) 
                Combiner::axpy(result, lambdas[i], this->get(i));
            
            return result;
        }
        
        /** Get the number of feature vectors */
        virtual Index size() const = 0;
        
        /** Get the i<sup>th</sup> feature vector */
        virtual FVector get(Index i) const = 0;

        /** Get the i<sup>th</sup> feature vector */
        virtual void add(const FVector &f) = 0;

        /** Get the i<sup>th</sup> feature vector */
        virtual void set(Index i, const FVector &f) = 0;

        /** 
          * Remove the i<sup>th</sup> feature vector 
          * @param if swap is true, then the last vector will be swapped with one to remove (faster)
          */
        virtual void remove(Index i, bool swap = false) = 0;
    };
    
    

    
     
     

    
    
    /**
     * @brief Builds a compact representation of an hermitian operator. 
     *
     * Computes \f$ \mathfrak{U} = \sum_{i} \alpha_i \mathcal U A A^\top \mathcal U ^ \top   \f$
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     *
     * @param <Scalar>
     *            The Scalar type
     * @param <F>
     *            The type of the base vectors in the original space
     * @ingroup OperatorBuilder
     */
    template <class _FVector> class OperatorBuilder {        
    public:
        typedef _FVector FVector;        
        typedef typename FVector::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        typedef typename Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> RealVector;
        typedef boost::shared_ptr<RealVector> RealVectorPtr;
        typedef boost::shared_ptr<const RealVector> RealVectorCPtr;
        
        typedef FeatureMatrix<FVector> FMatrix;
        typedef boost::shared_ptr<FMatrix> FMatrixPtr;
        typedef boost::shared_ptr<const FMatrix> FMatrixCPtr;
        
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef boost::shared_ptr<Matrix> MatrixPtr;
        typedef boost::shared_ptr<const Matrix> MatrixCPtr;

        //! Virtual destructor to build the vtable
        virtual ~OperatorBuilder() {}

        //! Returns the feature matrix
        virtual FMatrixCPtr getX() const = 0;
        
        /** @brief Get the feature pre-image combination matrix.
         * @return a reference to the matrix (a 0x0 matrix is returned instead of identity)
         */
        virtual MatrixCPtr getY() const = 0;
        
        //! Returns the diagonal matrix as a vector
        virtual RealVectorPtr getD() const = 0;
        
        
        /** Get the diagonal matrix of eigenvalues (or a 0 x 0 matrix if it does not define one) */
        virtual Eigen::Matrix<Real, Eigen::Dynamic, 1> getEigenvalues() {
            return Eigen::Matrix<Real, Eigen::Dynamic, 1>();
        }
       
        /**
         * Add a new vector to the density
         *
         * Computes \f$A^\prime \approx A + \alpha   X A A^T  X^\top\f$
         * 
         * @param alpha
         *            The coefficient for the update
         * @param mX  The feature matrix X with n feature vectors
         * @param mA  The mixture matrix (of dimensions n x k)
         */
        virtual void add(const FMatrix &mX, const Matrix &mA) = 0;

        
        virtual void add(Real alpha, const FVector &mX) {
            
        }

    };
    
    
    /**
     * @brief Uses other operator builders and combine them.
     * @ingroup OperatorBuilder
     */
    template <class FMatrix> class DivideAndConquerBuilder : public OperatorBuilder<FMatrix> {
    public:
        typedef typename OperatorBuilder<FMatrix>::Scalar Scalar;
        typedef typename OperatorBuilder<FMatrix>::FVector FVector;

        virtual void add(double alpha, const FVector &v) {
            
        }
    };
    
    /**
     * @brief Accumulation based computation of the density.
     *
     * Supposes that we can compute a linear combination of the pre-images
     * 
     * @ingroup OperatorBuilder
     */
    template <class FMatrix>
    class AccumulatorBuilder : public OperatorBuilder<typename FMatrix::FVector> {   
    public:
        typedef OperatorBuilder<typename FMatrix::FVector> Ancestor;

        typedef typename FMatrix::Vector FVector;
        typedef typename OperatorBuilder<FVector>::Matrix Matrix;
        typedef typename OperatorBuilder<FVector>::MatrixCPtr MatrixCPtr;
        typedef boost::shared_ptr<const FMatrix> FMatrixCPtr;
        
        AccumulatorBuilder() {
        }
        
        
        virtual void add(const typename Ancestor::FMatrix &_fMatrix, const typename Ancestor::Matrix &coefficients) {
            // Just add            
            for(Index j = 0; j < coefficients.cols(); j++) 
                fMatrix.add(_fMatrix.linearCombination(coefficients.col(j)));
        }
        
        //! Actually performs the computation
        void compute() {
        }
        
    private:
        
        //! concatenation of pre-image matrices
        FMatrix fMatrix;
    };
    
    
    
    /**
     * @brief Removes pre-images with the null space method
     */
    template <class FVector> 
    void removeUnusedPreImages(FeatureMatrix<FVector> &mF, Eigen::Matrix<typename FVector::Scalar, Eigen::Dynamic, Eigen::Dynamic> &mY) {
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());
        
        // Removes unused pre-images
        for(Index i = 0; i < N; i++) 
            while (N > 0 && mY.row(i).norm() < EPSILON) {
                mF.remove(i, true);
                if (i != N - 1) 
                    mY.row(i) = mY.row(N-1);
                N = N - 1;
            }
    }
    
    
    /**
     * @brief Removes pre-images with the null space method
     * 
     * Removes pre-images using the null space method
     * 
     * @param mF the feature matrix
     * @param nullSpace the null space vectors of the gram matrix of the feature matrix
    */
    template <class FVector, typename Derived>
    void removeNullSpacePreImages(FeatureMatrix<FVector> &mF, const Eigen::MatrixBase<Derived> &nullSpace) {
        
    }
    
    /**
     * @brief Removes unuseful pre-images 
     *
     * 1. Removes unused pre-images 
     * 2. Computes a \f$LDL^\dagger\f$ decomposition of the Gram matrix to find redundant pre-images
     * 3. Removes newly unused pre-images
     */
    template <class FVector, typename Derived>
    void removeUnusefulPreImages(FeatureMatrix<FVector> &mF, const Eigen::MatrixBase<Derived> &mY) {
        typedef typename FVector::Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        
        // Removes unused pre-images
        removeUnusedPreImages(mF, mY);
        
        // Dimension of the problem
        Index N = mY.rows();
        assert(N == mF.size());

        // LDL decomposition (stores the L^T matrix)
        Eigen::LDLT<Eigen::MatrixBase<Derived>, Eigen::Upper> ldlt(mF.getGramMatrix());
        Eigen::MatrixBase<Derived> &mLDLT = ldlt.matrixLDLT();
        
        // Get the rank
        Index rank = 0;
        for(Index i = 0; i < N; i++)
            if (mLDLT.get(i,i) < EPSILON) rank++;
        
       
        Eigen::Block<Matrix> mL1 = mLDLT.block(0,0,rank,rank);
        Eigen::Block<Matrix> mL2 = mLDLT.block(0,rank+1,rank,N-rank);
        
        if (rank != N) {
            // Gets the null space vectors in mL2
            mL1.template triangularView<Derived, Eigen::Upper>().solveInPlace(mL2);
            mL2 *= ldlt.transpositionsP().adjoint();
            
            // TODO: Remove the vectors
            removeNullSpacePreImages(mF, mL2);
                
            // Removes unused pre-images
            removeUnusedPreImages(mF, mY);
        }
    }
    
    
    
    /**
     * The KKT pre-solver to solver the QP problem
     * @ingroup coneqp
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
