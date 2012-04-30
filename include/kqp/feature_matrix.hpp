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

#ifndef __KQP_FEATURE_MATRIX_H__
#define __KQP_FEATURE_MATRIX_H__


#include <kqp/kqp.hpp>
#include <boost/shared_ptr.hpp>
#include <kqp/alt_matrix.hpp>

namespace kqp {

#define KQP_SCALAR_TYPEDEFS(Scalar) \
    typedef typename Eigen::NumTraits<Scalar>::Real Real; \
    typedef Matrix<Scalar, Dynamic, Dynamic> ScalarMatrix; \
    typedef Matrix<Scalar, Dynamic, 1> ScalarVector; \
    typedef Matrix<Real, Dynamic, Dynamic> RealMatrix; \
    typedef Matrix<Real, Dynamic, 1> RealVector; \
    typedef typename AltDense<Scalar>::type  ScalarAltMatrix; \
    typedef typename AltVector<Real>::type   RealAltVector; \
    typedef FeatureMatrix<Scalar> FMatrix; \
    typedef FeatureSpace<Scalar> FSpace; \
    typedef typename AltVector<Real>::ConstantVectorType ConstantRealVector; \
    typedef typename AltDense<Scalar>::IdentityType IdentityScalarMatrix; \
    typedef FeatureMatrixBase<Scalar> FMatrixBase; \
    typedef boost::shared_ptr< FeatureMatrixBase<Scalar> > FMatrixBasePtr; \
    typedef FeatureSpaceBase<Scalar> FSpaceBase; \
    typedef boost::shared_ptr< FeatureSpaceBase<Scalar> > FSpaceBasePtr;

    
    template<typename Scalar> class FeatureMatrixBase;
    template<typename Scalar> class FeatureMatrix;
    template<typename Scalar> class FeatureSpaceBase;
    template<typename Scalar> class FeatureSpace;
    
    /**
     * @brief Base for all feature matrix classes
     * 
     * @ingroup FeatureMatrix
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class Scalar> 
    class FeatureMatrixBase {
    public:       
        KQP_SCALAR_TYPEDEFS(Scalar);

        /** Number of pre-image vectors */
        virtual Index size() const = 0;
        
        /** Add pre-images vectors */
        virtual void add(const FMatrixBase &f, const std::vector<bool> *which = NULL) = 0;

        /** @brief Reduces the feature matrix to a subset of its vectors.
         * 
         * The list of indices is supposed to be ordered.
         *
         * @param begin Beginning of the list of indices
         * @param end End of the list of indices
         */
        virtual FMatrixBasePtr subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) const = 0;
        
        /** Assignement */
        virtual FMatrixBase &operator=(const FeatureMatrixBase<Scalar> &other) = 0;
        
        /** Copy */
        virtual FMatrixBasePtr copy() const = 0;
        
        /** Dynamic cast */
        template<typename T> inline const T &as() const { return dynamic_cast<const T&>(*this); }
        template<typename T> inline T &as() { return dynamic_cast<T&>(*this); }        
    };
    
    /**
     * @brief Base for all feature matrix classes
     * 
     * @ingroup FeatureMatrix
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <class Scalar> 
    class FeatureMatrix {
    public:       
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        FeatureMatrix() {}
        
#ifndef SWIG
        FeatureMatrix(FeatureMatrix &&other) : m_fMatrix(other.m_fMatrix) {}
#endif
        FeatureMatrix(const FeatureMatrix &other) : m_fMatrix(other.m_fMatrix->copy()) {}
        
        /** Creates a feature matrix and takes the ownership of the pointer */
        explicit FeatureMatrix(FeatureMatrixBase<Scalar> *fMatrix) : m_fMatrix(fMatrix) {}
        
        /** Creates a feature matrix, sharing the pointer */
        explicit FeatureMatrix(const boost::shared_ptr< FeatureMatrixBase<Scalar> > &fMatrix) : m_fMatrix(fMatrix) {}
        
        /** Number of pre-image vectors */
        inline Index size() const {
            return m_fMatrix->size();
        }
        
        /** Add pre-images vectors */
        inline void add(const FMatrix &f, const std::vector<bool> *which = NULL) {
            return m_fMatrix->add(*f, which);
        }
        
        /** @brief Reduces the feature matrix to a subset of its vectors.
         * 
         * The list of indices is supposed to be ordered.
         *
         * @param begin Beginning of the list of indices
         * @param end End of the list of indices
         */
        inline FMatrix subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) const {
            return FeatureMatrix(m_fMatrix->subset(begin, end));
        }
        
        /** @brief Reduces the feature matrix to a subset of its vectors.
         * 
         * @param list The list of boolean values (true = keep)
         */
        inline FMatrix subset(const std::vector<bool> &list) const {
            return FeatureMatrix(m_fMatrix->subset(list.begin(), list.end()));
        }
        
        
        FMatrix &operator=(const FMatrix &other) {
            if (typeid(m_fMatrix.get()) != typeid(other.m_fMatrix.get()))
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Non matching types for assignement (%s to %s)", %KQP_DEMANGLE(other) %KQP_DEMANGLE(*this));
            m_fMatrix = other.m_fMatrix->copy();
            return *this;
        }
        
        const FMatrixBase * operator->() const { return m_fMatrix.get();  }        
        FMatrixBase * operator->() { return m_fMatrix.get(); }
        
        const FMatrixBase & operator*() const { return *m_fMatrix.get(); }
        FMatrixBase & operator*() { return *m_fMatrix.get(); }

#ifndef SWIG
        FeatureMatrix<Scalar> &operator=(FeatureMatrix<Scalar> &&other) {
            if (typeid(other) != typeid(*this))
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Non matching types for assignement (%s to %s)", %KQP_DEMANGLE(other) %KQP_DEMANGLE(*this));
            m_fMatrix.swap(other.m_fMatrix);
            return *this;
        }
#endif
        
    private:
        boost::shared_ptr< FeatureMatrixBase<Scalar> > m_fMatrix;
        
        friend class FeatureSpace<Scalar>;
    };
    

    
    
    
    
    /**
     * @brief A feature space
     *
     */
    template<typename Scalar>
    class FeatureSpaceBase {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
               
        virtual ~FeatureSpaceBase() {}
        
        //! Dimension of the underlying space (-1 for infinity)
        virtual Index dimension() const = 0;
        
        //! Returns whether the pre-images can be linearly combined
        virtual bool canLinearlyCombine() const {
            return false;
        }
        
        //! Copy
        virtual boost::shared_ptr< FeatureSpaceBase<Scalar> > copy() const = 0;
        
        //! Gram matrix
        virtual const ScalarMatrix &k(const FMatrixBase &mX) const = 0;

        //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
        virtual ScalarMatrix k(const FMatrixBase &mX, const ScalarAltMatrix &mY, const RealAltVector &mD) const {
            return mD.asDiagonal() * mY.transpose() * k(mX) * mY * mD.asDiagonal();
        }
        
        //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
        virtual ScalarMatrix k(const FMatrixBase &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1, 
                               const FMatrixBase &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const = 0;

        
        //! Creates a new feature matrix
        virtual FMatrixBasePtr newMatrix() const = 0;        

        //! Creates a new feature matrix (copy)
        virtual FMatrixBasePtr newMatrix(const FeatureMatrixBase<Scalar> &) const = 0;        
        
        //! Linear combination of pre-images \f$ \alpha X A + \beta Y B \f$
        virtual FMatrixBasePtr linearCombination(const FMatrixBase &, const ScalarAltMatrix &, Scalar , const FMatrixBase *, const ScalarAltMatrix *, Scalar) const {
            KQP_THROW_EXCEPTION_F(illegal_operation_exception, "Cannot compute the linear combination in feature space [%s]", %KQP_DEMANGLE(*this));   
        }
        
        /** Dynamic casts */
        template<typename T> inline const T &as() const { return dynamic_cast<T&>(*this); }
        template<typename T> inline T &as() { return dynamic_cast<T&>(*this); }        

    };
    
    
    /**
     * @brief A feature space (container).
     *
     */
    template<typename Scalar>
    class FeatureSpace {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        // Undefined space
        FeatureSpace() {}

        FeatureSpace(FSpaceBase *ptr) : m_fSpace(ptr) {}
        FeatureSpace(const FSpaceBasePtr &ptr) : m_fSpace(ptr) {}
        FeatureSpace(const FeatureSpace &other) : m_fSpace(other.m_fSpace->copy()) {}
        FeatureSpace &operator=(const FeatureSpace &other) {
            m_fSpace = other.m_fSpace->copy();
            return *this;
        }
        
#ifndef SWIG
        FeatureSpace(FeatureSpace &&other) :  m_fSpace(std::move(other.m_fSpace)) {}
        FeatureSpace &operator=(FeatureSpace &&other) {
            m_fSpace = std::move(other.m_fSpace);
            return *this;
        }
#endif        
        
        //! Constant copy
        const FeatureSpace constCopy() const { return FeatureSpace(m_fSpace); }
        
        //! Dimension of the underlying space (-1 for infinity)
        inline Index dimension() const { return m_fSpace->dimension(); }
        
        //! Gram matrix
        inline const ScalarMatrix &k(const FMatrix &mX) const {
            return m_fSpace->k(*mX);
        }
        
        //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
        inline ScalarMatrix k(const FMatrix &mX, const ScalarAltMatrix &mY, const RealAltVector &mD) const {
            return m_fSpace->k(*mX,mY,mD);
        }
        
        //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
        inline ScalarMatrix k(const FMatrix &mX, const ScalarAltMatrix &mY) const {
            return m_fSpace->k(*mX, mY, RealVector::Ones(mY.cols()));
        }
        
        //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
        inline ScalarMatrix k(const FMatrix &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1, 
                              const FMatrix &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const {
            return m_fSpace->k(*mX1,mY1,mD1, *mX2,mY2,mD2);
        }
        
        
        //! Inner product \f$X_1^\dagger X_2\f$
        inline ScalarMatrix k(const FMatrix &mX1, const ScalarAltMatrix &mY1, const FMatrix &mX2, const ScalarAltMatrix &mY2) const {
            return k(mX1, mY1,  RealVector::Ones(mY1.cols()), mX2, mY2, RealVector::Ones(mY2.cols()));
        }
        
        //! Inner product \f$X_1^\dagger X_2\f$
        inline ScalarMatrix k(const FMatrix &mX1, const FMatrix &mX2) const {
            return k(mX1, IdentityScalarMatrix(mX1.size(),mX1.size()), RealVector::Ones(mX1.size()), 
                     mX2, IdentityScalarMatrix(mX2.size(),mX2.size()), RealVector::Ones(mX2.size()));
        }
        
        //! Creates a new feature matrix
        inline FMatrix newMatrix() const {
            return FMatrix(m_fSpace->newMatrix());
        }
        
        //! Creates a new feature matrix (copy)
        inline FMatrix newMatrix(const FMatrix &base) const {
            return FMatrix(m_fSpace->newMatrix(*base.m_fMatrix));
        }
        
        //! Returns whether the pre-images can be linearly combined
        inline bool canLinearlyCombine() const {
            return m_fSpace->canLinearlyCombine();
        };
        
        
        /** 
         * @brief Linear combination of the feature vectors.
         * 
         * Computes \f$ \alpha XA + \beta Y B  \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
         */
        inline FMatrix linearCombination(const FMatrix &mX, const ScalarAltMatrix &mA, Scalar alpha, const FMatrix &mY, const ScalarAltMatrix &mB, Scalar beta) const {
            // Check for correctedness
            if (mA.rows() != mX.size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot linearly combine with a matrix with %d rows (we have %d pre-images)", %mA.rows() %mX.size());
            
            // If we have no columns in A, then return an empty feature matrix
            if (mA.cols() == 0)
                return newMatrix();
            
            // Call the derived
            return FMatrix(m_fSpace->linearCombination(*mX, mA, alpha, &*mY, &mB, beta));
        }
        
        
        /** 
         * @brief Linear combination of the feature vectors.
         * 
         * Computes \f$ XA \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
         */
        inline FMatrix linearCombination(const FMatrix &mX, const ScalarAltMatrix &mA, Scalar alpha = (Scalar)1) const {
            // Check for correctedness
            if (mA.rows() != mX.size())
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot linearly combine with a matrix with %d rows (we have %d pre-images)", %mA.rows() %mX.size());
            
            // If we have no columns in A, then return an empty feature matrix
            if (mA.cols() == 0)
                return newMatrix();
            
            // Call the derived
            return FMatrix(m_fSpace->linearCombination(*mX, mA, alpha, NULL, NULL, 0));
        }
        
        
        const FSpaceBase * operator->() const { return m_fSpace.get();  }        
        FSpaceBase * operator->() { return m_fSpace.get(); }
        
        const FSpaceBase & operator*() const { return *m_fSpace.get(); }
        FSpaceBase & operator*() { return *m_fSpace.get(); }

        
    private:
        boost::shared_ptr< FeatureSpaceBase<Scalar> > m_fSpace;
        
    };


# ifndef SWIG
# define KQP_SCALAR_GEN(Scalar) extern template class FeatureMatrix<Scalar>; extern template class FeatureSpace<Scalar>;
# include <kqp/for_all_scalar_gen>
# endif 

}

#endif
