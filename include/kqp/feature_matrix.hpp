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

#include <pugixml.hpp>
#include <kqp/kqp.hpp>
#include <boost/shared_ptr.hpp>
#include <kqp/alt_matrix.hpp>


namespace kqp
{
#   include <kqp/define_header_logger.hpp>
    DEFINE_KQP_HLOGGER("kqp.feature_matrix");

#ifndef SWIG
template <typename Scalar>
struct ScalarDefinitions
{
    typedef typename Eigen::NumTraits<Scalar>::Real Real;
    typedef kqp::AltMatrix< typename kqp::AltDense<Scalar>::DenseType, typename kqp::AltDense<Scalar>::IdentityType >  ScalarAltMatrix;
    typedef kqp::AltMatrix< typename kqp::AltVector<Real>::VectorType,  typename kqp::AltVector<Real>::ConstantVectorType >   RealAltVector;
};
#endif

#define KQP_SCALAR_TYPEDEFS(Scalar) \
    typedef typename Eigen::NumTraits<Scalar>::Real Real; \
    /* Scalar Matrices */ \
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix; \
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ScalarVector; \
    typedef Eigen::Matrix<Real, Eigen::Dynamic, Dynamic> RealMatrix; \
    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RealVector; \
    typedef typename ScalarDefinitions<Scalar>::ScalarAltMatrix ScalarAltMatrix; \
    typedef typename ScalarDefinitions<Scalar>::RealAltVector RealAltVector; \
    typedef typename AltVector<Real>::ConstantVectorType ConstantRealVector; \
    typedef typename AltDense<Scalar>::IdentityType IdentityScalarMatrix; \
    /* Feature matrices */ \
    typedef FeatureMatrixBase<Scalar> FMatrixBase; \
    typedef boost::shared_ptr< FeatureMatrixBase<Scalar> > FMatrix; /* FIXME: remove  */ \
    typedef boost::shared_ptr< FeatureMatrixBase<Scalar> > FMatrixPtr; \
    typedef boost::shared_ptr< FeatureMatrixBase<Scalar> > FMatrixCPtr; \
    /* Feature spaces */ \
    typedef SpaceBase<Scalar> FSpaceBase; \
    typedef boost::shared_ptr< SpaceBase<Scalar> > FSpace; /* FIXME: remove */ \
    typedef boost::shared_ptr< SpaceBase<Scalar> > FSpacePtr; \
    typedef boost::shared_ptr< SpaceBase<Scalar> > FSpaceCPtr; \
    typedef boost::shared_ptr< FeatureMatrixBase<Scalar> > FMatrixBasePtr; \
    typedef boost::shared_ptr< FeatureMatrixBase<Scalar> > FMatrixBaseCPtr; 

//! Defines types for a matrix (Self should be defined before calling the macro)
#define KQP_MATRIX_TYPEDEFS(Scalar) \
        KQP_SCALAR_TYPEDEFS(Scalar) \
        typedef boost::shared_ptr<Self> SelfPtr; \
        typedef boost::shared_ptr<const Self> SelfCPtr; 
        
//! Defines types for a Space (Self should be defined before calling the macro)
#define KQP_SPACE_TYPEDEFS(SpaceName, Scalar) \
        KQP_SCALAR_TYPEDEFS(Scalar) \
        typedef boost::shared_ptr<Self> SelfPtr; \
        typedef boost::shared_ptr<const Self> SelfCPtr; \
        static const std::string &name() { static std::string NAME(SpaceName); return NAME; }


//! Information about templates
template<typename Scalar> struct ScalarInfo;

template<> struct ScalarInfo<double>
{
    static std::string name()
    {
        return "double";
    }
};
template<> struct ScalarInfo<float>
{
    static std::string name()
    {
        return "float";
    }
};
template<> struct ScalarInfo<std::complex<double> >
{
    static std::string name()
    {
        return "complex/double";
    }
};
template<> struct ScalarInfo<std::complex<float> >
{
    static std::string name()
    {
        return "complex/float";
    }
};


template<typename Scalar> class FeatureMatrixBase;
template<typename Scalar> class SpaceBase;


/**
  * Store for kernel values when computing kernels 
  */
template<typename Scalar> class KernelValues {
public:
    KernelValues(Scalar innerX, Scalar innerY, Scalar inner) :
        _innerX(innerX), _innerY(innerY), _inner(inner) {}

    KernelValues() {}    

    inline Scalar inner(int mode = 0) const { if (mode == -1) return _innerX; return mode == 1 ? _innerY : _inner; }
    inline Scalar innerX(int mode = 0) const { if (mode == 1) return _innerY; return _innerX; }
    inline Scalar innerY(int mode = 0) const { if (mode == -1) return _innerX; return _innerY; }

    Scalar _innerX, _innerY, _inner;
};


/**
 * @brief Base for all feature matrix classes
 *
 * @ingroup FeatureMatrix
 * @author B. Piwowarski <benjamin@bpiwowar.net>
 */
template <class Scalar>
class FeatureMatrixBase 
{
public:
    KQP_SCALAR_TYPEDEFS(Scalar);

    virtual ~FeatureMatrixBase() {}

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
    virtual FMatrixPtr subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) const = 0;
    
    /** @brief Reduces the feature matrix to a subset of its vectors.
     *
     * @param list The list of boolean values (true = keep)
     */
    inline FMatrixPtr subset(const std::vector<bool> &list) const
    {
        return subset(list.begin(), list.end());
    }

    /** Assignement */
    virtual FMatrixBase &operator=(const FeatureMatrixBase<Scalar> &other) = 0;

    /** Copy */
    virtual FMatrixBasePtr copy() const = 0;



#ifndef SWIG
    /** Dynamic cast */
    template<typename T> inline const T &as() const
    {
        return kqp::our_dynamic_cast<const T &>(*this);
    }
    template<typename T> inline T &as()
    {
        return kqp::our_dynamic_cast<T &>(*this);
    }

    /** Add pre-images vectors */
    inline void add(const FMatrix &f, const std::vector<bool> *which = NULL)
    {
        return add(*f, which);
    }
#endif




};


/**
 * Common ancestor class for all spaces
 */
class AbstractSpace 
{
public:
    virtual ~AbstractSpace() {}
    /**
     * Load from XML
     * \param node The node to load
     */
    virtual void load(const pugi::xml_node &node) = 0;

    //! Save in XML
    virtual void save(pugi::xml_node &node) const = 0;

};


/**
 * @brief A feature space
 *
 */
template<typename Scalar>
class SpaceBase : public AbstractSpace
{
public:
    KQP_SCALAR_TYPEDEFS(Scalar);

    
    static int &counter() { static int counter = 0; return counter; }

    virtual ~SpaceBase() {
        KQP_HLOG_DEBUG_F("Destroying %s (%d / counter = %d)", %KQP_DEMANGLE(*this) %this %--counter());
    }

    SpaceBase() {
        KQP_HLOG_DEBUG_F("Creating %s (%d / counter = %d)", %KQP_DEMANGLE(*this) %this %++counter());
    }

    //! Dimension of the underlying space (-1 for infinity)
    virtual Index dimension() const = 0;



    //! Copy
    virtual boost::shared_ptr< SpaceBase<Scalar> > copy() const = 0;

    //! Gram matrix
    virtual const ScalarMatrix &k(const FMatrixBase &mX) const = 0;

    //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
    virtual ScalarMatrix k(const FMatrixBase &mX, const ScalarAltMatrix &mY, const RealAltVector &mD) const
    {
        return mD.asDiagonal() * mY.adjoint() * k(mX) * mY * mD.asDiagonal();
    }

    //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
    virtual ScalarMatrix k(const FMatrixBase &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1,
                           const FMatrixBase &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const = 0;

    //! Inner products \f$Y_1^\dagger X_1^\dagger X_2 Y_2\f$
    inline ScalarMatrix k(const FMatrixBase &mX1, const ScalarAltMatrix &mY1,
                          const FMatrixBase &mX2, const ScalarAltMatrix &mY2) const
    {
        return k(mX1, mY1, RealVector::Ones(mY1.cols()), mX2, mY2, RealVector::Ones(mY2.cols()));
    }

    //! Inner products \f$X_1^\dagger X_2\f$
    inline ScalarMatrix k(const FMatrixBase &mX1, const FMatrixBase &mX2) const
    {
        return k(mX1, Eigen::Identity<Scalar>(mX1.size(), mX1.size()), RealVector::Ones(mX1.size()),
                 mX2, Eigen::Identity<Scalar>(mX2.size(), mX2.size()), RealVector::Ones(mX2.size()));
    }

    /**
     * \brief Update the partials.
     *
     * Update the partials with \f$ \alpha \times \frac{\partial k(X,Y)}{\partial \theta}\f$ where
     * \f$theta\f$ are the parameters
     * \param alpha The weight \f$\alpha\f$
     * \param partials The partials to update
     * \param offset The offset within the real values stored in \a partials
     * \param mX A vector in the feature space
     * \param mY A vector in the feature space
    */
    virtual void updatePartials(Real /*alpha*/, std::vector<Real> &/*partials*/, int /*offset*/, 
        const std::vector< KernelValues<Scalar> > &/* kernelValues */, int /* kOffset */, int /* mode */) const {}

    virtual void updatePartials(Real alpha, std::vector<Real> &partials, 
        const std::vector< KernelValues<Scalar> > &kernelValues, int mode) const {
        updatePartials(alpha, partials, 0, kernelValues, 0, mode);
    }

    virtual void update(std::vector< KernelValues<Scalar> > &, int /* kOffset */ = 0) const {}

    /**
     * Returns the number of parameters for this feature space
    */
    virtual int numberOfParameters() const
    {
        return 0;
    }

    /**
     * Returns the number of stored kernel values for this kernel
     */
    virtual int numberOfKernelValues() const
    {
        return 1;
    }

    virtual void getParameters(std::vector<Real> &, int) const
    {
    }

    virtual void setParameters(const std::vector<Real> &, int)
    {
    }

    //! Creates a new feature matrix
    virtual FMatrixBasePtr newMatrix() const = 0;

    //! Linear combination of pre-images \f$ \alpha X A + \beta Y B \f$
    virtual FMatrixBasePtr linearCombination(const FMatrixBase &, const ScalarAltMatrix &, Scalar , const FMatrixBase *, const ScalarAltMatrix *, Scalar) const
    {
        KQP_THROW_EXCEPTION_F(illegal_operation_exception, "Cannot compute the linear combination in feature space [%s]", % KQP_DEMANGLE(*this));
    }

#ifndef SWIG
    /** Dynamic casts */
    template<typename T> inline const T &as() const
    {
        return kqp::our_dynamic_cast<T &>(*this);
    }
    template<typename T> inline T &as()
    {
        return kqp::our_dynamic_cast<T &>(*this);
    }
#endif

    template<typename T> inline bool castable_as() const
    {
        return kqp::our_dynamic_cast<T *>(this) != 0;
    }

#ifndef SWIG
    //! Gram matrix
    inline const ScalarMatrix &k(const FMatrix &mX) const
    {
        return k(*mX);
    }

    //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
    inline ScalarMatrix k(const FMatrix &mX, const ScalarAltMatrix &mY, const RealAltVector &mD) const
    {
        return k(*mX, mY, mD);
    }

    //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
    inline ScalarMatrix k(const FMatrix &mX, const ScalarAltMatrix &mY) const
    {
        return k(*mX, mY, RealVector::Ones(mY.cols()));
    }

    //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
    inline ScalarMatrix k(const FMatrixBase &mX, const ScalarAltMatrix &mY) const
    {
        return k(mX, mY, RealVector::Ones(mY.cols()));
    }

    //! Inner products \f$D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2\f$
    inline ScalarMatrix k(const FMatrix &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1,
                          const FMatrix &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const
    {
        return k(*mX1, mY1, mD1, *mX2, mY2, mD2);
    }


    //! Inner product \f$X_1^\dagger X_2\f$
    inline ScalarMatrix k(const FMatrix &mX1, const ScalarAltMatrix &mY1, const FMatrix &mX2, const ScalarAltMatrix &mY2) const
    {
        return k(mX1, mY1,  RealVector::Ones(mY1.cols()), mX2, mY2, RealVector::Ones(mY2.cols()));
    }

    //! Inner product \f$X_1^\dagger X_2\f$
    inline ScalarMatrix k(const FMatrix &mX1, const FMatrix &mX2) const
    {
        return k(mX1, IdentityScalarMatrix(mX1->size(), mX1->size()), RealVector::Ones(mX1->size()),
                 mX2, IdentityScalarMatrix(mX2->size(), mX2->size()), RealVector::Ones(mX2->size()));
    }
#endif


    //! Returns whether the pre-images can be linearly combined
    inline bool canLinearlyCombine() const
    {
        return m_useLinearCombination && _canLinearlyCombine();
    };


    /**
     * @brief Linear combination of the feature vectors.
     *
     * Computes \f$ \alpha XA + \beta Y B  \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
     */
    inline FMatrix linearCombination(const FMatrixBase &mX, const ScalarAltMatrix &mA, Scalar alpha, const FMatrixBase &mY, const ScalarAltMatrix &mB, Scalar beta) const
    {
        // Check for correctedness
        if (mA.rows() != mX.size())
            KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot linearly combine with a matrix with %d rows (we have %d pre-images)", % mA.rows() % mX.size());

        // If we have no columns in A, then return an empty feature matrix
        if (mA.cols() == 0)
            return newMatrix();

        // Call the derived
        return FMatrix(linearCombination(mX, mA, alpha, &mY, &mB, beta));
    }


    /**
     * @brief Linear combination of the feature vectors.
     *
     * Computes \f$ XA \f$ where \f$X\f$ is the current feature matrix, and \f$A\f$ is the argument
     */
    inline FMatrix linearCombination(const FMatrixBase &mX, const ScalarAltMatrix &mA, Scalar alpha = (Scalar)1) const {
        // Check for correctedness
        if (mA.rows() != mX.size())
            KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot linearly combine with a matrix with %d rows (we have %d pre-images)", % mA.rows() % mX.size());

        // If we have no columns in A, then return an empty feature matrix
        if (mA.cols() == 0)
            return newMatrix();

        // Call the derived
        return FMatrix(linearCombination(mX, mA, alpha, NULL, NULL, 0));
    }

#ifndef SWIG
    //! \deprecated
    inline FMatrix linearCombination(const FMatrix &mX, const ScalarAltMatrix &mA, Scalar alpha = (Scalar)1) const {
        return linearCombination(*mX, mA, alpha);
    }
#endif

    //! Sets the flag for linear combination use (debug)
    void setUseLinearCombination(bool flag)
    {
        m_useLinearCombination = flag;
    }

    void getParameters(std::vector<Real> &parameters, size_t offset) const
    {
        getParameters(parameters, offset);
    }

    void setParameters(const std::vector<Real> &parameters, size_t offset)
    {
        setParameters(parameters, offset);
    }

    std::string demangle() const {
        return KQP_DEMANGLE(*this);
    }
    
protected:
    //! Returns whether the pre-images can be linearly combined (false by default)
    virtual bool _canLinearlyCombine() const
    {
        return false;
    }

private:

    //! Force the linear combination flag
    bool m_useLinearCombination;
};




# ifndef SWIG
# define KQP_SCALAR_GEN(Scalar) extern template class FeatureMatrixBase<Scalar>; extern template class SpaceBase<Scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
# endif

}

#endif
