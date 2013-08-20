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

#ifndef _KQP_DECOMPOSITION_H
#define _KQP_DECOMPOSITION_H

#include <utility>
#include <kqp/feature_matrix.hpp>
#include <kqp/evd_utils.hpp>


namespace kqp
{

/** A decomposition: holds three matrices X, Y and D
    
    The operator is \f$ XYDY^TX^T\f$ or \f$ XYD^2Y^TX^T \f$ (square root mode)
    
*/
template<typename Scalar>
class Decomposition
{
    
public:
    KQP_SCALAR_TYPEDEFS(Scalar);

    //! Feature space
    FSpaceCPtr fs;

    //! The feature matrix
    FMatrixPtr  mX;

    //! The linear combination matrix
    ScalarAltMatrix mY;

    //! The diagonal matrix
    RealAltVector mD;

    //! If this is a real decomposition
    bool orthonormal;
    
    //! Square root representation ?
    bool m_squareRoot;

    //! Number of rank updates
    Index updateCount;

    //! Default constructor with an undefined feature space
    Decomposition() : orthonormal(true), m_squareRoot(false) {}

    //! Default constructor with a feature space
    Decomposition(const FSpaceCPtr &fs) : fs(fs), mX(fs->newMatrix()), orthonormal(true), m_squareRoot(false) {}

    //! Full constructor
    Decomposition(const FSpaceCPtr &fs, const FMatrixCPtr &mX, const ScalarAltMatrix &mY, const RealAltVector &mD, bool orthonormal, bool squareRoot)
        : fs(fs), mX(mX->copy()), mY(mY), mD(mD), orthonormal(orthonormal), m_squareRoot(squareRoot), updateCount(0) {}

#ifndef SWIG
    //! Move constructor
    Decomposition(FSpaceCPtr && fs, FMatrixPtr && mX, const ScalarAltMatrix && mY, const RealAltVector && mD, bool orthonormal, bool squareRoot)
        : fs(fs), mX(mX), mY(mY), mD(mD), orthonormal(orthonormal), m_squareRoot(squareRoot), updateCount(0)
    {
    }

    //! Move constructor
    Decomposition(Decomposition && other)
    {
        *this = std::move(other);
    }

    //! Move assignement
    Decomposition &operator=(Decomposition && other)
    {
        take(other);
        return *this;
    }
#endif
    void take(Decomposition &other)
    {
        fs = other.fs;
        mX = other.mX;
        mY.swap(other.mY);
        mD.swap(other.mD);
        std::swap(orthonormal, other.orthonormal);
        std::swap(updateCount, other.updateCount);
        std::swap(m_squareRoot, other.m_squareRoot);
    }


    //! Copy constructor
    Decomposition(const Decomposition &other)
    {
        *this = other;
    }

    //! Copy assignement
    Decomposition &operator=(const Decomposition &other)
    {
        fs = other.fs;
        mX = other.mX->copy();
        mY = other.mY;
        mD = other.mD;
        orthonormal = other.orthonormal;
        updateCount = other.updateCount;
        m_squareRoot = other.m_squareRoot;
        return *this;
    }


    /**
     * Computes \f$ D_1^\dagger Y_1^\dagger X_1^\dagger X_2 Y_2 D_2 \f$
     */
    ScalarMatrix k(const Decomposition &other) const
    {
        return fs->k(mX, mY, mD, other.mX, other.mY, other.mD);
    }

    /** Check that the decomposition is valid */
    bool check() const {
        return mX->size() == mY.rows() && mY.cols() == mD.rows();
    }

    Index rank() const {
        return mD.size();
    }

    Index preimagesCount() const {
        return mX->size();
    }
    
    /** Serialize the matrices */
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/) {
        ar & mX;
        ar & mY;
        ar & mD;
        ar & orthonormal;
        ar & updateCount;
    }

    inline bool isOrthonormal() const {
        return orthonormal;
    }
    
    inline bool isSquareRoot() const {
        return m_squareRoot;
    }

    //! Change the type of representation
    inline void squareRoot(bool mode) {
        if (mode == m_squareRoot)
            return;
        
        if (mode)
            mD.unaryExprInPlace(Eigen::internal::scalar_sqrt_op<Real>());
        else
            mD.unaryExprInPlace(Eigen::internal::scalar_abs2_op<Real>());

        m_squareRoot = mode;
    }
    
    //! Orthonormalize the decomposition (non const version)
    void orthonormalize() {
        // Check if we have something to do
        if (isOrthonormal()) return;

        // Orthonornalize (keeping in mind that our diagonal might be the square root)
        if (m_squareRoot) 
            mD.unaryExprInPlace(Eigen::internal::scalar_abs2_op<Real>());
        Orthonormalize<Scalar>::run(fs, mX, mY, mD);
        if (m_squareRoot)
            mD.unaryExprInPlace(Eigen::internal::scalar_sqrt_op<Real>());

        // If we can linearly combine, use it to reduce the future amount of computation
        if (fs->canLinearlyCombine()) {
            mX = fs->linearCombination(mX, mY, 1);
            mY = Eigen::Identity<Scalar>(mX->size(),mX->size());
        } 
        
        orthonormal = true;
        
    }
    
    //! Multiply the operator by a real value
    void multiplyBy(Real alpha) {
        if (m_squareRoot) {
            if (alpha < 0) 
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot multiply a kernel operator by a negative value (%g)", %alpha);
            alpha = std::sqrt(alpha);
        }
        
        mD.unaryExprInPlace(Eigen::internal::scalar_multiple_op<Real>(alpha));
    }
    
    
    //! Computes the trace of the operator
    Real trace() const {
        if (isOrthonormal()) 
            return m_squareRoot ? mD.cwiseAbs2().sum() : mD.sum();
        
        if (m_squareRoot)
            return fs->k(*mX, mY, mD).trace();

        return fs->k(*mX, mY, mD, *mX, mY, RealVector::Ones(mX->size())).trace();
    }

    //! Normalize by dividing by the trace
    void traceNormalize() {
        this->multiplyBy((Scalar)1 / this->trace());
    }
    
};


}

#endif

