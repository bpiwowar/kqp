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

#ifndef __KQP_KERNEL_EVD_UTILS_H__
#define __KQP_KERNEL_EVD_UTILS_H__

#include <cassert>

#include <boost/scoped_ptr.hpp>
#include <boost/type_traits/is_complex.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Core>

#include <kqp/kernel_evd.hpp>
#include <kqp/subset.hpp>

namespace kqp {
    template <class Derived> 
    void thinEVD(const Eigen::SelfAdjointEigenSolver<Derived> &evd, 
                 Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors, 
                 Eigen::Matrix<typename Eigen::NumTraits<typename Derived::Scalar>::Real, Eigen::Dynamic, 1> &eigenvalues,
                 Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> *nullEigenvectors = nullptr
                 ) {
        typedef typename Derived::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        // We expect eigenvalues to be sorted by increasing order
        Index dimension = evd.eigenvectors().rows();
        
        const Eigen::Matrix<Real, Eigen::Dynamic, 1> &d = evd.eigenvalues();
        Real threshold = EPSILON * (Real)d.size();
        
        Index n = d.rows();
        Index negatives = 0, zeros = 0;
        
        for(Index i = 0; i < n; i++) {
            assert(i==0 || d[i-1] <= d[i]);
            if (-d[i] > threshold) negatives++;
            else if (d[i] < threshold) zeros++; 
            else break;
        }
        
        Index positives = n - negatives - zeros;
        
        eigenvalues.resize(positives+negatives);
        eigenvalues.head(negatives) = d.head(negatives);
        eigenvalues.tail(positives) = d.tail(positives);
        
        eigenvectors.resize(dimension, positives + negatives);
        eigenvectors.leftCols(negatives) = evd.eigenvectors().leftCols(negatives);
        eigenvectors.rightCols(positives) = evd.eigenvectors().rightCols(positives);
        
        if (nullEigenvectors) {
            *nullEigenvectors = evd.eigenvectors().block(0, negatives, dimension, zeros);
        }
    }
    
    
    //! Thin EVD with Alt matrices
    template <class Derived> 
    void thinEVD(const Eigen::SelfAdjointEigenSolver<Derived> &evd, 
                 typename AltDense<typename Derived::Scalar>::type &eigenvectors, 
                 typename AltVector<typename Eigen::NumTraits<typename Derived::Scalar>::Real>::type &eigenvalues,
                 typename AltDense<typename Derived::Scalar>::type *nullEigenvectors = nullptr) {
        
        typedef Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
        ScalarMatrix _eigenvectors;
        Eigen::Matrix<typename Eigen::NumTraits<typename Derived::Scalar>::Real, Eigen::Dynamic, 1> _eigenvalues;
        boost::scoped_ptr<ScalarMatrix> _nullEigenvectors;
        
        
        if (nullEigenvectors != nullptr) 
            _nullEigenvectors.reset(new ScalarMatrix());
        
        thinEVD(evd, _eigenvectors, _eigenvalues, _nullEigenvectors.get());

        eigenvectors.swap(_eigenvectors);
        eigenvalues.swap(_eigenvalues);
        if (nullEigenvectors != nullptr) 
            nullEigenvectors->swap(*_nullEigenvectors);
        
    }
    
    
    //! Re-orthonormalize a decomposition (complex case)
    template<typename FMatrix>
    typename boost::enable_if<boost::is_complex<typename ftraits<FMatrix>::Scalar>>::type
    orthonormalize(const FMatrix &mX,
                        Eigen::Matrix<typename ftraits<FMatrix>::Scalar, Eigen::Dynamic, Eigen::Dynamic> &mY,
                        Eigen::Matrix<typename ftraits<FMatrix>::Real, Eigen::Dynamic, 1> &mD) {
        typename ftraits<FMatrix>::ScalarMatrix m = mD.cwiseSqrt().asDiagonal() * mY.transpose() * mX.inner() * mY * mD.cwiseSqrt().asDiagonal();
        Eigen::SelfAdjointEigenSolver<decltype(m)> evd(m.template selfadjointView<Eigen::Lower>());
        kqp::thinEVD(evd, mY, mD);                    
        mY = mY * mD.cwiseSqrt().cwiseInverse().asDiagonal();
    }
    
    //! Re-orthonormalize a decomposition (complex case)
    template<typename FMatrix>
    typename boost::disable_if<boost::is_complex<typename ftraits<FMatrix>::Scalar>>::type
    orthonormalize(const FMatrix &mX,
                   Eigen::Matrix<typename ftraits<FMatrix>::Scalar, Eigen::Dynamic, Eigen::Dynamic> &mY,
                   Eigen::Matrix<typename ftraits<FMatrix>::Real, Eigen::Dynamic, 1> &mD) {
        
        // Negative case: copy what we need
        auto negatives = mD.array() < 0;
        typename ftraits<FMatrix>::RealVector _mD;
        typename ftraits<FMatrix>::ScalarMatrix _mY;
        
        Index n = negatives.sum();
        if (n > 0) {
            std::vector<bool> selection(n,false);
            for(Index j = 0; j < mD.rows(); j++)
                if (negatives[j]) selection[j] = true;
            
            select_rows(selection, mD, _mD);
            select_columns(selection, mY, _mY);
        }
        
        
        // Perform the EVD
        typename ftraits<FMatrix>::ScalarMatrix m;
        m.noalias() = mD.cwiseAbs().cwiseSqrt().asDiagonal() * mY.transpose() * mX.inner() * mY * mD.cwiseAbs().cwiseSqrt().asDiagonal();
        
        Eigen::SelfAdjointEigenSolver<decltype(m)> evd(m.template selfadjointView<Eigen::Lower>());
        kqp::thinEVD(evd, mY, mD);                    
        mY = mY * mD.cwiseAbs().cwiseSqrt().cwiseInverse().asDiagonal();
        
        // FIXME: won't work in the real case if we have negative eigenvalues
        if (n > 0) {
            m.resize(mD.rows(), mD.rows());
            auto _m = m.template selfadjointView<Eigen::Lower>();
            m = mD;
            _m.rankUpdate(mY.transpose() * mX.inner() * _mY * _mD, -2);
            typename ftraits<FMatrix>::ScalarMatrix mU;
            thinEVD(Eigen::SelfAdjointEigenSolver<decltype(m)>(_m), mU, mD);
            mD = evd.eigenvalues();
            mY = mY * mU;
        }
    }

}

#endif