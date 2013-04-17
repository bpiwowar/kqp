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

#include <kqp/kqp.hpp>
#include <Eigen/Eigenvalues>

#include <kqp/kernel_evd.hpp>
#include <kqp/subset.hpp>

namespace kqp {
    
    template<typename BaseMatrix>
    struct ThinEVD {
        typedef typename BaseMatrix::Scalar Scalar;
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        static void run(const Eigen::SelfAdjointEigenSolver<BaseMatrix> &evd, 
                        ScalarMatrix &eigenvectors, 
                        RealVector &eigenvalues,
                        ScalarMatrix *nullEigenvectors = nullptr,
                        Real threshold = -1) {
            
            // We expect eigenvalues to be sorted by increasing order
            Index dimension = evd.eigenvectors().rows();
            
            const Eigen::Matrix<Real,Dynamic,1> &d = evd.eigenvalues();
            if (threshold < 0)
                threshold = Eigen::NumTraits<Scalar>::epsilon() * (Real)d.size() *  d.cwiseAbs().maxCoeff();
            
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
        static void run(const Eigen::SelfAdjointEigenSolver<BaseMatrix> &evd, 
                        ScalarAltMatrix &eigenvectors, 
                        RealAltVector &eigenvalues,
                        ScalarAltMatrix *nullEigenvectors = nullptr,
                        Real threshold = -1) {
            
            ScalarMatrix _eigenvectors;
            RealVector _eigenvalues;
            boost::scoped_ptr<ScalarMatrix> _nullEigenvectors;
            
            
            if (nullEigenvectors != nullptr) 
                _nullEigenvectors.reset(new ScalarMatrix());
            
            ThinEVD<ScalarMatrix>::run(evd, _eigenvectors, _eigenvalues, _nullEigenvectors.get(), threshold);
            
            eigenvectors.swap(_eigenvectors);
            eigenvalues.swap(_eigenvalues);
            if (nullEigenvectors != nullptr) 
                nullEigenvectors->swap(*_nullEigenvectors);
            
        }
        
    };
    
    
    
    // template<typename Scalar, typename Cond = void> struct Orthonormalize;
    
    template<typename Scalar>
    struct Orthonormalize {
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        //! Orthonormalization with Alt matrices (generic method)
        static void run(const FSpaceCPtr &fs, const FMatrixCPtr &mX,
                        ScalarAltMatrix &mY,
                        RealAltVector &mD) {
            // FIXME: should swap if dense types
            ScalarMatrix _mY(mY);
            RealVector _mD(mD);        
            run(fs, mX, _mY, _mD);
            mY.swap(_mY);
            mD.swap(_mD);
        }    
        
        static void run(const FSpaceCPtr &fs, 
                        const FMatrixCPtr &mX,
                        ScalarMatrix &mY,
                        RealVector &mD) {
            
            // Negative case: copy what we need
            RealVector _mD;
            ScalarMatrix _mY;
            
            Index n = 0;
            std::vector<bool> selection(mD.rows(),false);
            for(Index j = 0; j < mD.rows(); j++) {
                if (mD[j] < 0) {
                    selection[j] = true;
                    n++;
                }
            }
                
            if (n > 0) {
                select_rows(selection, mD, _mD);
                select_columns(selection, mY, _mY);
            }
            
            // Perform the EVD
            ScalarMatrix m = fs->k(mX, mY, RealAltVector(mD.cwiseAbs().cwiseSqrt()));
            
            Eigen::SelfAdjointEigenSolver<decltype(m)> evd(m.template selfadjointView<Eigen::Lower>());
            
            // A = mY mD^1/2
            ScalarMatrix mY2;
            RealVector mD2;
            kqp::ThinEVD<ScalarMatrix>::run(evd, mY2, mD2);                    
            mD2.array() = mD2.array().cwiseAbs(); // just in case of small rounding errors
            mY *= mD.cwiseAbs().cwiseSqrt().asDiagonal() * mY2 * mD2.cwiseSqrt().cwiseInverse().asDiagonal();
            mD = std::move(mD2);
            
            // Handles negative eigenvalues
            if (n > 0) {
                m = mD.template cast<Scalar>().asDiagonal();
                m.template selfadjointView<Eigen::Lower>().rankUpdate(mY.adjoint() * fs->k(mX) * _mY * _mD.cwiseAbs().cwiseSqrt().asDiagonal() , -2);
                ScalarMatrix mU;
                ThinEVD<ScalarMatrix>::run(Eigen::SelfAdjointEigenSolver<ScalarMatrix>(m.template selfadjointView<Eigen::Lower>()), mU, mD);
                mY *= mU;
            }
        }
        
        
        
        
    };
}

#endif