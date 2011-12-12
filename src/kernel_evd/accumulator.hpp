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

#ifndef __KQP_ACCUMULATOR_BUILDER_H__
#define __KQP_ACCUMULATOR_BUILDER_H__

#include <boost/static_assert.hpp>
#include "kernel_evd.hpp"

namespace kqp{
    template <class Derived> 
    void thinEVD(const Eigen::SelfAdjointEigenSolver<Derived> &evd, Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors, 
                 Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> &eigenvalues) {
        typedef typename Derived::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        const Eigen::Matrix<Real, Eigen::Dynamic, 1> &d = evd.eigenvalues();
        double threshold = EPSILON * (double)d.size();
        
        Index rank = d.rows();
        for(Index i = 0; i < d.rows(); i++) {
            if (d[i] < 0)
                if (std::abs(d[i]) > threshold)
                    KQP_THROW_EXCEPTION_F(arithmetic_exception, "Negative value [%e] is above threshold [%e]", %d[i] %threshold);
                else rank--;
                else
                    if (d[i] < threshold) rank--;
                    else break;
        }
        
        eigenvalues = d.tail(rank).cwiseSqrt();
        eigenvectors = evd.eigenvectors().rightCols(rank);
    }
    
    /**
     * @brief Accumulation based computation of the density.
     *
     * Supposes that we can compute a linear combination of the pre-images.
     * Performs an SVD of the feature vectors (if doable) or and EVD of the 
     * inner product of feature vectors.
     * 
     * @ingroup OperatorBuilder
     */
    template <class FMatrix>
    class AccumulatorKernelEVD : public OperatorBuilder<typename FMatrix::FVector> {   
    public:
        typedef typename FMatrix::FVector FVector;
        
        typedef OperatorBuilder<typename FMatrix::FVector> Ancestor;
        typedef typename Ancestor::Scalar Scalar;
        typedef typename Ancestor::Real Real;

        AccumulatorKernelEVD() {
        }
        
        virtual void add(typename Ancestor::Real alpha, const typename Ancestor::FVector &v) {
            fMatrix->add(v);
            factors.resize(factors.rows() + 1);
            // FIXME: won't work if scalar is real and alpha is negative
            factors(factors.rows() - 1) = Eigen::internal::sqrt(alpha);
        }
        
        virtual void add(Real alpha, const typename Ancestor::FMatrix &_fMatrix, const typename Ancestor::Matrix &coefficients) {
            mY.reset();
            
            Index offset = factors.rows();
            factors.resize(offset + coefficients.cols());
            // FIXME: won't work if scalar is real and alpha is negative
            factors.segment(offset, coefficients.cols()).setConstant(Eigen::internal::sqrt(alpha));
            
            // Just add the vectors using linear combination          
            for(Index j = 0; j < coefficients.cols(); j++) 
                fMatrix->add(_fMatrix.linearCombination(alpha, coefficients.col(j)));
            
        }
        
        
        
        virtual typename Ancestor::FMatrixCPtr getX() const {
            compute();
            return fMatrix;
        }
        
        virtual typename Ancestor::MatrixCPtr getY() const {
            compute();
            return mY;
        }
        
        virtual typename Ancestor::RealVectorPtr getD() const {
            compute();
            return mD;
        }
        

        
        //! Actually performs the computation
        void compute() const {
            if (!mY.get()) {
                typedef typename FMatrix::InnerMatrix GramMatrix;
                const GramMatrix gram = factors.asDiagonal() * fMatrix->inner() * factors.asDiagonal();
                
                typedef Eigen::SelfAdjointEigenSolver<typename Ancestor::Matrix> EigenSolver;
                
                EigenSolver evd(gram.template selfadjointView<Eigen::Lower>());
                mY.reset(new typename Ancestor::Matrix());
                mD.reset(new typename Ancestor::RealVector());
                kqp::thinEVD(evd, *mY, *mD);
            }
        }
        
    private:
        mutable typename Ancestor::RealVectorPtr mD;
        mutable typename Ancestor::MatrixPtr mY;
        
        //! concatenation of pre-image matrices
        typename FMatrix::Ptr fMatrix;
        //! Factors
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> factors;
    };
}

#endif
