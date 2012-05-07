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
#ifndef __KQP_UNARY_KERNEL_SPACE_H__
#define __KQP_UNARY_KERNEL_SPACE_H__

#include <boost/unordered_map.hpp>
#include <kqp/feature_matrix.hpp>

namespace kqp {
    template<typename Scalar> 
    class UnaryKernelSpace : public SpaceBase< Scalar > {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);

        UnaryKernelSpace(const FSpace &base) :  m_base(base) {}

        ~UnaryKernelSpace() {}
        
        virtual FMatrixBasePtr newMatrix() const override { return m_base->newMatrix(); }
        virtual FMatrixBasePtr newMatrix(const FeatureMatrixBase<Scalar> &other) const override { return m_base->newMatrix(other); }        

        virtual const ScalarMatrix &k(const FMatrixBase &mX) const override {            
            ScalarMatrix &gram = m_gramCache[static_cast<const void*>(&mX)];
            
            if (mX.size() == gram.rows()) return gram;
            
            // We lose space here, could be used otherwise???
            Index current = gram.rows();
            if (current < mX.size()) 
                gram.conservativeResize(mX.size(), mX.size());
            
            Index tofill = mX.size() - current;

            fillGram(gram, tofill, mX);
            
            gram.bottomLeftCorner(tofill, current).noalias() = gram.topRightCorner(current, tofill).adjoint().eval();
            
            return gram;
        }
        

    protected:
        /**
         * @brief Fill a part of the Gram matrix.
         *
         * Fill the right columns (tofill) of a Gram matrix
         */
        virtual void fillGram(ScalarMatrix &gram, Index tofill, const FMatrixBase &mX) const = 0;

        //! The base feature space 
        FSpace m_base;

    private:              
        //! Gram matrices
        mutable boost::unordered_map<const void *, ScalarMatrix> m_gramCache;
        
    };


    //! Gaussian Kernel \f$k'(x,y) = \exp(\frac{\vert k(x,x) + k(y,y) - 2k(x,y) \vert}{\sigma^2})\f$
    template<typename Scalar> class GaussianSpace : public UnaryKernelSpace<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        using UnaryKernelSpace<Scalar>::m_base;
        using UnaryKernelSpace<Scalar>::k;

        GaussianSpace(Real sigma, const FSpace &base) : UnaryKernelSpace<Scalar>(base), m_sigma(sigma) {}
        virtual FSpaceBasePtr copy() const override { return FSpaceBasePtr(new GaussianSpace<Scalar>(m_sigma, m_base)); }        

        virtual Index dimension() const override { return -1; }
        virtual bool canLinearlyCombine() const override { return false; }

        
        virtual ScalarMatrix k(const FMatrixBase &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1, 
                               const FMatrixBase &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const override {
            return mD1.asDiagonal() * mY1.transpose() 
                    * this->f(m_base->k(mX1, mX2), m_base->k(mX1).diagonal(), m_base->k(mX2).diagonal())
                    * mY2 * mD2.asDiagonal();
        }
        
    protected:
        virtual void fillGram(ScalarMatrix &gram, Index tofill, const FMatrixBase &mX) const override {
            // Compute the remaining inner products
            auto norms = gram.diagonal();
            
            gram.rightCols(tofill) = this->f(m_base->k(mX).rightCols(tofill), norms, norms.tail(tofill));            
        }
        

    private:
        
        template<typename Derived, typename DerivedRow, typename DerivedCol>
        inline ScalarMatrix f(const Eigen::MatrixBase<Derived> &k, 
                                  const Eigen::MatrixBase<DerivedRow>& rowNorms, 
                                  const Eigen::MatrixBase<DerivedCol>& colNorms) const { 
            return ((rowNorms.derived().rowwise().replicate(k.cols()) + colNorms.derived().transpose().colwise().replicate(k.rows()) - 2 * k.derived()) / (m_sigma*m_sigma)).array().exp();
        }
      
        Real m_sigma;
    };
    
    //! Polynomial Kernel \f$k'(x,y) = (k(x,y) + D)^p\f$
    template<typename Scalar> class PolynomialSpace  : public UnaryKernelSpace<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        using UnaryKernelSpace<Scalar>::m_base;
        using UnaryKernelSpace<Scalar>::k;

        virtual FSpaceBasePtr copy() const override { return FSpaceBasePtr(new PolynomialSpace<Scalar>(m_bias, m_degree, m_base)); }        

        PolynomialSpace(Real bias, int degree, const FSpace &base) : UnaryKernelSpace<Scalar>(base), m_bias(bias), m_degree(degree) {}
        virtual Index dimension() const override { return -1; }
        virtual bool canLinearlyCombine() const override { return false; }

        virtual ScalarMatrix k(const FMatrixBase &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1, 
                               const FMatrixBase &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const override {
            return mD1.asDiagonal() * mY1.transpose() * this->f(m_base->k(mX1,mX2)) * mY2 * mD2.asDiagonal();
        }
        
    protected:
        virtual void fillGram(ScalarMatrix &gram, Index tofill, const FMatrixBase &mX) const override {           
            gram.rightCols(tofill) = this->f(m_base->k(mX).rightCols(tofill));
        }

        
    private:
        template<typename Derived>
        inline ScalarMatrix f(const Eigen::MatrixBase<Derived> &k) const { 
            return (k.derived() + ScalarMatrix::Constant(k.rows(), k.cols(), m_bias)).array().pow(m_degree);
        }
      
        Real m_bias;
        int m_degree;
    };
    
}

#define KQP_SCALAR_GEN(scalar) \
    extern template class kqp::GaussianSpace<scalar>; \
    extern template class kqp::PolynomialSpace<scalar>;
#include <kqp/for_all_scalar_gen.h.inc>
#undef KQP_SCALAR_GEN

#endif