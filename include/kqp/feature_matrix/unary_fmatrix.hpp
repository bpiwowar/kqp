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
#ifndef __KQP_SPARSE_FEATURE_MATRIX_H__
#define __KQP_SPARSE_FEATURE_MATRIX_H__

#include <kqp/feature_matrix.hpp>

namespace kqp {
    template<typename KernelUnaryOp, typename Base> class CwiseUnaryKernelFunction 
        : public FeatureMatrix< UnaryKernelFunction<UnaryOp, Base> > {
    public:
        KQP_FMATRIX_COMMON_DEFS(UnaryKernelFunction);

#ifndef SWIG
        UnaryKernelFunction(Base &&base, const KernelUnaryOp &op) : m_base(std::move(base)), m_op(op) {}
#endif

        UnaryKernelFunction(const Base &base, const KernelUnaryOp &op) : m_base(base), m_op(op) {}

        //! Virtual destructor
        virtual ~UnaryKernelFunction() {}
        
        //! Default Constructor
        UnaryKernelFunction() {}
        
    protected:
        void _add(const Self &other, const std::vector<bool> *which = NULL)  {
            m_base.add(other.m_base, which);
        }
                        
        Self _linear_combination(const ScalarAltMatrix & mA, Scalar alpha, const Self *mY, const ScalarAltMatrix *mB, Scalar beta) const {
            if (!m_op.isLinear()) 
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, "Cannot linearly combine wiht operator %s", %KQP_DEMANGLE(m_op));
                
            Base lcBase = m_base.linear_combination(mA, alpha, mY ? &mY->m_base : NULL, mB, beta);
            return Self(std::move(lcBase), m_op);
        }
        
        void _subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Self &into) const {
            Base base = m_base.subset(begin, end, into);
            if (this == &into) {
                m_base = std::move(base);
                m_gramMatrix.resize(0,0)
            } else {
                into = Self(std::move(base), m_op);
            }
        }
        
        const ScalarMatrix &_inner() {
            if (size() == m_gramMatrix.rows()) return m_gramMatrix;
            
            // We lose space here, could be used otherwise???
            Index current = gramMatrix.rows();
            if (current < size()) 
                gramMatrix.conservativeResize(size(), size());
            
            Index tofill = size() - current;

            // Compute the remaining inner products
            auto norms = gramMatrix.diagonal();
            
            m_gramMatrix.bottomRightCorner(tofill, tofill).noalias() = 
                m_op.apply(m_base.inner().bottomRightCorner(tofill, tofill), norms.tail(tofill), norms.tail(tofill));
                
            m_gramMatrix.topRightCorner(current, tofill).noalias() = 
                m_op.apply(m_base.inner().topRightCorner(current, tofill), norms.head(current), norms.tail(tofill));
            
            m_gramMatrix.bottomLeftCorner(tofill, current) = m_gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return m_gramMatrix;
        }
        
        //! Computes the inner product with another matrix
        template<class DerivedMatrix>
        void _inner(const Self &other, DerivedMatrix &result) const {
            m_op.inner(m_base, other.m_base, result)
        }
        
    private:        
        //! The base feature matrix
        Base m_base;
        
        //! The unary operator
        KernelUnaryOp m_op;
        
        //! Cache for inner matrix
        ScalarMatrix m_gramMatrix;
    };


    //! Gaussian Kernel \f$k'(x,y) = \exp(\frac{\vert k(x,x) + k(y,y) - 2k(x,y) \vert}{\sigma^2})\f$
    template<typename FMatrix> struct GaussianKernelOp {
        KQP_FMATRIX_TYPES(FMatrix);
        
        GaussianKernelOp(Real sigma) : m_sigma(sigma) {}
     
        template<class DerivedMatrix>
        void inner(const FMatrix &x, const FMatrix &y, DerivedMatrix &result) const {
            result.noalias() = apply(kqp::inner(this->m_base, other.m_base), x.norms(), y.norms());
        }
        
        template<Derived>
        inline ScalarMatrix apply(const Eigen::MatrixBase<Derived> &k, 
                                  const Eigen::MatrixBase<DerivedRow>& rowNorms, 
                                  const Eigen::MatrixBase<DerivedCol>& colNorms) const { 
            return ((rowNorms.rowwise().replicate(k.cols()) + colNorms.transpose().colwise().replicate(k.rows()) - 2 * k) / (sigma*sigma)).exp();
        }
      
        Real m_sigma;
    };
    
    //! Polynomial Kernel \f$k'(x,y) = (k(x,y) + D)^p\f$
    template<typename FMatrix> struct PolynomialKernelOp {
        KQP_FMATRIX_TYPES(FMatrix);
        
        PolynomialKernelOp(Real bias, int degree) : m_bias(bias), m_degree(degree) {}
     
        template<class DerivedMatrix>
        void inner(const FMatrix &x, const FMatrix &y, DerivedMatrix &result) const {
            result.nolias() = (inner(x,y) + m_bias).pow(m_degree);
        }
        
        template<Derived>
        inline ScalarMatrix apply(const Eigen::MatrixBase<Derived> &k, 
                                  const Eigen::MatrixBase<DerivedRow>& rowNorms, 
                                  const Eigen::MatrixBase<DerivedCol>& colNorms) const { 
            return (k + m_bias).pow(m_degree);
        }
      
        Real m_bias;
        int m_degree;
    };
    
}

#endif