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
#ifndef _KQP_EIGEN_IDENTITY_H_
#define _KQP_EIGEN_IDENTITY_H_

namespace kqp {
    template<typename Derived> struct DiagonalBlockWrapper;
    template<typename Derived> class AltMatrixBase;
    struct IdentityStorage {};
}

namespace Eigen {

    template<typename Derived> class SparseMatrixBase;
    
    template<typename Scalar>
	class Identity {
    public:
        typedef typename MatrixXd::Index Index;
        typedef Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Eigen::Matrix<Scalar,Dynamic,1> > VectorType;
        
        Identity() : m_rows(0), m_cols(0) {}
        Identity(Index rows, Index cols) : m_rows(rows), m_cols(cols) {}
        
        void swap(Identity &other) {
            std::swap(m_rows, other.m_rows);
            std::swap(m_cols, other.m_cols);
        }
        
        Index rows() const { return this->m_rows; }
        Index cols() const { return this->m_cols; }
        
        kqp::DiagonalBlockWrapper<VectorType> block(Index startCol, Index startRow, Index blockRows, Index blockCols) const {
            return kqp::DiagonalBlockWrapper<VectorType>(VectorType(std::min(m_rows,m_cols),1,1),  startCol, startRow, blockRows, blockCols);
        }
        
        template<typename CwiseUnaryOp>
        Identity unaryExpr(const CwiseUnaryOp &) {
            KQP_THROW_EXCEPTION(kqp::illegal_argument_exception, "Cannot apply a unary expression on the Identity matrix");
        }
        
        Scalar trace() const { return std::min(m_rows, m_cols); }
        Scalar sum() const { return std::min(m_rows, m_cols); }
        Scalar squaredNorm() const { return std::min(m_rows, m_cols); }
        
        void resize(Index rows, Index cols) {
            m_rows = rows;
            m_cols = cols;
        }
        
        void conservativeResize(Index rows, Index cols) {
            this->resize(rows, cols);
        }
        
        Scalar operator()(Index i, Index j) const {
            if (i == j) return 1;
            return 0;
        }

        
    private:
        Index m_rows;
        Index m_cols;
    };
    
    template <typename Derived, typename Scalar> \
	const DiagonalWrapper<Derived>  operator* (const Identity<Scalar> &lhs, const DiagonalWrapper<Derived> &rhs) { \
        eigen_assert(lhs.cols() == lhs.rows()); \
        eigen_assert(lhs.cols() == rhs.rows() \
                     && "invalid matrix product" \
                     && "if you wanted a coeff-wise or a dot product use the respective explicit functions"); \
		return rhs.derived(); \
	}
    
    template <typename Derived, typename Scalar> \
	const DiagonalWrapper<Derived> operator* (const DiagonalWrapper<Derived> &lhs, const Identity<Scalar> &rhs) { \
        eigen_assert(rhs.cols() == rhs.rows()); \
        eigen_assert(lhs.cols() == rhs.rows() \
                     && "invalid matrix product" \
                     && "if you wanted a coeff-wise or a dot product use the respective explicit functions"); \
		return lhs.derived();\
	}


    
    /** Pre-Multiplication by identity */
#   define KQP_IDENTITY_PRE_MULT(matrixtype) \
	template <typename Derived, typename Scalar> \
	const typename Eigen::internal::ref_selector<Derived>::type operator* (const Identity<Scalar> &lhs, const matrixtype &rhs) { \
        eigen_assert(lhs.cols() == lhs.rows()); \
        eigen_assert(lhs.cols() == rhs.rows() \
                     && "invalid matrix product" \
                     && "if you wanted a coeff-wise or a dot product use the respective explicit functions"); \
		return rhs.derived(); \
	}
    
	/** Post-Multiplication by identity */
#   define KQP_IDENTITY_POST_MULT(matrixtype) \
	template <typename Derived, typename Scalar> \
	const typename Eigen::internal::ref_selector<Derived>::type operator* (const matrixtype &lhs, const Identity<Scalar> &rhs) { \
        eigen_assert(rhs.cols() == rhs.rows()); \
        eigen_assert(lhs.cols() == rhs.rows() \
                     && "invalid matrix product" \
                     && "if you wanted a coeff-wise or a dot product use the respective explicit functions"); \
		return lhs.derived();\
	}
    
	/** Multiplication with the identity */
#   define KQP_IDENTITY_MULT(type) KQP_IDENTITY_PRE_MULT(type) KQP_IDENTITY_POST_MULT(type)
    
    KQP_IDENTITY_MULT(kqp::AltMatrixBase<Derived>)
    KQP_IDENTITY_MULT(MatrixBase<Derived>)
    KQP_IDENTITY_MULT(SparseMatrixBase<Derived>)
    
	namespace internal {
        template<typename _Scalar>
		struct traits<Identity<_Scalar>> {
            typedef kqp::IdentityStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef _Scalar Scalar;
	 	};
    }
}

#endif
