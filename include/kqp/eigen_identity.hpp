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

#include <kqp/kqp.hpp>
 
namespace kqp {
    template<typename Derived> struct DiagonalBlockWrapper;
    template<typename Derived> class AltMatrixBase;
    struct IdentityStorage {};
}

namespace Eigen {

    template<typename Derived> class SparseMatrixBase;
    
    /** Square identity matrix */
    template<typename Scalar>
	class Identity {
    public:
#ifndef SWIG
        typedef typename MatrixXd::Index Index;
        typedef Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Eigen::Matrix<Scalar,Dynamic,1> > VectorType;
		
		class RowWise {
		public:
			RowWise(Index rows, Index cols) : m_rows(rows), m_cols(cols) {}

			auto squaredNorm() const -> decltype(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(-1)) {
				eigen_assert(m_rows == m_cols); // otherwise, it is a bit more complex! We need a constant expression
				return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(m_rows);
			}
		private:
			Index m_rows;
			Index m_cols;
		};
		
		RowWise rowwise() const { return RowWise(m_rows, m_cols); }
#endif
		
        
        Identity() : m_rows(0), m_cols(0) {}
        Identity(Index size) : m_rows(size), m_cols(size) {}
        
        
        void swap(Identity &other) {
            std::swap(m_rows, other.m_rows);
            std::swap(m_cols, other.m_cols);
        }
        
        Index rows() const { return this->m_rows; }
        Index cols() const { return this->m_cols; }
        
		

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
        
#ifndef SWIG
        /** @deprecated */
        Identity(Index rows, Index cols) : m_rows(rows), m_cols(cols) { eigen_assert(rows == cols); }

        kqp::DiagonalBlockWrapper<VectorType> block(Index startCol, Index startRow, Index blockRows, Index blockCols) const {
            return kqp::DiagonalBlockWrapper<VectorType>(VectorType(std::min(m_rows,m_cols),1,1),  startCol, startRow, blockRows, blockCols);
        }

        template<typename CwiseUnaryOp>
        Identity unaryExpr(const CwiseUnaryOp &) {
            throw kqp::illegal_argument_exception();
            //KQP_THROW_EXCEPTION(kqp::illegal_argument_exception, "Cannot apply a unary expression on the Identity matrix");
        }
        
        auto getVectorIdentity() const -> decltype(Eigen::Matrix<Scalar,Dynamic,1>::Ones(0).asDiagonal()) {
            return Eigen::Matrix<Scalar,Dynamic,1>::Ones(this->rows()).asDiagonal();
        }

        auto getIdentityMatrix() const -> decltype(Eigen::Matrix<Scalar,Dynamic,Dynamic>::Identity(0,0)) {
            return Eigen::Matrix<Scalar,Dynamic,Dynamic>::Identity(this->rows(),this->rows());
        }
#endif
        
    private:
        Index m_rows;
        Index m_cols;
    };
    
    
#ifndef SWIG
    template <typename Derived, typename Scalar> \
	const DiagonalWrapper<Derived>  operator* (const Identity<Scalar> &KQP_M_DEBUG(lhs), const DiagonalWrapper<Derived> &rhs) { \
        eigen_assert(lhs.cols() == rhs.rows() \
                     && "invalid matrix product" \
                     && "if you wanted a coeff-wise or a dot product use the respective explicit functions"); \
		return rhs.derived(); \
	}
    
    template <typename Derived, typename Scalar> \
	const DiagonalWrapper<Derived> operator* (const DiagonalWrapper<Derived> &lhs, const Identity<Scalar> &KQP_M_DEBUG(rhs)) { \
        eigen_assert(lhs.cols() == rhs.rows() \
                     && "invalid matrix product" \
                     && "if you wanted a coeff-wise or a dot product use the respective explicit functions"); \
		return lhs.derived();\
	}


    
    /** Pre-Multiplication by identity */
#   define KQP_IDENTITY_PRE_MULT(matrixtype) \
	template <typename Derived, typename Scalar> \
	const typename Eigen::internal::ref_selector<Derived>::type operator* (const Identity<Scalar> &KQP_M_DEBUG(lhs), const matrixtype &rhs) { \
        eigen_assert(lhs.cols() == rhs.rows() \
                     && "invalid matrix product" \
                     && "if you wanted a coeff-wise or a dot product use the respective explicit functions"); \
		return rhs.derived(); \
	}
    
	/** Post-Multiplication by identity */
#   define KQP_IDENTITY_POST_MULT(matrixtype) \
	template <typename Derived, typename Scalar> \
	const typename Eigen::internal::ref_selector<Derived>::type operator* (const matrixtype &lhs, const Identity<Scalar> &KQP_M_DEBUG(rhs)) { \
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
    

    /** Operator - or +  */
#   define KQP_IDENTITY_ADD(op, matrixtype) \
    template <typename Derived, typename Scalar> \
    auto operator op (const Identity<Scalar> &lhs, const matrixtype &rhs) -> decltype(lhs.getIdentityMatrix() op rhs.derived()) { \
        eigen_assert(lhs.cols() == rhs.cols()); \
        eigen_assert(lhs.rows() == rhs.rows()); \
        return lhs.getIdentityMatrix() op rhs.derived(); \
    } \
    template <typename Derived, typename Scalar> \
    auto operator op (const matrixtype &lhs, const Identity<Scalar> &rhs) -> decltype(lhs.derived() op rhs.getIdentityMatrix()) { \
        eigen_assert(lhs.cols() == rhs.cols()); \
        eigen_assert(lhs.rows() == rhs.rows()); \
    return lhs.derived() op rhs.getIdentityMatrix();\
    }

#   define KQP_IDENTITY_OP(mtype) KQP_IDENTITY_ADD(+, mtype) KQP_IDENTITY_ADD(-, mtype)
    
    KQP_IDENTITY_OP(MatrixBase<Derived>);
    KQP_IDENTITY_OP(SparseMatrixBase<Derived>);
    KQP_IDENTITY_OP(kqp::AltMatrixBase<Derived>)

	namespace internal {
        template<typename _Scalar>
		struct traits<Identity<_Scalar>> {
            typedef kqp::IdentityStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef _Scalar Scalar;
	 	};
    }
#endif // SWIG

}

#endif
