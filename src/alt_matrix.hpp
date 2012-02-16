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

#ifndef _KQP_ALTMATRIX_H_
#define _KQP_ALTMATRIX_H_

#include <iostream>
#include <algorithm>
#include "Eigen/Core"
#include "kqp.hpp"

namespace Eigen {
    template<typename Lhs, typename Rhs, bool Tr> class AltDenseProduct;
    template<typename Lhs, typename Rhs, bool Tr> class AltDiagonalProduct;
}

namespace kqp {
    
    struct AltMatrixStorage {};
    
    template<typename Derived> class AltMatrixBase  : public Eigen::EigenBase< Derived > {
    public:
        typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
        typedef typename Eigen::internal::packet_traits<Scalar>::type PacketScalar;
        typedef typename Eigen::internal::traits<Derived>::StorageKind StorageKind;
        typedef typename Eigen::internal::traits<Derived>::Index Index;
        
        enum {
            RowsAtCompileTime = Eigen::internal::traits<Derived>::RowsAtCompileTime,
            ColsAtCompileTime = Eigen::internal::traits<Derived>::ColsAtCompileTime,
            SizeAtCompileTime = (Eigen::internal::size_at_compile_time<Eigen::internal::traits<Derived>::RowsAtCompileTime,
                                 Eigen::internal::traits<Derived>::ColsAtCompileTime>::ret),
            MaxRowsAtCompileTime = RowsAtCompileTime,
            MaxColsAtCompileTime = ColsAtCompileTime,
            MaxSizeAtCompileTime = (Eigen::internal::size_at_compile_time<MaxRowsAtCompileTime,
                                    MaxColsAtCompileTime>::ret),
            
            IsVectorAtCompileTime = RowsAtCompileTime == 1 || ColsAtCompileTime == 1,
            Flags = Eigen::internal::traits<Derived>::Flags,
            CoeffReadCost = Eigen::internal::traits<Derived>::CoeffReadCost,
            IsRowMajor = Flags&Eigen::RowMajorBit ? 1 : 0,
            
            _HasDirectAccess = 0,
            
        };
        typedef typename Eigen::internal::conditional<_HasDirectAccess, const Scalar&, Scalar>::type CoeffReturnType;
        
        
    };
    
    // The different underlying types of AltMatrix
    enum AltMatrixType {
        /// Identity
        IDENTITY,
        
        /// Diagonal matrix, stored as a column vector
        DIAGONAL,
        
        /// Dense matrix
        DENSE
    };
    
    
    /**
     * This matrix can be either the Identity or a dense matrix.
     *
     * As this is known only at execution time, we have to create a new matrix type for Eigen.
     *
     */
    template<typename _Scalar> class AltMatrix : public AltMatrixBase< AltMatrix<_Scalar> > {
    public:
        // Type when nested: a const reference
        typedef const AltMatrix& Nested;
        
        enum {
            Flags = 0x0,
            IsVectorAtCompileTime = 0,
            Options = 0
        };
        
        typedef _Scalar Scalar;
        typedef long Index;
        typedef AltMatrix<Scalar> PlainObject;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
        
        static AltMatrix<Scalar> Identity(Index size, Scalar alpha = (Scalar)1)  {
            return AltMatrix<Scalar>(IDENTITY, size, size, alpha);
        }
        
        Index rows() const { return _rows; }
        Index cols() const { return _cols; }
        AltMatrixType type() const { return _type; }
        Scalar alpha() const { return _alpha; }
        
        const Eigen::Transpose<const AltMatrix> adjoint() const { return Eigen::Transpose<const AltMatrix>(*this); }
        
        
        // Initialisation from a dense matrix
        template<class Derived>
        AltMatrix(const Eigen::MatrixBase<Derived> &other) : _type(DENSE), _dense_matrix(other), _rows(other.rows()), _cols(other.cols()), _alpha(1) {
        }
        
        template<class Derived>
        AltMatrix(const Eigen::DiagonalBase<Derived> &other) 
        : _type(DIAGONAL), _dense_matrix(other.derived().diagonal()), _rows(other.rows()), _cols(other.cols()), _alpha(1) {
        }
        
        /// Default initialisation: dense empty matrix
        AltMatrix() : _type(DENSE), _rows(0), _cols(0), _alpha(1) {
            
        }
        
        /// Return a const reference to the underlying dense matrix
        const DenseMatrix &dense_matrix() const {
            return _dense_matrix;
        }
        
        //! Takes the ownership
        void swap_dense(DenseMatrix &other) {
            _dense_matrix.swap(other);
            configure_dense();
        }
        
        //! Swap two matrices
        void swap(AltMatrix<Scalar> &other) {
            _dense_matrix.swap(other._dense_matrix);
            std::swap(_type, other._type);
            std::swap(_rows, other._rows);
            std::swap(_cols, other._cols);
            std::swap(_alpha, other._alpha);            
        }
        
        template<class Derived>
        AltMatrix &operator=(const Eigen::MatrixBase<Derived> &m) {
            _dense_matrix = m;
            configure_dense();
            return *this;
        }
        
        Scalar operator()(Index i, Index j) const {
            switch(_type) {
                case IDENTITY:
                    return i == j ? _alpha : 0;
                case DIAGONAL:
                    return i == j ? _alpha * _dense_matrix(i,0) : 0;
                case DENSE:
                    return _dense_matrix(i,j);
            }
            KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
        }
        
        inline Scalar coeff(Index i, Index j) const {
            return operator()(i,j);
            
            
        }
        
        template<typename Dest> inline void evalTo(Dest& dst) const {
            switch(_type) {
                case IDENTITY:
                    dst = _alpha * DenseMatrix::Identity(_rows, _cols);
                    break;
                case DIAGONAL:
                    dst = (_alpha * _dense_matrix.col(0)).asDiagonal();
                    break;
                case DENSE:
                    dst = _dense_matrix;
                    break;
                default:
                    KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
            }
        }
        
    private:
        
        template<typename Lhs, typename Rhs, bool Tr> friend class Eigen::AltDenseProduct;
        template<typename Lhs, typename Rhs, bool Tr> friend class Eigen::AltDiagonalProduct;
        
        // Derive class members from the dense matrix
        void configure_dense() {
            _type = DENSE;
            _rows = _dense_matrix.rows();
            _cols = _dense_matrix.cols();
            _alpha = 1;
        }
        
        
        AltMatrix(AltMatrixType type, Index rows, Index cols, Scalar alpha) :  _type(type), _rows(rows), _cols(cols), _alpha(alpha) {
        }
        
        /// Type of the matrix
        AltMatrixType _type;
        
        /// Storage (used for DIAGONAL and DENSE types)
        DenseMatrix _dense_matrix;
        
        /// Cache for the number of rows
        Index _rows;
        
        /// Cache for the number of columns
        Index _cols;
        
        /// The scaling factor
        Scalar _alpha;
    };
    
    
    template<typename Scalar>
    std::ostream &operator<<(std::ostream &out, const AltMatrix<Scalar> &alt_matrix) {
        switch(alt_matrix.type()) {
            case IDENTITY:
                return out << "[Identity matrix of dimension " << alt_matrix.rows() << " with scalar=" << KQP_DEMANGLE((Scalar)0) << "]";
            case DIAGONAL:
                return out << "[Diagonal matrix of dimension " << alt_matrix.rows() << " with scalar=" << KQP_DEMANGLE((Scalar)0) << "]"
                << std::endl << alt_matrix.dense_matrix().transpose();
            case DENSE:
                return out << "[Dense matrix]" << std::endl << alt_matrix.dense_matrix();
        }
        KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
    }
    
}

namespace Eigen {
    
    // --- AltMatrix traits ---
    
    namespace internal {
        template<> struct promote_storage_type<Dense,kqp::AltMatrixStorage>
        { typedef Dense ret; };
        template<> struct promote_storage_type<kqp::AltMatrixStorage, Dense>
        { typedef Dense ret; };
        
        template<typename Scalar> struct eval<kqp::AltMatrix<Scalar>, kqp::AltMatrixStorage> {
            typedef const kqp::AltMatrix<Scalar> & type;
        };
        template<typename Scalar> struct eval<const kqp::AltMatrix<Scalar>, kqp::AltMatrixStorage> {
            typedef const kqp::AltMatrix<Scalar> & type;
        };
        
        template<typename Scalar> struct eval<Eigen::Transpose<kqp::AltMatrix<Scalar> >, kqp::AltMatrixStorage> {
            typedef const Eigen::Transpose< kqp::AltMatrix<Scalar> >& type;
        };
        template<typename Scalar> struct eval<Eigen::Transpose<const kqp::AltMatrix<Scalar> >, kqp::AltMatrixStorage> {
            typedef const Eigen::Transpose< kqp::AltMatrix<Scalar> >& type;
        };
        /// Traits for an AltMatrix
        template<typename _Scalar> 
        struct traits<kqp::AltMatrix<_Scalar> > {
            typedef _Scalar Scalar;
            typedef kqp::AltMatrixStorage StorageKind;
            typedef MatrixXpr XprKind;
            
            typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Index Index;
            enum {
                RowsAtCompileTime    = Dynamic,
                ColsAtCompileTime    = Dynamic,
                MaxRowsAtCompileTime = Dynamic,
                MaxColsAtCompileTime = Dynamic,
                
                CoeffReadCost = NumTraits<Scalar>::ReadCost,
                SupportedAccessPatterns = 0x0,
                Flags = 0x0,
                
            };
        };
    }
    
    // --- Transpose of an AltMatrix
    template<typename Scalar> class TransposeImpl<kqp::AltMatrix<Scalar>,kqp::AltMatrixStorage> 
    : public kqp::AltMatrixBase<Transpose<kqp::AltMatrix<Scalar> > > {
    public:
        typedef TransposeImpl<kqp::AltMatrix<Scalar>,kqp::AltMatrixStorage> Base;
        typedef kqp::AltMatrix<Scalar> PlainObject;
        typedef const Eigen::Transpose<kqp::AltMatrix<Scalar> > Object;
        
        inline const PlainObject & matrix() const { return static_cast<Object&>(*this).nestedExpression(); }
        
        Scalar alpha() const { return Eigen::internal::conj(matrix().alpha()); }
        kqp::AltMatrixType type() const { return matrix().type(); }
        const typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::AdjointReturnType dense_matrix() const { return matrix().dense_matrix().adjoint(); }
    };
    
    
    template<typename Scalar> class TransposeImpl<const kqp::AltMatrix<Scalar>,kqp::AltMatrixStorage> 
    : public kqp::AltMatrixBase<Transpose<kqp::AltMatrix<Scalar> > > {
    public:
        
        typedef kqp::AltMatrix<Scalar> PlainObject;
        typedef TransposeImpl<const kqp::AltMatrix<Scalar>,kqp::AltMatrixStorage> Base;
        typedef Eigen::Transpose<kqp::AltMatrix<Scalar> > Object;
        
        inline const PlainObject & matrix() const { return static_cast<Object&>(*this).nestedExpression(); }
        
        double alpha() const { return Eigen::internal::conj(matrix().alpha()); }
        kqp::AltMatrixType type() const { return matrix().type(); }
        const typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::AdjointReturnType dense_matrix() const { return matrix().dense_matrix().adjoint(); }
    };
    
} // namespace Eigen



// ---- Product: AltMatrix with Dense ---
namespace Eigen { 
    template<typename Lhs, typename Rhs, bool Tr>
    class AltDenseProduct;// : public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,Tr>, Lhs, Rhs> {};
    
    template<typename Lhs, typename Rhs>
    class AltDenseProduct<Lhs,Rhs,false> : public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,false>, Lhs, Rhs> 
    {
    public:
        EIGEN_PRODUCT_PUBLIC_INTERFACE(AltDenseProduct)
        typedef internal::traits<AltDenseProduct> Traits;
        
        
        AltDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {
        }
        
        template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar beta) const {
            Scalar x = (this->lhs().alpha() * beta);
            switch(this->lhs().type()) {
                case kqp::IDENTITY:
                    dest += x * this->rhs();
                    break;
                case kqp::DIAGONAL:
                    dest += x * (this->lhs().dense_matrix().col(0).asDiagonal() * this->rhs());
                    break;
                case kqp::DENSE:
                    dest += x * this->lhs().dense_matrix() * this->rhs();
                    break;
                default:
                    abort();
            }
            
        }
    private:
        AltDenseProduct operator=(const AltDenseProduct &);
    };
    
    template<typename Lhs, typename Rhs>
    class AltDenseProduct<Lhs,Rhs,true> : public ProductBase<AltDenseProduct<Lhs,Rhs,true>, Lhs, Rhs>
    {
    public:
        typedef AltDenseProduct<Lhs,Rhs,true> Self;
        EIGEN_PRODUCT_PUBLIC_INTERFACE(Self)
        typedef internal::traits<AltDenseProduct> Traits;
        
        AltDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}
        
        template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar beta) const {
            Scalar x = (this->rhs().alpha() * beta);
            switch(this->rhs().type()) {
                case kqp::IDENTITY:
                    dest += x * this->lhs();
                    break;
                case kqp::DIAGONAL:
                    dest += x * this->lhs() * this->rhs().dense_matrix().col(0).asDiagonal();
                    break;
                case kqp::DENSE:
                    dest += x * this->lhs() * this->rhs().dense_matrix();
                    break;
                default:
                    abort();
            }
        }
    private:
        AltDenseProduct operator=(const AltDenseProduct &);
    };
    
    template<typename Lhs, typename Rhs> struct AltDenseProductReturnType
    {
        typedef AltDenseProduct<Lhs,Rhs,false> Type;
    };
    
    template<typename Lhs, typename Rhs> struct DenseAltProductReturnType
    {
        typedef AltDenseProduct<Lhs,Rhs,true> Type;
    };
    
    
    
    template<class Derived,class OtherDerived>
    inline const typename AltDenseProductReturnType<Derived,OtherDerived>::Type
    operator*(const kqp::AltMatrixBase<Derived> &a, const Eigen::MatrixBase<OtherDerived> &b) {
        return typename AltDenseProductReturnType<Derived,OtherDerived>::Type(a.derived(), b.derived());
    }
    
    template<class Derived,class OtherDerived>
    inline const typename DenseAltProductReturnType<Derived,OtherDerived>::Type
    operator*(const Eigen::MatrixBase<Derived> &a, const kqp::AltMatrixBase<OtherDerived> &b) {
        return typename DenseAltProductReturnType<Derived,OtherDerived>::Type(a.derived(), b.derived());
    }    
    
    
    // Alt - Dense
    template<typename Lhs, typename Rhs>
    struct AltDenseDifference : public CwiseBinaryOp<internal::scalar_difference_op<typename internal::traits<Lhs>::Scalar>, Lhs, Rhs> {
        typedef CwiseBinaryOp<internal::scalar_difference_op<typename internal::traits<Lhs>::Scalar>, Lhs, Rhs> Base;
        
        AltDenseDifference(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}
    };
    
    template<class Derived,class OtherDerived>
    const AltDenseDifference<Derived, OtherDerived> 
    operator-(const kqp::AltMatrixBase<Derived> &a, const Eigen::MatrixBase<OtherDerived> &b) {
        return AltDenseDifference<Derived, OtherDerived>(a.derived(), b.derived());
    }
    
    
    
    
    namespace internal {        
        /// Traits for a product dense x alt
        template<typename Lhs, typename Rhs, bool Tr>
        struct traits<Eigen::AltDenseProduct<Lhs,Rhs,Tr> >
        : traits<ProductBase<Eigen::AltDenseProduct<Lhs,Rhs,Tr>, Lhs, Rhs> >
        {
            typedef typename scalar_product_traits<typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar>::ReturnType Scalar;
            
            
            typedef Dense StorageKind;
            typedef MatrixXpr XprKind;
            
            typedef typename Lhs::Index Index;
            
            typedef typename Lhs::Nested LhsNested;
            typedef typename Rhs::Nested RhsNested;
            typedef typename remove_all<LhsNested>::type _LhsNested;
            typedef typename remove_all<RhsNested>::type _RhsNested;
            
            enum {
                Flags = EvalBeforeNestingBit,
                RowsAtCompileTime    = int(traits<Lhs>::RowsAtCompileTime),
                ColsAtCompileTime    = int(traits<Rhs>::ColsAtCompileTime),
                MaxRowsAtCompileTime = int(traits<Lhs>::MaxRowsAtCompileTime),
                MaxColsAtCompileTime = int(traits<Rhs>::MaxColsAtCompileTime),
                
                LhsCoeffReadCost = traits<_LhsNested>::CoeffReadCost,
                RhsCoeffReadCost = traits<_RhsNested>::CoeffReadCost,
                
                CoeffReadCost = LhsCoeffReadCost + RhsCoeffReadCost + NumTraits<Scalar>::MulCost,
                
                _HasDirectAccess = 0
            };
        };
        
        
    } // end namespace internal
    
    
} // end namespace Eigen

// ---- Product: AltMatrix with Diagonal ---
namespace Eigen { 
    
    
    template<typename Lhs, typename Rhs, bool InverseOrder> class AltDiagonalProduct;
    
    
    // Diagonal times AltMatrix
    template<typename Lhs, typename Rhs>
    class AltDiagonalProduct<Lhs, Rhs, true> : public kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,true> >
    {
    public:
        typedef typename Lhs::Nested LhsNested;
        typedef typename internal::remove_all<LhsNested>::type _LhsNested;
        
        typedef typename Rhs::Nested RhsNested;
        typedef typename internal::remove_all<RhsNested>::type _RhsNested;
        
        typedef AltDiagonalProduct<Lhs,Rhs,true> Self;
        typedef kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,true> > Base;
        
        EIGEN_GENERIC_PUBLIC_INTERFACE(Self)
        typedef internal::traits<AltDiagonalProduct> Traits;
        
        
        AltDiagonalProduct(const Lhs& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs) {
            eigen_assert(lhs.cols() == rhs.rows()
                         && "invalid matrix product"
                         && "if you wanted a coeff-wise or a dot product use the respective explicit functions");
        }
        
        inline Index rows() const { return m_lhs.rows(); }
        inline Index cols() const { return m_rhs.cols(); }
        
        template<typename Dest> inline void evalTo(Dest& dst) const {
            dst = eval();
        }
        kqp::AltMatrix<Scalar> eval() const {
            switch(m_rhs.type()) {
                case kqp::IDENTITY:
                    return kqp::AltMatrix<Scalar>( m_rhs.alpha() * m_lhs.diagonal() );
                case kqp::DIAGONAL: 
                    return kqp::AltMatrix<Scalar>((m_rhs.alpha() * m_rhs.dense_matrix().col(0).cwiseProduct(m_lhs.diagonal())).asDiagonal());
                case kqp::DENSE: {
                    return kqp::AltMatrix<Scalar>(m_rhs.alpha() * (m_lhs * m_rhs.dense_matrix()));  
                }
            }
            abort();
        }
        
        
    protected:
        
        const LhsNested m_lhs;
        const RhsNested m_rhs;
        
    private:
        AltDiagonalProduct operator=(const AltDiagonalProduct &);
    };
    
    // AltMatrix times Diagonal
    template<typename Lhs, typename Rhs>
    class AltDiagonalProduct<Lhs, Rhs, false> : public kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,false> >
    {
    public:
        typedef typename Lhs::Nested LhsNested;
        typedef typename internal::remove_all<LhsNested>::type _LhsNested;
        
        typedef typename Rhs::Nested RhsNested;
        typedef typename internal::remove_all<RhsNested>::type _RhsNested;
        
        typedef AltDiagonalProduct<Lhs,Rhs,true> Self;
        typedef kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,true> > Base;
        
        EIGEN_GENERIC_PUBLIC_INTERFACE(Self)
        typedef internal::traits<AltDiagonalProduct> Traits;
        
        
        AltDiagonalProduct(const Lhs& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs) {
            eigen_assert(lhs.cols() == rhs.rows()
                         && "invalid matrix product"
                         && "if you wanted a coeff-wise or a dot product use the respective explicit functions");
        }
        
        inline Index rows() const { return m_lhs.rows(); }
        inline Index cols() const { return m_rhs.cols(); }
        
        template<typename Dest> inline void evalTo(Dest& dst) const {
            dst = eval();
        }
        
        kqp::AltMatrix<Scalar> eval() const {
            switch(m_lhs.type()) {
                case kqp::IDENTITY:
                    return kqp::AltMatrix<Scalar>( m_lhs.alpha() * m_rhs.diagonal() );
                case kqp::DIAGONAL: 
                    return kqp::AltMatrix<Scalar>((m_lhs.alpha() * m_lhs.dense_matrix().col(0).cwiseProduct(m_rhs.diagonal())).asDiagonal());
                case kqp::DENSE: {
                    return kqp::AltMatrix<Scalar>(m_lhs.alpha() * ( m_lhs.dense_matrix() * m_rhs ));  
                }
            }
            abort();
        }
        
        
    protected:
        
        const LhsNested m_lhs;
        const RhsNested m_rhs;
        
    private:
        AltDiagonalProduct operator=(const AltDiagonalProduct &);
    };
    
    
    
    
    
    
    
    template<class Derived,class OtherDerived>
    inline const AltDiagonalProduct<Derived,OtherDerived,false>
    operator*(const kqp::AltMatrixBase<Derived> &a, const Eigen::DiagonalBase<OtherDerived> &b) {
        return AltDiagonalProduct<Derived,OtherDerived,false>(a.derived(), b.derived());
    }
    
    template<class Derived,class OtherDerived>
    inline const AltDiagonalProduct<Derived,OtherDerived,true>
    operator*(const Eigen::DiagonalBase<Derived> &a, const kqp::AltMatrixBase<OtherDerived> &b) {
        return AltDiagonalProduct<Derived,OtherDerived,true>(a.derived(), b.derived());
    }    
    
    
    
    
    namespace internal {       
        template<typename Type> struct kqp_traits {
            enum {
                CoeffReadCost = traits<Type>::CoeffReadCost
            };
        };
        
        template<typename Derived> struct kqp_traits<Eigen::DiagonalWrapper<Derived> > {
            enum {
                CoeffReadCost = 1
            };            
        };
        
        template<typename Lhs, typename Rhs, bool Tr> struct eval<const AltDiagonalProduct<Lhs, Rhs, Tr> > {
            typedef typename scalar_product_traits<typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar>::ReturnType Scalar;
            typedef const kqp::AltMatrix<Scalar> & type;
            
        };
        template<typename Lhs, typename Rhs, bool Tr> struct eval<AltDiagonalProduct<Lhs, Rhs, Tr> > {
            typedef typename scalar_product_traits<typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar>::ReturnType Scalar;
            typedef kqp::AltMatrix<Scalar> & type;
        };
        
        
        /// Traits for a product dense x alt
        template<typename Lhs, typename Rhs, bool Tr>
        struct traits<Eigen::AltDiagonalProduct<Lhs,Rhs,Tr> >
        : traits<ProductBase<Eigen::AltDiagonalProduct<Lhs,Rhs,Tr>, Lhs, Rhs> >
        {
            typedef typename scalar_product_traits<typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar>::ReturnType Scalar;
            
            
            typedef kqp::AltMatrixStorage StorageKind;
            typedef MatrixXpr XprKind;
            
            typedef typename Lhs::Index Index;
            
            typedef typename Lhs::Nested LhsNested;
            typedef typename Rhs::Nested RhsNested;
            typedef typename remove_all<LhsNested>::type _LhsNested;
            typedef typename remove_all<RhsNested>::type _RhsNested;
            
            enum {
                Flags = EvalBeforeNestingBit,
                RowsAtCompileTime    = int(traits<Lhs>::RowsAtCompileTime),
                ColsAtCompileTime    = int(traits<Rhs>::ColsAtCompileTime),
                MaxRowsAtCompileTime = int(traits<Lhs>::MaxRowsAtCompileTime),
                MaxColsAtCompileTime = int(traits<Rhs>::MaxColsAtCompileTime),
                
                LhsCoeffReadCost = 1,
                RhsCoeffReadCost = 1,
                
                CoeffReadCost = LhsCoeffReadCost + RhsCoeffReadCost + NumTraits<Scalar>::MulCost,
                
                _HasDirectAccess = 0
            };
        };
        
        
    } // end namespace internal
    
    
} // end namespace Eigen

#endif


