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

#include <Eigen/Core>
#include <kqp/kqp.hpp>

// Forward declarations
namespace Eigen {
    template<typename Lhs, typename Rhs, int ProductOrder> class AltDenseProduct;
    template<typename Lhs, typename Rhs, int ProductOrder> class AltDiagonalProduct;
    
    namespace internal {
        //! Local traits
        template<typename Derived> struct kqp_traits;
    }
}

namespace kqp {
    
    //! Type for AltMatrix storage type
    struct AltMatrixStorage {};
    
    template<typename Scalar> class AltMatrix; 
    
    // ----
    // ---- Alt Matrix base
    // ----
    
    //! Base class for AltMatrix
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
        
        typedef kqp::AltMatrix<Scalar> PlainObject;
    };
    
    
    
    
    // ----
    // ---- Alt Matrix
    // ----
    
    
    // The different underlying types of AltMatrix
    enum AltMatrixType {
        /// Identity
        IDENTITY,
        
        /// Diagonal matrix, stored as a column vector
        DIAGONAL,
        
        /// Dense matrix
        DENSE
    };
    
    
    template<typename _AltMatrix> class AltBlock;
    template<typename _AltMatrix> class RowWise;
    
    /**
     * This matrix can be either the Identity or a dense matrix.
     *
     * As this is known only at execution time, we have to create a new matrix type for Eigen.
     *
     */
    template<typename _Scalar> class AltMatrix : public AltMatrixBase< AltMatrix<_Scalar> > {
    public:
        // Type when nested: a const reference
        typedef AltMatrix<_Scalar> Self;
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
        typedef DenseMatrix DenseType;
        
        static AltMatrix<Scalar> Identity(Index size)  {
            return AltMatrix<Scalar>(IDENTITY, size, size);
        }
        
        Index rows() const { return _rows; }
        Index cols() const { return _cols; }
        AltMatrixType type() const { return _type; }
        
        //! Returns the adjoint 
        Eigen::Transpose<const AltMatrix> transpose() const { return Eigen::Transpose<const AltMatrix>(*this); }
        
        
        // Initialisation from a dense matrix
        template<class Derived>
        AltMatrix(const Eigen::MatrixBase<Derived> &other) : _type(DENSE), _dense_matrix(other), _rows(other.rows()), _cols(other.cols()) {
        }
        
        template<class Derived>
        AltMatrix(const Eigen::DiagonalBase<Derived> &other) 
        : _type(DIAGONAL), _dense_matrix(other.derived().diagonal()), _rows(other.rows()), _cols(other.cols()) {
        }
        
        /// Default initialisation: dense empty matrix
        AltMatrix() : _type(DENSE), _rows(0), _cols(0) {
            
        }
        
        /// Return a const reference to the underlying dense matrix
        const DenseMatrix &dense_matrix() const {
            return _dense_matrix;
        }
        
        const DenseMatrix &asDense() const {
            return _dense_matrix;
        }
        
        
        typedef Eigen::DiagonalWrapper<typename DenseMatrix::ConstColXpr> DiagonalType;
        const DiagonalType asDiagonal() const {
            return _dense_matrix.col(0).asDiagonal();
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
                    return i == j ? 1 : 0;
                case DIAGONAL:
                    return i == j ? _dense_matrix(i,0) : 0;
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
                    dst = DenseMatrix::Identity(_rows, _cols);
                    break;
                case DIAGONAL:
                    dst = ( _dense_matrix.col(0)).asDiagonal();
                    break;
                case DENSE:
                    dst = _dense_matrix;
                    break;
                default:
                    KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
            }
        }
        
        // ---- Resizing ----
        
        
        void conservativeResize(Index newWidth, Index newHeight) {
            switch(_type) {
                case IDENTITY:
                    if (newWidth != newHeight)
                        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize the identity matrix to a non squared matrix (%d x %d)", %newWidth %newHeight);
                    break;
                case DIAGONAL:
                    if (newWidth != newHeight)
                        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize the identity matrix to a non squared matrix (%d x %d)", %newWidth %newHeight);
                    _dense_matrix.conservativeResize(newHeight, 1);
                    break;
                case DENSE:
                    _dense_matrix.conservativeResize(newHeight, newWidth);
                    break;
                default:
                    KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
            }  
            
            _rows = newHeight;
            _cols = newWidth;
      }
        
        void resize(Index newHeight, Index newWidth) {
            switch(_type) {
                case IDENTITY:
                    if (newWidth != newHeight)
                        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize the identity matrix to a non squared matrix (%d x %d)", %newWidth %newHeight);
                    break;
                case DIAGONAL:
                    if (newWidth != newHeight)
                        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize the identity matrix to a non squared matrix (%d x %d)", %newWidth %newHeight);
                    _dense_matrix.resize(newHeight, 1);
                    break;
                case DENSE:
                    _dense_matrix.resize(newHeight, newWidth);
                    break;
                default:
                    KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
            }  
            
            _rows = newHeight;
            _cols = newWidth;
        }
        
        // ---- Blocks ----
        
        typedef AltBlock< Self > Block;
        typedef const AltBlock< AltMatrix<_Scalar> > ConstBlock;
        
        Block row(Index i) { return Block(*this, i, 0, 1, cols()); }
        ConstBlock row(Index i) const { return ConstBlock(const_cast<Self&>(*this), i, 0, 1, cols()); }
        
        Block col(Index j) { return Block(*this, 0, j, rows(), 1); }
        ConstBlock col(Index j) const { return ConstBlock(const_cast<Self&>(*this), 0, j, rows(), 1); }
        
        Block block(Index i, Index j, Index width, Index height) { return Block(*this, i, j, width, height); }
        ConstBlock block(Index i, Index j, Index width, Index height) const { return ConstBlock(const_cast<Self&>(*this), i, j, width, height); }
        
        const RowWise<AltMatrix> rowwise() const {
            return RowWise<AltMatrix>(const_cast<Self&>(*this));
        }
        
    private:
        
        template<typename Lhs, typename Rhs, int ProductOrder> friend class Eigen::AltDenseProduct;
        template<typename Lhs, typename Rhs, int ProductOrder> friend class Eigen::AltDiagonalProduct;
        
        // Derive class members from the dense matrix
        void configure_dense() {
            _type = DENSE;
            _rows = _dense_matrix.rows();
            _cols = _dense_matrix.cols();
        }
        
        
        AltMatrix(AltMatrixType type, Index rows, Index cols) :  _type(type), _rows(rows), _cols(cols) {
        }
        
        /// Type of the matrix
        AltMatrixType _type;
        
        /// Storage (used for DIAGONAL and DENSE types)
        DenseMatrix _dense_matrix;
        
        /// Cache for the number of rows
        Index _rows;
        
        /// Cache for the number of columns
        Index _cols;        
    };
    
    
    template<typename AltMatrix> class RowWise {
        AltMatrix &alt_matrix;
    public:
        typedef typename AltMatrix::Scalar Scalar;
        typedef typename Eigen::NumTraits< Scalar >::Real Real;
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RealVector;
        
        RowWise(AltMatrix &alt_matrix) : alt_matrix(alt_matrix) {
        }
        
        RealVector squaredNorm() const {
            switch(alt_matrix.type()) {
                case IDENTITY:
                    return RealVector::Ones(alt_matrix.rows());
                case DIAGONAL:
                    return alt_matrix.dense_matrix().rowwise().squaredNorm();
                case DENSE:
                    return alt_matrix.dense_matrix().rowwise().squaredNorm();
            }
            KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");  
        }
    };

    //! Block view of an AltMatrix
    template<typename AltMatrix> class AltBlock {
    public:       
        typedef typename AltMatrix::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        
        AltBlock(AltMatrix &alt_matrix, Index row, Index col, Index width, Index height) : 
        alt_matrix(alt_matrix), row(row), col(col), width(width), height(height),
        range(std::max(row, col), std::min(row+width, col+height) - std::max(row, col) + 1)
        {
        }
        
        Real squaredNorm() const {
            switch(alt_matrix.type()) {
                case IDENTITY:
                    return range.second > 0 ?  range.second : 0;
                case DIAGONAL:
                    return range.second > 0 ?  alt_matrix.dense_matrix().block(range.first, 0, range.second, 1).squaredNorm() : 0;
                case DENSE:
                    return alt_matrix.dense_matrix().block(row,col,width,height).squaredNorm();
            }
            KQP_THROW_EXCEPTION(assertion_exception, "Unknown AltMatrix type");
        }
        
        AltBlock & operator=(const AltBlock &/*b*/) {
            KQP_THROW_EXCEPTION(not_implemented_exception, "block equality altblocks");
        }
        
    private:
        AltMatrix &alt_matrix;
        Index row, col, width, height;
        std::pair<Index, Index> range;
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
            typedef Eigen::Transpose< kqp::AltMatrix<Scalar> > type;
        };
        template<typename Scalar> struct eval<Eigen::Transpose<const kqp::AltMatrix<Scalar> >, kqp::AltMatrixStorage> {
            typedef Eigen::Transpose< const kqp::AltMatrix<Scalar> > type;
        };
        
        
        /// KQP traits
        template<typename Scalar>
        struct kqp_traits< kqp::AltMatrix<Scalar> > {
            typedef typename kqp::AltMatrix<Scalar>::DenseMatrix::AdjointReturnType DenseType;  
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
        
        /// Traits for an AltMatrix
        template<typename _Scalar> 
        struct traits<kqp::AltMatrixBase<_Scalar> > {
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
    template<typename MatrixType> class TransposeImpl<MatrixType, kqp::AltMatrixStorage> 
    : public kqp::AltMatrixBase< Transpose<MatrixType> > {
    public:
        typedef TransposeImpl<MatrixType,kqp::AltMatrixStorage> Base;
        typedef MatrixType PlainObject;
        
        
        typedef const Eigen::Transpose<MatrixType> Object;
        typedef Eigen::Transpose<MatrixType> Nested;
        
        typedef typename PlainObject::DenseMatrix::AdjointReturnType DenseType;
        typedef typename PlainObject::DiagonalType DiagonalType;
        
        inline const PlainObject & matrix() const { return static_cast<Object&>(*this).nestedExpression(); }
        
        kqp::AltMatrixType type() const { return matrix().type(); }
        
        const DenseType asDense() const { return matrix().dense_matrix().transpose(); }
        const DiagonalType asDiagonal() const { return matrix().transpose().asDiagonal(); }
    };
    
    
    
    namespace internal {
        
        template<typename Derived>
        struct kqp_traits<const Derived> /*: kqp_traits<Derived> */{
            typedef double DenseType;
        };
        
        template<typename Derived>
        struct kqp_traits< Transpose< Derived > > {
            typedef typename Derived::DenseMatrix::AdjointReturnType DenseType;  
        };
        
        
    } // namespace internal
    
} // namespace Eigen


// --- Alt - Dense
namespace Eigen {
    
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
}

// ----
// ---- PRODUCT: Alt Matrix x Dense
// ----

namespace Eigen { 
    template<typename Lhs, typename Rhs, int ProductOrder>
    class AltDenseProduct;// : public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,Tr>, Lhs, Rhs> {};
    
    // --- Alt * Dense
    template<typename Lhs, typename Rhs>
    class AltDenseProduct<Lhs,Rhs,OnTheLeft> : internal::no_assignment_operator, public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,OnTheLeft>, Lhs, Rhs> 
    {
    public:
        EIGEN_PRODUCT_PUBLIC_INTERFACE(AltDenseProduct)
        typedef internal::traits<AltDenseProduct> Traits;
        
        
        AltDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {
        }
        
        template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar beta) const {
            switch(this->lhs().derived().type()) {
                case kqp::IDENTITY:
                    dest += beta * this->rhs();
                    break;
                case kqp::DIAGONAL:
                    dest += beta * (this->lhs().derived().asDense().col(0).asDiagonal() * this->rhs());
                    break;
                case kqp::DENSE:
                    dest += beta * this->lhs().derived().asDense() * this->rhs();
                    break;
                default:
                    abort();
            }
            
        }
    };
    template<class Derived,class OtherDerived>
    inline const AltDenseProduct<Derived,OtherDerived,OnTheLeft>
    operator*(const kqp::AltMatrixBase<Derived> &a, const Eigen::MatrixBase<OtherDerived> &b) {
        return AltDenseProduct<Derived,OtherDerived,OnTheLeft>(a.derived(), b.derived());
    }
    
    // --- Dense * Alt
    template<typename Lhs, typename Rhs>
    class AltDenseProduct<Lhs,Rhs,OnTheRight> : internal::no_assignment_operator, public ProductBase<AltDenseProduct<Lhs,Rhs,OnTheRight>, Lhs, Rhs>
    {
    public:
        typedef AltDenseProduct<Lhs,Rhs,OnTheRight> Self;
        EIGEN_PRODUCT_PUBLIC_INTERFACE(Self)
        typedef internal::traits<AltDenseProduct> Traits;
        
        AltDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {
        }
        
        template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar beta) const {
            switch(this->rhs().type()) {
                case kqp::IDENTITY:
                    dest += beta * this->lhs();
                    break;
                case kqp::DIAGONAL:
                    dest += beta * this->lhs() * this->rhs().asDense().col(0).asDiagonal();
                    break;
                case kqp::DENSE:
                    dest += beta * this->lhs() * this->rhs().asDense();
                    break;
                default:
                    abort();
            }
        }
    };
    
    
    template<class Derived,class OtherDerived>
    inline const AltDenseProduct<Derived,OtherDerived,OnTheRight>
    operator*(const Eigen::MatrixBase<Derived> &a, const kqp::AltMatrixBase<OtherDerived> &b) {
        return AltDenseProduct<Derived,OtherDerived,OnTheRight>(a.derived(), b.derived());
    }    
    
    
    
    
    
    
    
    
    
    namespace internal {        
        /// Traits for a product dense x alt
        template<typename Lhs, typename Rhs, int ProductOrder>
        struct traits<Eigen::AltDenseProduct<Lhs,Rhs,ProductOrder> >
        : traits<ProductBase<Eigen::AltDenseProduct<Lhs,Rhs,ProductOrder>, Lhs, Rhs> >
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



// ----
// ---- PRODUCT: Alt Matrix x Diagonal
// ----
namespace Eigen { 
    
    
    // AltMatrix times Diagonal (OnTheLeft)
    // Diagonal times AltMatrix (OnTheRight)
    template<typename Lhs, typename Rhs, int ProductOrder> class AltDiagonalProduct;
    
    
    template<typename Lhs, typename Rhs> class AltDiagonalProduct<Lhs,Rhs,OnTheLeft>: internal::no_assignment_operator, public kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,OnTheLeft> >
    {
    public:
        
        typedef typename Lhs::Nested LhsNested;
        typedef typename internal::remove_all<LhsNested>::type _LhsNested;
        
        typedef typename Rhs::Nested RhsNested;
        typedef typename internal::remove_all<RhsNested>::type _RhsNested;
        
        typedef kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,OnTheLeft> > Base;
        
        EIGEN_GENERIC_PUBLIC_INTERFACE(AltDiagonalProduct)
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
            switch(m_lhs.derived().type()) {
                case kqp::IDENTITY:
                    return kqp::AltMatrix<Scalar>(  m_rhs.diagonal() );
                case kqp::DIAGONAL: 
                    return kqp::AltMatrix<Scalar>( m_lhs.asDense().cwiseProduct(m_rhs.diagonal()).asDiagonal());
                case kqp::DENSE: {
                    return kqp::AltMatrix<Scalar>( m_lhs.dense_matrix() * m_rhs);  
                }
            }
            
            // Unknown type
            abort();
        }
        
        
        typedef typename Eigen::internal::kqp_traits<_LhsNested>::DenseType AltDenseType;
        typedef Eigen::DiagonalProduct<AltDenseType, Lhs, OnTheLeft> DenseType; 
        
        DenseType asDense() const {
            return m_lhs * m_rhs.derived().asDense();
        }
        
        
        kqp::AltMatrixType type() const {
            switch(m_lhs.derived().type()) {
                case kqp::IDENTITY:
                case kqp::DIAGONAL: 
                    return kqp::DIAGONAL;
                case kqp::DENSE: 
                    return kqp::DENSE;
            }
            abort();
        }
        
        
    protected:
        
        const LhsNested m_lhs;
        const RhsNested m_rhs;
    };
    
    template<typename Lhs, typename Rhs> class AltDiagonalProduct<Lhs,Rhs,OnTheRight> : internal::no_assignment_operator, public kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,OnTheRight> >
    {
    public:
        
        typedef typename Lhs::Nested LhsNested;
        typedef typename internal::remove_all<LhsNested>::type _LhsNested;
        
        typedef typename Rhs::Nested RhsNested;
        typedef typename internal::remove_all<RhsNested>::type _RhsNested;
        
        typedef kqp::AltMatrixBase< AltDiagonalProduct<Lhs,Rhs,OnTheLeft> > Base;
        
        EIGEN_GENERIC_PUBLIC_INTERFACE(AltDiagonalProduct);
        typedef internal::traits<AltDiagonalProduct> Traits;
        
        
        AltDiagonalProduct(const Lhs& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs) {
            eigen_assert(lhs.cols() == rhs.rows()
                         && "invalid matrix product"
                         && "if you wanted a coeff-wise or a dot product use the respective explicit functions");
        }
        
        inline Index rows() const { return m_lhs.rows(); }
        inline Index cols() const { return m_rhs.cols(); }
        
        template<typename Dest> inline void evalTo(Dest& dst) const {
            switch(m_rhs.derived().type()) {
                case kqp::IDENTITY:
                    dst = m_lhs.diagonal();
                    break;
                case kqp::DIAGONAL: 
                    dst = m_rhs.asDense().col(0).cwiseProduct(m_lhs.diagonal()).asDiagonal();
                    break;
                case kqp::DENSE: 
                    dst = m_lhs * m_rhs.asDense();  
                    break;
                default:
                    abort();
                    
            }
        }
        
        kqp::AltMatrix<Scalar> eval() const {            
            switch(m_rhs.derived().type()) {
                case kqp::IDENTITY:
                    return kqp::AltMatrix<Scalar>(  m_lhs.diagonal() );
                case kqp::DIAGONAL: 
                    return kqp::AltMatrix<Scalar>( m_rhs.asDense().col(0).cwiseProduct(m_lhs.diagonal()).asDiagonal());
                case kqp::DENSE: {
                    return kqp::AltMatrix<Scalar>( m_lhs * m_rhs.asDense());  
                }
            }
            
            // Unknown type
            abort();
        }
        
        
        typedef typename internal::remove_all<typename Eigen::internal::kqp_traits<_RhsNested>::DenseType>::type AltDenseType;
        typedef Eigen::DiagonalProduct<AltDenseType, Lhs, OnTheLeft> DenseType; 
        
        DenseType asDense() const {
            return m_lhs * m_rhs.derived().asDense();
        }
        
        
        kqp::AltMatrixType type() const {
            switch(m_rhs.derived().type()) {
                case kqp::IDENTITY:
                case kqp::DIAGONAL: 
                    return kqp::DIAGONAL;
                case kqp::DENSE: 
                    return kqp::DENSE;
            }
            abort();
        }
        
        
    protected:
        
        const LhsNested m_lhs;
        const RhsNested m_rhs;
    };
    
    
    
    
    // Alt * Diagonal
    template<class Derived,class OtherDerived>
    inline const AltDiagonalProduct<Derived,OtherDerived,OnTheLeft>
    operator*(const kqp::AltMatrixBase<Derived> &a, const Eigen::DiagonalBase<OtherDerived> &b) {
        return AltDiagonalProduct<Derived,OtherDerived,OnTheLeft>(a.derived(), b.derived());
    }
    
    // Diagonal * Alt
    template<class Derived,class OtherDerived>
    inline const AltDiagonalProduct<Derived,OtherDerived,OnTheRight>
    operator*(const Eigen::DiagonalBase<Derived> &a, const kqp::AltMatrixBase<OtherDerived> &b) {
        return AltDiagonalProduct<Derived,OtherDerived,OnTheRight>(a.derived(), b.derived());
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
        
        template<typename Lhs, typename Rhs, int ProductOrder>
        struct nested< AltDiagonalProduct<Lhs, Rhs, ProductOrder> > {
            typedef AltDiagonalProduct<Lhs, Rhs, ProductOrder> type;
        };
        
        
        template<typename Lhs, typename Rhs, int ProductOrder> struct eval<AltDiagonalProduct<Lhs, Rhs, ProductOrder> > {
            typedef typename scalar_product_traits<typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar>::ReturnType Scalar;
            typedef kqp::AltMatrix<Scalar> type;
        };
        
        
        /// Traits for a product dense x alt
        template<typename Lhs, typename Rhs, int ProductOrder>
        struct traits<Eigen::AltDiagonalProduct<Lhs,Rhs,ProductOrder> >
        : traits<ProductBase<Eigen::AltDiagonalProduct<Lhs,Rhs,ProductOrder>, Lhs, Rhs> >
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


