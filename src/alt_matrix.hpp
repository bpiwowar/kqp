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

namespace Eigen {
    template<typename Lhs, typename Rhs, bool Tr> class AltDenseProduct;
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
    
    enum AltMatrixType {
        DIAGONAL,
        DENSE
    };
    
    
    /**
     * This matrix can be either the Identity or a dense matrix.
     *
     * As this is known only at execution time, we have to create a new matrix type for Eigen.
     *
     */
    template<typename _Scalar, bool Tr = false> class AltMatrix : public AltMatrixBase< AltMatrix<_Scalar, Tr> > {
    public:
        typedef const AltMatrix& Nested;
        enum {
            Flags = 0x0,
            IsVectorAtCompileTime = 0,
            Options = 0
        };
        
        typedef _Scalar Scalar;
        typedef long Index;
        typedef AltMatrix<Scalar> PlainObject;
        
        
        enum Mode {
            COPY,
            SWAP,
            REFERENCE
        };
        
        template<class Derived>
        static AltMatrix<Scalar> copy_of(const Eigen::MatrixBase<Derived> &dense_matrix, Scalar alpha = (Scalar)1) {
            return AltMatrix<Scalar>(dense_matrix, alpha, COPY);               
        }
        
        template<class Derived>
        static AltMatrix<Scalar> reference_of(const Eigen::MatrixBase<Derived> &dense_matrix, Scalar alpha = (Scalar)1) {
            return AltMatrix<Scalar>(dense_matrix, alpha, REFERENCE);                           
        }
        
        template<class Derived>
        static AltMatrix<Scalar> swap_of(const Eigen::MatrixBase<Derived> &dense_matrix, Scalar alpha = (Scalar)1) {
            return AltMatrix<Scalar>(dense_matrix, alpha, SWAP);               
        }
        
        static AltMatrix<Scalar> Identity(Index size, Scalar alpha = (Scalar)1)  {
            return AltMatrix<Scalar>(DIAGONAL, size, size, alpha);
        }
        
        Index rows() const { return _rows; }
        Index cols() const { return _cols; }
        AltMatrixType type() const { return _type; }
        Scalar alpha() const { return _alpha; }
        
        AltMatrix<Scalar,!Tr> adjoint() const { 
            return AltMatrix<Scalar, !Tr>(*this);
        }
        
    protected:
        
        template<bool Tr2>
        explicit AltMatrix(const AltMatrix<Scalar, Tr2> &other) :
        dense_matrix_ptr(other.dense_matrix_ptr), _type(other._type), _rows(Tr == Tr2 ? other._rows : other._cols), 
        _cols(Tr == Tr2 ? other._cols : other._rows), _alpha(Eigen::internal::conj(other._alpha))
        {
        }
        
    private:
        AltMatrix& operator=(const AltMatrix &);
        template<typename Lhs, typename Rhs, bool Tr2> friend class Eigen::AltDenseProduct;
        friend class AltMatrix<Scalar, !Tr>;
        
        template<typename Scalar> friend const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & matrix_of(const AltMatrix<Scalar, Tr> &m);
        template<typename Scalar> friend const typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::AdjointReturnType matrix_of(const AltMatrix<Scalar, Tr> &m);
        
        template<class Derived>
        AltMatrix(const Eigen::MatrixBase<Derived> &_dense_matrix, Scalar alpha, Mode mode) 
        : _type(DENSE),  _rows(_dense_matrix.rows()), _cols(_dense_matrix.cols()), _alpha(alpha) {
            switch(mode) {
                case COPY: dense_matrix = _dense_matrix; dense_matrix_ptr = &dense_matrix; break;
                case REFERENCE: dense_matrix_ptr = &_dense_matrix.derived(); break; 
                case SWAP: dense_matrix.swap(_dense_matrix); dense_matrix_ptr = &dense_matrix; break;
            }
        }
        
        AltMatrix(AltMatrixType type, Index rows, Index cols, Scalar alpha) :  _type(type), _rows(rows), _cols(cols), _alpha(alpha) {
        }
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense_matrix;
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> *dense_matrix_ptr;
        
        AltMatrixType _type;
        Index _rows;
        Index _cols;
        
        Scalar _alpha;
    };
    
    template<typename Scalar>
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&  matrix_of(const kqp::AltMatrix<Scalar, false> &m) {
        return *m.dense_matrix_ptr;
    }
    
    template<typename Scalar>
    const typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::AdjointReturnType matrix_of(const kqp::AltMatrix<Scalar, true> &m) {
        return m.dense_matrix_ptr->adjoint();
    }
    
}

namespace Eigen {
    
    template<typename Lhs, typename Rhs, bool Tr>
    class AltDenseProduct : public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,Tr>, Lhs, Rhs> {};
    
    template<typename Lhs, typename Rhs>
    class AltDenseProduct<Lhs,Rhs,false> : public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,false>, Lhs, Rhs> 
    {
    public:
        typedef ProductBase<AltDenseProduct, Lhs, Rhs> Base;
        EIGEN_DENSE_PUBLIC_INTERFACE(AltDenseProduct)
        typedef internal::traits<AltDenseProduct> Traits;
        
    //private:
        
        typedef typename Traits::LhsNested LhsNested;
        typedef typename Traits::RhsNested RhsNested;
        typedef typename Traits::_LhsNested _LhsNested;
        typedef typename Traits::_RhsNested _RhsNested;
        
    public:
        
        AltDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {
        }
        
        template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar beta) const {
            Scalar x = (this->lhs().alpha() * beta);
            switch(this->lhs().type()) {
                case kqp::DIAGONAL:
                    dest += x * this->rhs();
                    break;
                case kqp::DENSE:
                    dest += x * kqp::matrix_of(this->lhs()) * this->rhs();
                    break;
                default:
                    abort();
            }
            
        }
    private:
        AltDenseProduct operator=(const AltDenseProduct &);
    };
    
    template<typename Lhs, typename Rhs>
    class AltDenseProduct<Lhs,Rhs,true> : public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,true>, Lhs, Rhs>
    {
    public:
        typedef AltDenseProduct<Lhs,Rhs,true> Self;
        EIGEN_PRODUCT_PUBLIC_INTERFACE(Self)
        
        AltDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}
        
        template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar beta) const {
            Scalar x = (this->rhs().alpha() * beta);
            switch(this->rhs().type()) {
                case kqp::DIAGONAL:
                    dest += x * this->lhs();
                    break;
                case kqp::DENSE:
                    dest += x * this->lhs() * kqp::matrix_of(this->rhs());
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
    
    namespace internal {
        template<> struct promote_storage_type<Dense,kqp::AltMatrixStorage>
        { typedef Dense ret; };
        template<> struct promote_storage_type<kqp::AltMatrixStorage, Dense>
        { typedef Dense ret; };
        
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
        
        
        template<typename _Scalar, bool Tr> 
        struct traits<kqp::AltMatrix<_Scalar, Tr> > {
            typedef _Scalar Scalar;
            typedef kqp::AltMatrixStorage StorageKind;
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
    } // end namespace internal
    
    
}

#endif


