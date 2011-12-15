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

#include "kqp.hpp"
namespace kqp {
    
    struct AltMatrixStorage {};
    
    template<typename Derived> class AltMatrixBase  : public Eigen::EigenBase< Derived > {
        
    };
    
    /**
     * This matrix can be either the Identity or a dense matrix.
     *
     * As this is known only at execution time, we have to create a new matrix type for Eigen.
     *
     */
    template<typename _Scalar> class AltMatrix : public AltMatrixBase< AltMatrix<_Scalar> > {
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
        
        enum Type {
            DIAGONAL,
            DENSE
        };
        
        enum Mode {
            COPY,
            SWAP,
            REFERENCE
        };
        
        template<class Derived>
        static AltMatrix<Scalar> copy_of(const MatrixBase<Derived> &dense_matrix, Scalar alpha = (Scalar)1) {
            return AltMatrix<Scalar>(dense_matrix, alpha, COPY);               
        }
        
        template<class Derived>
        static AltMatrix<Scalar> reference_of(const MatrixBase<Derived> &dense_matrix, Scalar alpha = (Scalar)1) {
            return AltMatrix<Scalar>(dense_matrix, alpha, REFERENCE);                           
        }
        
        template<class Derived>
        static AltMatrix<Scalar> swap_of(const MatrixBase<Derived> &dense_matrix, Scalar alpha = (Scalar)1) {
            return AltMatrix<Scalar>(dense_matrix, alpha, SWAP);               
        }
        
        static AltMatrix<Scalar> Identity(Index size, Scalar alpha = (Scalar)1)  {
            return AltMatrix<Scalar>(DIAGONAL, size, size, alpha);
        }
        
        Index rows() const { return _rows; }
        Index cols() const { return _cols; }
        Type type() const { return _type; }
        Scalar alpha() const { return _alpha; }
        
    private:
        template<typename Lhs, typename Rhs, bool Tr> friend class AltDenseProduct;
        
        template<class Derived>
        AltMatrix(const MatrixBase<Derived> &_dense_matrix, Scalar alpha, Mode mode) 
        : _type(DENSE),  _rows(_dense_matrix.rows()), _cols(_dense_matrix.cols()), _alpha(alpha) {
            switch(mode) {
                case COPY: dense_matrix = _dense_matrix; dense_matrix_ptr = &dense_matrix; break;
                case REFERENCE: dense_matrix_ptr = &dense_matrix; break; 
                case SWAP: dense_matrix.swap(_dense_matrix); dense_matrix_ptr = &dense_matrix; break;
            }
        }
        
        AltMatrix(Type type, Index rows, Index cols, Scalar alpha) :  _type(type), _rows(rows), _cols(cols), _alpha(alpha) {
        }
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense_matrix;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> *dense_matrix_ptr;
        
        Type _type;
        Index _rows;
        Index _cols;
        
        Scalar _alpha;
        
        
    };
    
    template<typename Lhs, typename Rhs, bool Tr>
    class AltDenseProduct : public Eigen::ProductBase<AltDenseProduct<Lhs,Rhs,Tr>, Lhs, Rhs> {
    public:
        EIGEN_PRODUCT_PUBLIC_INTERFACE(AltDenseProduct)
        
        AltDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}
        
        template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar beta) const {
            Scalar x = (this->lhs().alpha() * beta);
            switch(this->lhs().type()) {
                case Lhs::DIAGONAL:
                    dest += x * this->rhs();
                    break;
                case Lhs::DENSE:
                    dest += x * *this->lhs().dense_matrix_ptr * this->rhs();
                    break;
                default:
                    abort();
            }
            
        }
    };
    
    template<typename Lhs, typename Rhs> struct AltDenseProductReturnType
    {
        typedef AltDenseProduct<Lhs,Rhs,false> Type;
    };
    
    
    template<class Derived,class OtherDerived>
    inline const typename AltDenseProductReturnType<Derived,OtherDerived>::Type
    operator*(const AltMatrixBase<Derived> &a, const Eigen::MatrixBase<OtherDerived> &b) {
        return typename AltDenseProductReturnType<Derived,OtherDerived>::Type(a.derived(), b.derived());
    }
}


namespace Eigen {
    
    
    namespace internal {
        template<> struct promote_storage_type<Dense,kqp::AltMatrixStorage>
        { typedef Dense ret; };
        template<> struct promote_storage_type<kqp::AltMatrixStorage, Dense>
        { typedef Dense ret; };
        
        template<typename Lhs, typename Rhs, bool Tr>
        struct traits<kqp::AltDenseProduct<Lhs,Rhs,Tr> >
        : traits<ProductBase<kqp::AltDenseProduct<Lhs,Rhs,Tr>, Lhs, Rhs> >
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
                Flags = Tr ? RowMajorBit : 0,
                RowsAtCompileTime    = Tr ? int(traits<Rhs>::RowsAtCompileTime)     : int(traits<Lhs>::RowsAtCompileTime),
                ColsAtCompileTime    = Tr ? int(traits<Lhs>::ColsAtCompileTime)     : int(traits<Rhs>::ColsAtCompileTime),
                MaxRowsAtCompileTime = Tr ? int(traits<Rhs>::MaxRowsAtCompileTime)  : int(traits<Lhs>::MaxRowsAtCompileTime),
                MaxColsAtCompileTime = Tr ? int(traits<Lhs>::MaxColsAtCompileTime)  : int(traits<Rhs>::MaxColsAtCompileTime),
                
                LhsCoeffReadCost = traits<_LhsNested>::CoeffReadCost,
                RhsCoeffReadCost = traits<_RhsNested>::CoeffReadCost,
                
            };
        };
        
        
        template<typename _Scalar> 
        struct traits<kqp::AltMatrix<_Scalar> > {
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


