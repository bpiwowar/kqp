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

#include <kqp/kqp.hpp>
#include <Eigen/Core>

namespace kqp {
    
    // Should use template template aliases?
    
    
    // Forward declarations
    template<typename T1, typename T2> class AltMatrix;
    template<typename _AltMatrix> class AltBlock;
    template<typename _AltMatrix> class RowWise;
    template<typename Derived> struct scalar;
    
    
    // Returns the associated scalar
    template<typename Derived> struct scalar { typedef typename Eigen::internal::traits<Derived>::Scalar type; };
    template<> struct scalar<double> { typedef double type; };
    template<> struct scalar<float> { typedef float type; };
    template<> struct scalar<std::complex<double>> { typedef std::complex<double> type; };
    template<> struct scalar<std::complex<float>> { typedef std::complex<float> type; };
    
    
    // ---- Predefined Alt-based matrices
    
    //! Diagonal or Identity matrix
    template<typename Scalar> struct AltDiagonal {
        typedef Eigen::DiagonalWrapper<Eigen::Matrix<Scalar,Eigen::Dynamic,1>> Diagonal;
        typedef typename Eigen::MatrixBase<Eigen::Matrix<Scalar, Eigen::Dynamic,Eigen::Dynamic> >::IdentityReturnType Identity;
        
        typedef AltMatrix<Diagonal,Identity> type;
    };
    
    template<typename Scalar> struct AltVector {
        typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1>  Vector;
        typedef typename Eigen::Matrix<Scalar,Eigen::Dynamic,1>::ConstantReturnType ConstantReturnType;
        
        typedef AltMatrix<Vector, ConstantReturnType> type;
    };
    
    
    //! Dense or Identity matrix
    template<typename Scalar> struct AltDense {
        typedef AltMatrix< Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> , typename Eigen::MatrixBase<Eigen::Matrix<Scalar, Eigen::Dynamic,Eigen::Dynamic> >::IdentityReturnType > type;
        
        static  inline type Identity(Index n) { 
            return Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(n,n); 
        }
    };    
    
    //! Storage type for AltMatrix
    struct AltMatrixStorage {};
    
    template<typename Operator, typename Derived> class AltCwiseUnaryOp;
    template<typename Derived> class AltAsDiagonal;
    
    //! Base class for any AltMatrix expression
    template<typename Derived> 
    class AltMatrixBase : public Eigen::EigenBase<Derived> {
    public:
        typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
        
        AltCwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, const Derived> cwiseSqrt() const {
            return this->derived(); 
        }
        
        AltCwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, const Derived>  cwiseAbs() const {
            return this->derived();
        }
        
        AltCwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, const Derived>  cwiseAbs2() const {
            return this->derived();
        }
        
        AltCwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, const Derived> cwiseInverse() const {
            return this->derived();
        }
        
        AltAsDiagonal<const Derived> asDiagonal() const {
            return this->derived();
        }
        
    };
    
    
    // --- As diagonal
    template<typename XprType> class AltAsDiagonal : Eigen::internal::no_assignment_operator, public AltMatrixBase<AltAsDiagonal<XprType>> {
    protected:
        const typename XprType::Nested m_xpr;
    public:
        
        inline AltAsDiagonal(const XprType& xpr) : m_xpr(xpr) {
        }
        
        EIGEN_STRONG_INLINE Index rows() const { return m_xpr.rows(); }
        EIGEN_STRONG_INLINE Index cols() const { return m_xpr.rows(); }
        
        bool isT1() const { return m_xpr.derived().isT1(); }
        
        auto t1() const -> decltype(m_xpr.derived().t1().asDiagonal()) { return m_xpr.derived().t1().asDiagonal(); }
        auto t2() const -> decltype(m_xpr.derived().t2().asDiagonal()) { return m_xpr.derived().t2().asDiagonal(); }
        
    };
    
    // --- Unary operator
    template<typename UnaryOp, typename XprType> class AltCwiseUnaryOp :
    Eigen::internal::no_assignment_operator, public AltMatrixBase<AltCwiseUnaryOp<UnaryOp,XprType>> {
    protected:
        const typename XprType::Nested m_xpr;
        const UnaryOp m_functor;
    public:
        typedef AltCwiseUnaryOp<UnaryOp,XprType> Nested;
        inline AltCwiseUnaryOp(const XprType& xpr, const UnaryOp& func = UnaryOp())
        : m_xpr(xpr), m_functor(func) {
        }
        
        EIGEN_STRONG_INLINE Index rows() const { return m_xpr.rows(); }
        EIGEN_STRONG_INLINE Index cols() const { return m_xpr.cols(); }
        
        bool isT1() const { return m_xpr.derived().isT1(); }
        
        typedef typename Eigen::internal::remove_all<decltype(m_xpr.derived().t1())>::type T1;
        Eigen::CwiseUnaryOp<UnaryOp,T1> t1() const  { 
            return Eigen::CwiseUnaryOp<UnaryOp,T1>(m_xpr.derived().t1(), m_functor); 
        }
        
        typedef typename Eigen::internal::remove_all<decltype(m_xpr.derived().t2())>::type T2;
        Eigen::CwiseUnaryOp<UnaryOp,T2> t2() const  { 
            return Eigen::CwiseUnaryOp<UnaryOp,T2>(m_xpr.derived().t2(), m_functor); 
        }
        
        template<typename Dest> inline void evalTo(Dest& dst) const {
            if (isT1()) dst = t1(); else dst = t2();
        }
    };
    
    
    
    // --- Helper functions
    template<typename Derived>
    Eigen::Transpose<const Derived> transpose(const Eigen::MatrixBase<Derived>& x) { return x.transpose(); }
    
    template<typename Derived> 
    Eigen::DiagonalWrapper<const Derived> transpose(const Eigen::DiagonalWrapper<const Derived> & x)  { return x; }
    
    template<typename Derived>
    auto blockSquaredNorm(const Eigen::MatrixBase<Derived>& x, Index row, Index col, Index rowSize, Index colSize) -> decltype(x.squaredNorm()) { 
        return x.block(row,col,rowSize,colSize).squaredNorm(); 
    }
    
    template<typename Derived> 
    auto blockSquaredNorm(const Eigen::DiagonalWrapper<Derived> & x, Index row, Index col, Index rowSize, Index colSize) -> decltype(x.diagonal().squaredNorm())  { 
        return x.diagonal().segment(std::max(row,col), std::min(row+rowSize, col+colSize)).squaredNorm(); 
    }
    
    
    template<typename Derived>
    auto squaredNorm(const Eigen::MatrixBase<Derived>& x) -> decltype(x.derived().squaredNorm()) { return x.derived().squaredNorm(); }
    
    template<typename Derived> 
    auto squaredNorm(const Eigen::DiagonalWrapper<Derived> & x) -> decltype(x.diagonal().squaredNorm())  { return x.diagonal().squaredNorm(); }
    
    // ---- Transpose
    
    
    
    //! Transpose
    template<typename Derived>
    struct Transpose : public AltMatrixBase< Transpose<Derived> > {
        typename Eigen::internal::ref_selector<Derived>::type nested;
    public:
        
        Transpose(Derived &nested) : nested(nested) {}
        
        Index rows() const { return nested.cols(); }
        Index cols() const { return nested.rows(); }
        
        bool isT1() const { return nested.isT1(); }
        
        typedef decltype(kqp::transpose(nested.t1())) T1;
        typedef decltype(kqp::transpose(nested.t2())) T2;
        
        inline T1 t1() const  { return transpose(nested.t1()); }
        inline T2 t2() const  { return transpose(nested.t2()); }
        
        
        void printExpression(std::ostream &out) const {
            out << "Transpose(";
            nested.printExpression(out);
            out << ")";
        }
        
        
    };
    
    
    
    template <typename Derived>
    void printExpression(std::ostream &out, const kqp::AltMatrixBase<Derived> &o) {
        (static_cast<const Derived&>(o)).printExpression(out);
    }
    
    template <typename Derived>
    void printExpression(std::ostream &out, const Eigen::EigenBase<Derived> &x) {
        out << KQP_DEMANGLE(x) << "[" << x.rows() << " x " << x.cols() << "]";
    }
    
    template <typename Derived>
    void printExpression(std::ostream &out, const Eigen::Transpose<Derived> &x) {
        out << "transpose(";
        printExpression(out, static_cast<const Derived&>(x.derived()));
        out << ")";
    }
    
    template <typename Derived>
    void printExpression(std::ostream &out, const Eigen::DiagonalWrapper<Derived> &x) {
        out << "diag[" << x.rows() << " x " << x.cols() << "]";
    }
    
    template <typename Scalar, int a, int b>
    void printExpression(std::ostream &out, const Eigen::Matrix<Scalar, a, b> &x) {
        out << "dense/" << &x << "[" << x.rows() << " x " << x.cols() << "]";
    }
    
    template <typename Derived, typename OtherDerived>
    void printExpression(std::ostream &out, const Eigen::DiagonalProduct<Derived, OtherDerived, Eigen::OnTheLeft> &x) {
        out << "dproduct[";
        kqp::printExpression(out, x.m_diagonal);
        out << " x ";
        kqp::printExpression(out, x.m_matrix);
        out << "]";
    }
    template <typename Derived, typename OtherDerived>
    void printExpression(std::ostream &out, const Eigen::DiagonalProduct<Derived, OtherDerived, Eigen::OnTheRight> &x) {
        out << "dproduct[";
        kqp::printExpression(out, x.m_matrix);
        out << " x ";
        kqp::printExpression(out, x.m_diagonal);
        out << "]";
    }
    
    
    template <typename Derived, typename OtherDerived>
    void printExpression(std::ostream &out, const Eigen::GeneralProduct<Derived, OtherDerived> &x) {
        out << "product[";
        kqp::printExpression(out, x.m_lhs);
        out << " x ";
        kqp::printExpression(out, x.m_rhs);
        out << "]";
    }
    
    
    
    
    //! Default storage type for AltMatrix nested types
    template<typename Derived>
    struct storage {
        typedef Derived & ReturnType; 
        typedef const Derived & ConstReturnType; 
        
        Derived m_value;
        
        storage() {}
        storage(const Derived &value) : m_value(value) {}
        ConstReturnType get() const { return m_value; }
        ReturnType get() { return m_value; }
        
        void swap(Derived &value) { m_value.swap(value); }
        Index rows() const { return m_value.rows(); }
        Index cols() const { return m_value.cols(); }
        
        void resize(Index rows, Index cols) {
            m_value.resize(rows,cols);
        }
        
        void conservativeResize(Index rows, Index cols) {
            m_value.conservativeResize(rows,cols);
        }      
        typename Eigen::internal::traits<Derived>::Scalar operator()(Index i, Index j) const {
            return m_value(i,j);
        }
        
        template<typename CwiseUnaryOp>
        void unaryExprInPlace(const CwiseUnaryOp &op) {
            m_value = m_value.unaryExpr(op);
        }
        
        typedef typename Eigen::NumTraits<typename Eigen::internal::traits<Derived>::Scalar>::Real Real;
        Real squaredNorm() const { return m_value.squaredNorm(); }
        
    };
    
    //! Storage for a constant matrix
    template<typename Scalar, typename Derived>
    struct storage< typename Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Derived> > {
        typedef typename Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Derived> ReturnType;
        typedef const ReturnType ConstReturnType;
        
        Scalar m_value;
        Index m_rows;
        Index m_cols;
        
        storage() : m_value(0), m_rows(0), m_cols(0) {}
        storage(const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>,Derived> &value) 
        : m_value(value.coeff(0,0)), m_rows(value.rows()), m_cols(value.cols()) {}
        
        ReturnType get() const { return ReturnType(m_rows, m_cols, m_value); }
        
        void swap(ReturnType &) { KQP_THROW_EXCEPTION(illegal_argument_exception, "Cannot swap a constant matrix"); }
        Index rows() const { return m_rows; }
        Index cols() const { return m_cols; }
        
        void resize(Index rows, Index cols) {
            m_rows = rows;
            m_cols = cols;
        }
        
        void conservativeResize(Index rows, Index cols) {
            if (rows > m_rows || cols > m_cols) 
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize to a non smaller size (%d x %d to %d x %d)",
                                      %m_rows%m_cols%rows%cols);
            this->resize(rows,cols);
        }
        
        Scalar operator()(Index, Index) const {
            return m_value;
        }
        
        template<typename CwiseUnaryOp>
        void unaryExprInPlace(const CwiseUnaryOp &op) {
            m_value = op(m_value);
        }
        
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        Real squaredNorm() const { return std::abs(m_value)*std::abs(m_value) * (Real)m_rows * (Real)m_cols; }

        
    };
    
    //! Storage for a diagonal wrapper
    template<typename Derived>
    struct storage< Eigen::DiagonalWrapper<Derived> > {
        typedef const Eigen::DiagonalWrapper<const Derived> ConstReturnType;
        typedef const Eigen::DiagonalWrapper<Derived> ReturnType;
        
        Derived m_value;
        
        storage() {}
        storage(const Eigen::DiagonalWrapper<Derived> &value) : m_value(value.diagonal()) {}
        ConstReturnType get() const { return m_value.asDiagonal(); }
        ReturnType get() { return static_cast<ReturnType>(m_value.asDiagonal()); }
        
        void swap(ReturnType &value) { m_value.swap(value); }
        Index rows() const { return m_value.rows(); }
        Index cols() const { return m_value.rows(); }
        
        void resize(Index rows, Index cols) {
            if (rows != cols) KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize to a non diagonal size (%d x%d)", %rows%cols);
            m_value.resize(rows,cols);
        }
        
        void conservativeResize(Index rows, Index cols) {
            if (rows != cols) KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize to a non diagonal size (%d x%d)", %rows%cols);
            m_value.conservativeResize(rows,cols);
        }
        
        typename Eigen::internal::traits<Derived>::Scalar operator()(Index i, Index j) const {
            return i == j ? m_value(i) : 0;
        }
        
        template<typename CwiseUnaryOp>
        void unaryExprInPlace(const CwiseUnaryOp &op) {
            m_value = m_value.unaryExpr(op);
        }
        
        typedef typename Eigen::NumTraits<typename Eigen::internal::traits<Derived>::Scalar>::Real Real;
        Real squaredNorm() const { return m_value.squaredNorm(); }

    };
    
    //! Storage for the identity
    template<typename Scalar>
    struct storage< Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >  > {
        Index m_rows, m_cols; 
        
        typedef Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > Type;
        typedef Type ReturnType;
        typedef const Type ConstReturnType;
        
        storage() {}
        storage(const Type &value) : m_rows(value.rows()), m_cols(value.cols()) {}
        ReturnType get() const { return  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(m_rows, m_cols); }
        
        void swap(Type &) { 
            KQP_THROW_EXCEPTION(not_implemented_exception, "Not sure what to do");
        }
        
        
        void resize(Index rows, Index cols) {
            m_rows = rows;
            m_cols = cols;
        }
        
        void conservativeResize(Index rows, Index cols) {
            resize(rows, cols);
        }
        
        Index rows() const { return m_rows; }
        Index cols() const { return m_cols; }
        
        Scalar operator()(Index i, Index j) const {
            return i == j ? 1 : 0;
        }
        
        template<typename CwiseUnaryOp>
        void unaryExprInPlace(const CwiseUnaryOp &) {
            KQP_THROW_EXCEPTION(illegal_operation_exception, "Cannot modify the identity");
        }
        
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        Real squaredNorm() const { return (Real)m_rows * (Real)m_cols; }

    };
    
    
    
    
    //! Alt Matrix
    template<typename T1, typename T2> 
    class AltMatrix : public AltMatrixBase< AltMatrix<T1, T2> > {
        storage<T1> m_t1;
        storage<T2> m_t2;
        
        bool m_isT1;
        
    public:
        typedef typename Eigen::internal::traits<T1>::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        typedef const AltMatrix& Nested;
        
        AltMatrix() : m_isT1(true) {}
        AltMatrix(const T1 &t1) : m_t1(t1), m_t2(), m_isT1(true) {}
        AltMatrix(const T2 &t2) : m_t1(), m_t2(t2), m_isT1(false) {}
        
        
        Index rows() const { return m_isT1 ? m_t1.rows() : m_t2.rows(); }
        Index cols() const { return m_isT1 ? m_t1.cols() : m_t2.cols(); }
        
        Real squaredNorm() const { return m_isT1 ? m_t1.squaredNorm() : m_t2.squaredNorm(); }
        Scalar trace() const { return m_isT1 ? m_t1.trace() : m_t2.trace(); }
        
        template<typename CwiseUnaryOp>
        void unaryExprInPlace(const CwiseUnaryOp &op) {
            if (isT1()) m_t1.unaryExprInPlace(op);
            else        m_t2.unaryExprInPlace(op);
        }
        
        bool isT1() const { return m_isT1; }
        
        inline typename storage<T1>::ConstReturnType t1() const { return m_t1.get(); }
        inline typename storage<T2>::ConstReturnType t2() const { return m_t2.get(); }
        
        inline typename storage<T1>::ReturnType t1() { return m_t1.get(); }
        inline typename storage<T2>::ReturnType t2() { return m_t2.get(); }
        
        
        void swap(T1 &t1) { m_isT1 = true; m_t1.swap(t1); }
        void swap(T2 &t2) { m_isT1 = true; m_t2.swap(t2); }
        
        //! Returns the adjoint
        Transpose<AltMatrix> transpose() const { return Transpose<AltMatrix>(const_cast<AltMatrix&>(*this)); }
        
        Scalar operator()(Index i, Index j) const {
            return m_isT1 ? m_t1(i,j) : m_t2(i,j);
        }
        
        inline Scalar coeff(Index i, Index j) const {
            return operator()(i,j);
        }
        
        template<typename Dest> inline void evalTo(Dest& dst) const {
            if (m_isT1)
                dst = m_t1.get();
            else 
                dst = m_t2.get();
        }
        
        
        void printExpression(std::ostream &out) const {
            out << "AltMatrix/" << this << "[";
            if (m_isT1) kqp::printExpression(out, m_t1.get()); else kqp::printExpression(out, m_t2.get());
            out << "]";
        }
        
        // ---- Resizing ----
        
        
        void conservativeResize(Index rows, Index cols) {
            if (m_isT1)
                m_t1.conservativeResize(rows, cols);
            else
                m_t2.conservativeResize(rows, cols);
        }
        
        void resize(Index rows, Index cols) {
            if (m_isT1)
                m_t1.resize(rows, cols);
            else
                m_t2.resize(rows, cols);
        }
        
        // ---- Blocks ----
        
        typedef AltMatrix<T1,T2> Self;
        typedef AltBlock< Self > Block;
        typedef const AltBlock< Self > ConstBlock;
        
        Block row(Index i) { return Block(*this, i, 0, 1, cols()); }
        ConstBlock row(Index i) const { return ConstBlock(const_cast<Self&>(*this), i, 0, 1, cols()); }
        
        Block col(Index j) { return Block(*this, 0, j, rows(), 1); }
        ConstBlock col(Index j) const { return ConstBlock(const_cast<Self&>(*this), 0, j, rows(), 1); }
        
        Block block(Index i, Index j, Index height, Index width) { return Block(*this, i, j, height, width); }
        ConstBlock block(Index i, Index j, Index height, Index width) const { return ConstBlock(const_cast<Self&>(*this), i, j, width, height); }
        
        const RowWise<AltMatrix> rowwise() const {
            return RowWise<AltMatrix>(const_cast<Self&>(*this));
        }
        
    };
    
    
    template<typename T1, typename T2>
    std::ostream &operator<<(std::ostream &out, const AltMatrix<T1,T2> &alt_matrix) {
        return out << "Alt/" << (alt_matrix.isT1() ? kqp::demangle(typeid(T1)) : kqp::demangle(typeid(T2)));
    }
    
    
    template <typename Derived> auto rowwise(const Derived &x) -> decltype(x.rowwise()) {
        return x.rowwise();
    }
    
    template <typename Derived> auto rowwise(const Eigen::DiagonalWrapper<Derived> &x) -> decltype(x.diagonal().rowwise()) {
        return x.diagonal().rowwise();
    }
    
    //! Row wise view of an Alt matrix
    template<typename AltMatrix> class RowWise {
        AltMatrix &alt_matrix;
    public:
        typedef typename AltMatrix::Scalar Scalar;
        typedef typename Eigen::NumTraits< Scalar >::Real Real;
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RealVector;
        
        RowWise(AltMatrix &alt_matrix) : alt_matrix(alt_matrix) {
        }
        
        RealVector squaredNorm() const {
            if (alt_matrix.isT1()) return rowwise(alt_matrix.t1()).squaredNorm();
            else return rowwise(alt_matrix.t2()).squaredNorm();
        }
    };
    
    //! Block view of an AltMatrix
    template<typename AltMatrix> class AltBlock  {
    public:
        typedef typename AltMatrix::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        
        AltBlock(AltMatrix &alt_matrix, Index row, Index col, Index height, Index width) :
        alt_matrix(alt_matrix), row(row), col(col), height(height), width(width),
        range(std::max(row, col), std::min(row+width, col+height) - std::max(row, col) + 1)
        {
        }
        
        Real squaredNorm() const {
            if (alt_matrix.isT1())
                return kqp::blockSquaredNorm(alt_matrix.t1(),row,col,height,width);
            else
                return kqp::blockSquaredNorm(alt_matrix.t2(),row,col,height,width);
        }
        
        
        // Assignement 
        
        
        template<typename Op, typename Derived, typename OtherDerived>
        void assign(const Eigen::CwiseNullaryOp<Op, Derived> &, const OtherDerived &, Index , Index ) {
            KQP_THROW_EXCEPTION(not_implemented_exception, "Cannot assign a constant matrix to anything");
        }
        
        template<typename Derived, int Rows, int Cols, typename OtherDerived>
        void assign(Eigen::Matrix<Derived, Rows, Cols> &mTo, const Eigen::MatrixBase<OtherDerived> &mFrom, Index fromRow, Index fromCol) {
            mTo.block(row,col,height,width) = mFrom.derived().block(fromRow, fromCol, height, width);
        }
        
        
        template<typename Derived>
        AltBlock<AltMatrix>& assignTo(const AltBlock<Derived> &m) {
            if (m.height != this->height || m.width != this->width)
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Block sizes differ in assignement (%d x %d != %d x %d)", %height %width %m.height %m.width);
            if (alt_matrix.isT1()) {
                if (m.alt_matrix.isT1()) this->assign(alt_matrix.t1(), m.alt_matrix.t1(), m.row, m.col);
                else this->assign(alt_matrix.t1(), m.alt_matrix.t2(), m.row, m.col);
            } else {
                if (m.alt_matrix.isT1()) this->assign(alt_matrix.t2(), m.alt_matrix.t1(), m.row, m.col);
                else this->assign(alt_matrix.t2(), m.alt_matrix.t2(), m.row, m.col);                
            }
            
            return *this;
        }
        
        
    private:
        AltMatrix &alt_matrix;
        Index row, col, height, width;
        std::pair<Index, Index> range;
    };
    
    
    template <typename Derived, typename OtherDerived>
    void copy(const Eigen::MatrixBase<Derived> &from, const Eigen::MatrixBase<OtherDerived> &to) {
        const_cast<Eigen::MatrixBase<OtherDerived>&>(to).derived() = from;
    }
    
    
    template <typename Derived, typename OtherDerived>
    void copy(const AltBlock<Derived> &from, const AltBlock<OtherDerived> &to) {
        const_cast<AltBlock<OtherDerived>&>(to).template assignTo<Derived>(from);
    }
    
    
    
    // --- Lazy evaluation
    template<typename Derived> 
    struct NoAlias {
        Derived &m;
        
        NoAlias(Derived &m) : m(m) {}
        
        template<typename OtherDerived>
        Derived &operator=(const AltMatrixBase<OtherDerived> &expr) {
            expr.derived().lazyAssign(m);
            return m;
        }
        
        template<typename OtherDerived>
        Derived &operator=(const Eigen::MatrixBase<OtherDerived> &expr) {
            return m.noalias() = expr.derived();
        }
        
    };
    
    template<typename Derived>
    NoAlias<Derived> noalias(Eigen::MatrixBase<Derived> &m) {
        return NoAlias<Derived>(m.derived());
    }
    
    
    // ---
    // --- Matrix multiplication
    // ----
    
    
    // --- Forward declarations
    
    template<typename Lhs, typename Rhs, int ProductOrder> class AltEigenProduct;
    
    //! Defines the expression type of a product
    template<typename Lhs, typename Rhs, int Side, bool isT1> struct ProductType;
    
    
    //! Defines an expression type when the Alt matrix is on the left
    template<typename _Lhs, typename _Rhs, bool isT1> struct ProductType<_Lhs, _Rhs, Eigen::OnTheLeft, isT1> {
        const _Lhs &_lhs;
        typedef decltype(_lhs.t1()) LhsT1;
        typedef decltype(_lhs.t2()) LhsT2;
        typedef typename Eigen::internal::conditional<isT1, LhsT1, LhsT2>::type Lhs;
        const Lhs &lhs;
        
        typedef _Rhs Rhs;        
        const Rhs &rhs;
        
        typedef decltype(lhs * rhs) Type;
    };
    
    //! Defines an expression type when the Alt matrix is on the right
    template<typename _Lhs, typename _Rhs, bool isT1> struct ProductType<_Lhs, _Rhs, Eigen::OnTheRight, isT1> {
        typedef _Lhs Lhs;
        const Lhs &lhs;
        
        const _Rhs &_rhs;
        typedef decltype(_rhs.t1()) RhsT1;
        typedef decltype(_rhs.t2()) RhsT2;
        typedef typename Eigen::internal::conditional<isT1, RhsT1, RhsT2>::type Rhs;
        const Rhs &rhs;
        
        typedef decltype(lhs * rhs) Type;
    };
    
    
    //! Defines the expression type of a multiplication
    template<typename _Lhs, typename _Rhs, int Side, bool isT1>
    struct MultExpression {
        typedef ProductType<_Lhs,_Rhs,Side,isT1> Types;
        
        typedef typename Types::Type Expression;
        typedef typename Types::Lhs Lhs;
        typedef typename Types::Rhs Rhs;
        
        Expression expression;
        MultExpression (const Lhs &lhs, const Rhs &rhs) : expression(lhs * rhs) {}
    };
    
    
    // --- Multiplication between an Alt matrix and anything
    
    template<typename Lhs, typename Rhs, int Side>
    class AltEigenProduct : Eigen::internal::no_assignment_operator, public AltMatrixBase< AltEigenProduct<Lhs,Rhs,Side> >
    {
        typedef MultExpression<Lhs, Rhs, Side, true> HolderT1;
        typedef MultExpression<Lhs, Rhs, Side, false> HolderT2;
        
        typedef typename HolderT1::Expression ExprIfT1;
        typedef typename HolderT2::Expression  ExprIfT2;
        
        union {
            HolderT1 *_t1;
            HolderT2 *_t2;
        };
        
        bool m_isT1;
        
    public:
        typedef typename kqp::scalar<Lhs>::type Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        // Initialisation when the Alt is on the left
        friend void initAltEigenProduct(AltEigenProduct<Lhs, Rhs, Eigen::OnTheLeft> &product, const Lhs &lhs, const Rhs &rhs) {
            typedef MultExpression<Lhs, Rhs, Eigen::OnTheLeft, true> HolderT1;
            typedef MultExpression<Lhs, Rhs, Eigen::OnTheLeft, false> HolderT2;
            if ((product.m_isT1 = lhs.isT1()))  product._t1 = new HolderT1(lhs.t1(), rhs);
            else                                product._t2 = new HolderT2(lhs.t2(), rhs); 
        }
        
        // Initialisation when the Alt is on the right
        friend void initAltEigenProduct(AltEigenProduct<Lhs, Rhs, Eigen::OnTheRight> &product, const Lhs &lhs, const Rhs &rhs) {
            typedef MultExpression<Lhs, Rhs, Eigen::OnTheRight, true> HolderT1;
            typedef MultExpression<Lhs, Rhs, Eigen::OnTheRight, false> HolderT2;
            if ((product.m_isT1 = rhs.isT1()))  product._t1 = new HolderT1(lhs, rhs.t1());
            else                                product._t2 = new HolderT2(lhs, rhs.t2()); 
        }
        
        
        AltEigenProduct(const Lhs& lhs, const Rhs& rhs) {
            initAltEigenProduct(*this, lhs, rhs);
        }
        
        AltEigenProduct() {
            if (m_isT1) delete _t1; else delete _t2;
        }
        
        Index rows() const { return m_isT1 ? _t1->expression.rows() : _t2->expression.rows(); }
        Index cols() const {return m_isT1 ? _t1->expression.rows() : _t2->expression.rows();  }
        
        Real squaredNorm() const { return m_isT1 ? _t1->expression.squaredNorm() : _t2->expression.squaredNorm();  }
        Scalar trace() const { return m_isT1 ? _t1->expression.trace() : _t2->expression.trace();  }
        
        
        bool isT1() const { return m_isT1; }
        
        
        inline const ExprIfT1 & t1() const { return _t1->expression; }
        inline const ExprIfT2 & t2() const { return _t2->expression; }
        
        template<typename Dest> void evalTo(Dest& dest) const {
            if (m_isT1) 
                dest = t1();
            else 
                dest = t2();
        }
        
        
        template<typename Dest> void lazyAssign(Dest& dest) const {
            if (m_isT1) 
                noalias(dest) = t1();
            else 
                noalias(dest) = t2();
        }
        
        
        void printExpression(std::ostream &out) const {
            out << "altx[";
            if (m_isT1) 
                kqp::printExpression(out, t1());
            else 
                kqp::printExpression(out, t2());
            out << "]";
        }
    };
    
    
    // --- Alt * Dense
    template<class Derived,class OtherDerived>
    inline const AltEigenProduct<Derived,OtherDerived,Eigen::OnTheLeft>
    operator*(const kqp::AltMatrixBase<Derived> &a, const Eigen::MatrixBase<OtherDerived> &b) {
        return AltEigenProduct<Derived,OtherDerived,Eigen::OnTheLeft>(a.derived(), b.derived());
    }
    
    // --- Dense * Alt
    template<class Derived,class OtherDerived>
    inline const AltEigenProduct<Derived,OtherDerived,Eigen::OnTheRight>
    operator*(const Eigen::MatrixBase<Derived> &a, const kqp::AltMatrixBase<OtherDerived> &b) {
        return AltEigenProduct<Derived,OtherDerived,Eigen::OnTheRight>(a.derived(), b.derived());
    }
    
    
    // --- Alt * Diagonal
    template<class Derived,class OtherDerived>
    inline const AltEigenProduct<Derived,OtherDerived,Eigen::OnTheLeft>
    operator*(const kqp::AltMatrixBase<Derived> &a, const Eigen::DiagonalBase<OtherDerived> &b) {
        return AltEigenProduct<Derived,OtherDerived,Eigen::OnTheLeft>(a.derived(), b.derived());
    }
    
    // --- Diagonal * Alt
    template<class Derived,class OtherDerived>
    inline const AltEigenProduct<Derived,OtherDerived,Eigen::OnTheRight>
    operator*(const Eigen::DiagonalBase<Derived> &a, const kqp::AltMatrixBase<OtherDerived> &b) {
        return AltEigenProduct<Derived,OtherDerived,Eigen::OnTheRight>(a.derived(), b.derived());
    }
    
    
    // --- Alt * Alt
    template<class Derived,class OtherDerived>
    inline const AltEigenProduct<Derived,OtherDerived,Eigen::OnTheLeft>
    operator*(const kqp::AltMatrixBase<Derived> &a, const kqp::AltMatrixBase<OtherDerived> &b) {
        return AltEigenProduct<Derived,OtherDerived, Eigen::OnTheLeft>(a.derived(), b.derived());
    }
    
    // --- Scalar * AltMatrix
    template<class Derived>
    inline AltEigenProduct<typename Eigen::internal::traits<Derived>::Scalar, Derived, Eigen::OnTheRight>
    operator*(typename Eigen::internal::traits<Derived>::Scalar alpha, const kqp::AltMatrixBase<Derived> &a) {
        return AltEigenProduct<typename Eigen::internal::traits<Derived>::Scalar, Derived, Eigen::OnTheRight>(alpha, a.derived());
    }  
    
}

namespace Eigen {
    
    // --- Multiplication of two diagonal wrappers
    template<typename Derived, typename OtherDerived>
    auto operator*(const DiagonalWrapper<Derived> &a, const DiagonalWrapper<OtherDerived> &b) 
    -> decltype(a.diagonal().cwiseProduct(b.diagonal()).asDiagonal()) {
        return a.diagonal().cwiseProduct(b.diagonal()).asDiagonal();
    }
    
    // --- Multiplication scalar * diagonal
    template<typename Derived>
    auto operator*(typename Eigen::internal::traits<Derived>::Scalar alpha, const DiagonalWrapper<Derived> &a) -> decltype((alpha * a.diagonal()).asDiagonal()) {
        return (alpha * a.diagonal()).asDiagonal();
    }
    
    
    // --- Traits
    namespace internal {
        // AlMatrixBase
        template<typename Derived>
        struct traits< kqp::AltMatrixBase<Derived> > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<Derived>::Scalar Scalar;
            
            enum {
                Flags = 0,
                RowsAtCompileTime = Eigen::Dynamic,
                ColsAtCompileTime = Eigen::Dynamic
            };
            
        };
        
        // AltMatrix
        template<typename T1, typename T2>
        struct traits< kqp::AltMatrix<T1,T2> > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<T1>::Scalar Scalar;
            enum {
                Flags = NestByRefBit,
                RowsAtCompileTime = Eigen::Dynamic,
                ColsAtCompileTime = Eigen::Dynamic
            };
        };
        
        // Transpose of AltMatrix
        template<typename Derived>
        struct traits< kqp::Transpose< kqp::AltMatrixBase<Derived> > > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<Derived>::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Eigen::Dynamic,
                ColsAtCompileTime = Eigen::Dynamic
            };
        };
        
        template<typename BaseT1, typename BaseT2>
        struct traits< kqp::Transpose< kqp::AltMatrix<BaseT1, BaseT2> > > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits< kqp::AltMatrix<BaseT1, BaseT2> >::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Eigen::Dynamic,
                ColsAtCompileTime = Eigen::Dynamic
            };
        };
        template<typename UnaryOp, typename Derived>
        struct traits<kqp::AltCwiseUnaryOp<UnaryOp, Derived>> {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<Derived>::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Eigen::Dynamic,
                ColsAtCompileTime = Eigen::Dynamic
            };
        };
        
        template<typename Derived>
        struct traits<kqp::AltAsDiagonal<Derived>> {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<Derived>::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Eigen::Dynamic,
                ColsAtCompileTime = Eigen::Dynamic
            };
        };
        
        
        // Alt * Eigen
        template<typename Lhs, typename Rhs, int Side>
        struct traits< kqp::AltEigenProduct<Lhs, Rhs, Side> > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            
            typedef typename scalar_product_traits<typename kqp::scalar<Lhs>::type, typename kqp::scalar<Rhs>::type>::ReturnType Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Eigen::Dynamic,
                ColsAtCompileTime = Eigen::Dynamic
            };
        };
        
        
    }
}
#endif
