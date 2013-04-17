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

#include <boost/shared_ptr.hpp>
#include <algorithm>

#include <kqp/kqp.hpp>
#include <kqp/eigen_identity.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace kqp {
    
    // Should use template template aliases?
    
    
    // Forward declarations
    template<typename T1, typename T2> class AltMatrix;
    template<typename _AltMatrix> class AltBlock;
    template<typename _AltMatrix> class RowWise;
    template<typename Derived> struct scalar;
    template<typename Derived> struct DiagonalBlockWrapper;
    template<typename Derived> class AltArrayWrapper;
    
    // Returns the associated scalar
    template<typename Derived> struct scalar { typedef typename Eigen::internal::traits<Derived>::Scalar type; };
    template<> struct scalar<double> { typedef double type; };
    template<> struct scalar<float> { typedef float type; };
    template<> struct scalar<std::complex<double>> { typedef std::complex<double> type; };
    template<> struct scalar<std::complex<float>> { typedef std::complex<float> type; };
    
    
    // ---- Predefined Alt-based matrices
    
    //! Diagonal or Identity matrix
    template<typename Scalar> struct AltDiagonal {
        typedef Eigen::DiagonalWrapper<Eigen::Matrix<Scalar,Dynamic,1>> Diagonal;
        typedef typename Eigen::MatrixBase<Eigen::Matrix<Scalar,Dynamic,Dynamic> >::IdentityReturnType Identity;
        
        typedef AltMatrix<Diagonal,Identity> type;
    };
    
    //! A dense vector or a constant vector
    template<typename Scalar> struct AltVector {
        typedef Eigen::Matrix<Scalar,Dynamic,1>  VectorType;
        typedef typename Eigen::Matrix<Scalar,Dynamic,1>::ConstantReturnType ConstantVectorType;
        
        typedef AltMatrix<VectorType, ConstantVectorType> type;
    };
    
    
    //! Dense or Identity matrix
    template<typename Scalar> struct AltDense {
        typedef Eigen::Matrix<Scalar,Dynamic,Dynamic> DenseType;
        typedef Eigen::Identity<Scalar> IdentityType;
        typedef AltMatrix<DenseType, IdentityType> type;
        
        static  inline type Identity(Index n) { 
            return type(Eigen::Matrix<Scalar,Dynamic,Dynamic>::Identity(n,n));
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
        
        AltArrayWrapper<const Derived> array() const {
            return this->derived();
        }
        
      };
    
    
    /** Rank update with an Eigen matrix expression (used for recursion termination) */
    template<typename MatrixType, unsigned int UpLo, typename Derived>
    void rankUpdate(Eigen::SelfAdjointView<MatrixType, UpLo> &&matrix, const Eigen::MatrixBase<Derived> &mA, const typename MatrixType::Scalar alpha) {
        matrix.rankUpdate(mA, alpha);
    }

    /** 
    * Rank update with an AltMatrix 
    * @param matrix The matrix to be updated
    * @param mA The matrix used for update
    * @param alpha The update coefficient
    */
    template<typename MatrixType, unsigned int UpLo, typename Derived> 
    void rankUpdate2(Eigen::SelfAdjointView<MatrixType, UpLo> &&matrix, const AltMatrixBase<Derived> &mA, const typename MatrixType::Scalar alpha) {
        if (mA.derived().isT1())
            rankUpdate(std::move(matrix), mA.derived().t1(), alpha);
        else 
            rankUpdate(std::move(matrix), mA.derived().t2(), alpha);
    }

    
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
        
        template<typename Dest> inline void evalTo(Dest& dst) const
        { if (isT1()) dst = t1(); else dst = t2(); }

        
    };
    
    // --- Unary operator
    template<typename UnaryOp, typename XprType> class AltCwiseUnaryOp :
    Eigen::internal::no_assignment_operator, public AltMatrixBase<AltCwiseUnaryOp<UnaryOp,XprType>> {
    protected:
        const typename XprType::Nested m_xpr;
        const UnaryOp m_functor;
    public:
        typedef AltCwiseUnaryOp<UnaryOp,XprType> Nested;
        typedef typename Eigen::internal::traits<Nested>::Scalar Scalar;
        
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
        
        Scalar sum() const {
            if (isT1()) return t1().sum(); 
            else return t2().sum();
        }
    };
    
    
    
    // --- Helper functions
    template<typename Derived>
    const typename Eigen::MatrixBase<Derived>::AdjointReturnType adjoint(const Eigen::MatrixBase<Derived>& x) { return x.adjoint(); }
    
    template<typename Derived> 
    Eigen::DiagonalWrapper<const Derived> adjoint(const Eigen::DiagonalWrapper<const Derived> & x)  { return x; }

    template<typename Scalar>
    Eigen::Identity<Scalar> adjoint(const Eigen::Identity<Scalar> & x)  { return x; }

    template<typename Derived>
    auto blockSquaredNorm(const Eigen::MatrixBase<Derived>& x, Index row, Index col, Index rowSize, Index colSize) -> decltype(x.squaredNorm()) { 
        return x.block(row,col,rowSize,colSize).squaredNorm(); 
    }
    
    template<typename Derived> 
    auto blockSquaredNorm(const Eigen::DiagonalWrapper<Derived> & x, Index row, Index col, Index rowSize, Index colSize) -> decltype(x.diagonal().squaredNorm())  { 
        return x.diagonal().segment(std::max(row,col), std::min(row+rowSize, col+colSize)).squaredNorm(); 
    }
    
    template<typename Scalar>
    Scalar blockSquaredNorm(const Eigen::Identity<Scalar> &, Index row, Index col, Index rowSize, Index colSize)   {
        return std::max(row,col) - std::min(row+rowSize, col+colSize);
    }
    
    template<typename Derived>
    auto squaredNorm(const Eigen::MatrixBase<Derived>& x) -> decltype(x.derived().squaredNorm()) { return x.derived().squaredNorm(); }
    
    template<typename Derived> 
    auto squaredNorm(const Eigen::DiagonalWrapper<Derived> & x) -> decltype(x.diagonal().squaredNorm())  { return x.diagonal().squaredNorm(); }
    
    // ---- Transpose
    
    
    
    //! Adjoint
    template<typename Derived>
    struct Adjoint : public AltMatrixBase< Adjoint<Derived> > {
        typename Eigen::internal::ref_selector<Derived>::type nested;
    public:
        
        Adjoint(Derived &nested) : nested(nested) {}
        
        Index rows() const { return nested.cols(); }
        Index cols() const { return nested.rows(); }
        
        bool isT1() const { return nested.isT1(); }
        
        typedef decltype(kqp::adjoint(nested.t1())) T1;
        typedef decltype(kqp::adjoint(nested.t2())) T2;
        
        inline T1 t1() const  { return adjoint(nested.t1()); }
        inline T2 t2() const  { return adjoint(nested.t2()); }
        
        
        void printExpression(std::ostream &out) const {
            out << "Adjoint(";
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

    template <typename Scalar>
    void printExpression(std::ostream &out, const Eigen::Identity<Scalar> &x) {
        out << "Id[" << KQP_DEMANGLE(Scalar) << "; " << x.rows() << " x " << x.cols() << "]";
    }

    template <typename Derived>
    void printExpression(std::ostream &out, const Eigen::Transpose<Derived> &x) {
        out << "adjoint(";
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
    
    
    
    // ---- Resizing

    //! Resize of fully dynamic matrices
    template<class Derived>
    typename boost::enable_if_c<(Eigen::internal::traits<Derived>::RowsAtCompileTime == Dynamic) && (Eigen::internal::traits<Derived>::ColsAtCompileTime == Dynamic), void>::type 
    resize(Eigen::EigenBase<Derived> &matrix, bool conservative, Index rows, Index cols) {
        if (conservative) matrix.derived().conservativeResize(rows, cols); else matrix.derived().resize(rows, cols);
    }
    
    
    //! Resize of vectors
    template<class Derived>
    typename boost::enable_if_c<(Eigen::internal::traits<Derived>::RowsAtCompileTime == Dynamic) && (Eigen::internal::traits<Derived>::ColsAtCompileTime != Dynamic), void>::type 
    resize(Eigen::EigenBase<Derived> &matrix, bool conservative, Index rows, Index cols) {
        if (cols != matrix.cols())
            KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot change the number of columns in a fixed column-sized matrix (%d to %d)", %matrix.cols() %cols);
        if (conservative) matrix.derived().conservativeResize(rows); else matrix.derived().resize(rows);
    }
    
    //! Resize of row vectors
    template<class Derived>
    typename boost::enable_if_c<(Eigen::internal::traits<Derived>::RowsAtCompileTime != Dynamic) && (Eigen::internal::traits<Derived>::ColsAtCompileTime == Dynamic), void>::type 
    resize(Eigen::EigenBase<Derived> &matrix, bool conservative, Index rows, Index cols) {
        if (rows != matrix.rows())
            KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Cannot change the number of rows in a fixed row-sized matrix (%d to %d)", %matrix.rows() %rows);
        if (conservative) matrix.derived().conservativeResize(cols); else matrix.derived().resize(cols);
    }

    //! Default resize
    template<typename Scalar>
    void resize(Eigen::Identity<Scalar> &matrix, bool conservative, Index rows, Index cols) {
        if (conservative)
            matrix.conservativeResize(rows, cols);
        else
            matrix.resize(rows, cols);
    }


    
    // --- AltMatrix inner storage of Eigen matrices
    
    //! Default storage type for AltMatrix nested types
    template<typename Derived>
    struct storage {
        typedef Derived & ReturnType; 
        typedef const Derived & ConstReturnType; 
        typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;

        Derived m_value;
        
        storage() {}
        storage(const Derived &value) : m_value(value) {}
        ConstReturnType get() const { return m_value; }
        ReturnType get() { return m_value; }

        void swap(storage &other) { m_value.swap(other.m_value); }
        void swap(Derived &value) { m_value.swap(value); }
        Index rows() const { return m_value.rows(); }
        Index cols() const { return m_value.cols(); }
        
        Scalar trace() const { return m_value.trace(); }
        void resize(Index rows, Index cols) {
            kqp::resize(m_value, false, rows, cols);
        }
        
        void conservativeResize(Index rows, Index cols) {
            kqp::resize(m_value, true, rows, cols);
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
        Scalar sum() const { return m_value.sum(); }
        const std::type_info &getTypeId() const { return typeid(Derived); }

        auto block(Index startRow, Index startCol, Index blockRows, Index blockCols) -> decltype(m_value.block(0,0,0,0)) {
            return m_value.block(startRow, startCol, blockRows, blockCols);
        }

        auto block(Index startRow, Index startCol, Index blockRows, Index blockCols) const 
            -> decltype(const_cast<ConstReturnType>(m_value).block(0,0,0,0)) {
            return m_value.block(startRow, startCol, blockRows, blockCols);
        }

    };
    
    //! Storage for a constant matrix
    template<typename Scalar, typename Derived>
    struct storage< typename Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Derived> > {
        typedef typename Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Derived> ReturnType;
        typedef const ReturnType ConstReturnType;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        Scalar m_value;
        Index m_rows;
        Index m_cols;
        
        storage() : m_value(0), m_rows(0), m_cols(0) {}
        storage(const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>,Derived> &value) 
        : m_value(value.coeff(0,0)), m_rows(value.rows()), m_cols(value.cols()) {}
        
        ReturnType get() const { return ReturnType(m_rows, m_cols, m_value); }

        
        void swap(storage &other)  { 
            std::swap(m_value,other.m_value);
            std::swap(m_rows,other.m_rows);
            std::swap(m_cols,other.m_cols);
        }
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
        
        Real squaredNorm() const { return std::abs(m_value)*std::abs(m_value) * (Real)m_rows * (Real)m_cols; }
        Scalar sum() const { return (Scalar)m_rows * (Scalar)m_cols; }

        const std::type_info &getTypeId() const { return typeid(ReturnType); }
        
        
        ReturnType block(Index, Index, Index blockRows, Index blockCols) const {
            return ReturnType(blockRows, blockCols, m_value);
        }

    };
    
    

    //! Storage for a diagonal wrapper
    template<typename Derived>
    struct storage< Eigen::DiagonalWrapper<Derived> > {
        typedef Eigen::DiagonalWrapper<const Derived> ConstReturnType;
        typedef Eigen::DiagonalWrapper<const Derived> ReturnType;
        typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;

        Derived m_value;
        
        storage() {}
        storage(const Eigen::DiagonalWrapper<Derived> &value) : m_value(value.diagonal()) {}
        ConstReturnType get() const { return m_value.asDiagonal(); }
        ReturnType get() { return static_cast<ReturnType>(m_value.asDiagonal()); }
        
        void swap(storage &other) { 
            m_value.swap(other.m_value);
        }
        void swap(ReturnType &value) { m_value.swap(value); }
        Index rows() const { return m_value.rows(); }
        Index cols() const { return m_value.rows(); }
        
        void resize(Index rows, Index cols) {
            if (rows != cols) KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize to a non diagonal size (%d x%d)", %rows%cols);
            m_value.resize(rows);
        }
        
        void conservativeResize(Index rows, Index cols) {
            if (rows != cols) KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot resize to a non diagonal size (%d x%d)", %rows%cols);
            m_value.conservativeResize(rows);
        }
        
        typename Eigen::internal::traits<Derived>::Scalar operator()(Index i, Index j) const {
            return i == j ? m_value(i) : 0;
        }
        
        template<typename CwiseUnaryOp>
        void unaryExprInPlace(const CwiseUnaryOp &op) {
            m_value = m_value.unaryExpr(op);
        }
        
        Real squaredNorm() const { return m_value.squaredNorm(); }
        Scalar sum() const { return m_value.sum(); }
        const std::type_info &getTypeId() const { return typeid(ReturnType); }
        
        DiagonalBlockWrapper<Derived> block(Index startRow, Index startCol, Index blockRows, Index blockCols) const {
            return DiagonalBlockWrapper<Derived>(m_value, startRow, startCol, blockRows, blockCols);
        }

    };


    //! Storage for the identity (with cwise)
    template<typename Scalar>
    struct storage< Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Eigen::Matrix<Scalar,Dynamic,Dynamic> >  > {
        Index m_rows, m_cols; 
        
        typedef Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Eigen::Matrix<Scalar,Dynamic,Dynamic> > Type;
        typedef Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Eigen::Matrix<Scalar,Dynamic,1> > VectorType;
        typedef Type ReturnType;
        typedef const Type ConstReturnType;

        storage() {}
        storage(const Type &value) : m_rows(value.rows()), m_cols(value.cols()) {}
        ReturnType get() const { return  Eigen::Matrix<Scalar,Dynamic,Dynamic>::Identity(m_rows, m_cols); }
        
        void swap(storage &other) { 
            std::swap(m_rows, other.m_rows);
            std::swap(m_cols, other.m_cols);
        }
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
        Real squaredNorm() const { return std::min(m_rows, m_cols); }
        Scalar sum() const { return std::min(m_rows, m_cols); }
        const std::type_info &getTypeId() const { return typeid(Type); }
        
        
        
         DiagonalBlockWrapper<VectorType> block(Index startCol, Index startRow, Index blockRows, Index blockCols) const {
             return DiagonalBlockWrapper<VectorType>(VectorType(std::min(m_rows,m_cols),1,1),  startCol, startRow, blockRows, blockCols);
        }

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
        Scalar trace() const { return m_isT1 ? m_t1.get().trace() : m_t2.get().trace(); }
        Scalar sum() const { return m_isT1 ? m_t1.get().sum() : m_t2.get().sum(); }

        template<typename CwiseUnaryOp>
        void unaryExprInPlace(const CwiseUnaryOp &op) {
            if (isT1()) m_t1.unaryExprInPlace(op);
            else m_t2.unaryExprInPlace(op);
        }
        
        const std::type_info &getTypeId() const { 
            if (isT1()) return m_t1.getTypeId();
            return m_t2.getTypeId();
        }
        
        bool isT1() const { return m_isT1; }
        bool isT2() const { return !m_isT1; }
        
        inline typename storage<T1>::ConstReturnType t1() const { return m_t1.get(); }
        inline typename storage<T2>::ConstReturnType t2() const { return m_t2.get(); }
        
        inline typename storage<T1>::ReturnType t1() { return m_t1.get(); }
        inline typename storage<T2>::ReturnType t2() { return m_t2.get(); }
        
        const storage<T1> &getStorage1() const { return m_t1; }
        const storage<T2> &getStorage2() const { return m_t2; }
        
        
        void swap(AltMatrix &other) {
            m_t1.swap(other.m_t1);
            m_t2.swap(other.m_t2);
            std::swap(m_isT1, other.m_isT1);
        }
        
        void swap(T1 &t1) { m_isT1 = true; m_t1.swap(t1); }
        void swap(T2 &t2) { m_isT1 = true; m_t2.swap(t2); }
        
        //! Returns the adjoint
        Adjoint<AltMatrix> adjoint() const { return Adjoint<AltMatrix>(const_cast<AltMatrix&>(*this)); }
        
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
        typedef AltBlock< const Self > ConstBlock;

        friend class AltBlock<Self> ;
        friend class AltBlock<const Self>;
        
        Block row(Index i) { return Block(*this, i, 0, 1, cols()); }
        ConstBlock row(Index i) const { return ConstBlock(const_cast<Self&>(*this), i, 0, 1, cols()); }
        
        Block col(Index j) { return Block(*this, 0, j, rows(), 1); }
        ConstBlock col(Index j) const { return ConstBlock(const_cast<Self&>(*this), 0, j, rows(), 1); }
        
        Block block(Index i, Index j, Index height, Index width) { return Block(*this, i, j, height, width); }
        ConstBlock block(Index i, Index j, Index height, Index width) const { return ConstBlock(const_cast<Self&>(*this), i, j, height, width); }
        
        ConstBlock topRows(Index h) const { return ConstBlock(*this, 0, 0, h, cols()); }
        Block topRows(Index h)  { return Block(*this, 0, 0, h, cols()); }

        ConstBlock bottomRows(Index h) const { return ConstBlock(*this, rows() - h, 0, h, cols()); }
        Block bottomRows(Index h) { return Block(*this, rows() - h, 0, h, cols()); }

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
    
    template <typename Scalar> typename Eigen::Identity<Scalar>::RowWise rowwise(const Eigen::Identity<Scalar> &x) {
		return x.rowwise();
    }

    
    //! Row wise view of an Alt matrix
    template<typename AltMatrix> class RowWise {
        AltMatrix &alt_matrix;
    public:
        typedef typename AltMatrix::Scalar Scalar;
        typedef typename Eigen::NumTraits< Scalar >::Real Real;
        typedef Eigen::Matrix<Real,Dynamic,1> RealVector;
        
        RowWise(AltMatrix &alt_matrix) : alt_matrix(alt_matrix) {
        }
        
        RealVector squaredNorm() const {
            if (alt_matrix.isT1()) return rowwise(alt_matrix.t1()).squaredNorm();
            else return rowwise(alt_matrix.t2()).squaredNorm();
        }
    };
    
    
    
    
    
    
    // ---- Block view of an AltMatrix

    
    template<typename AltMatrix> class AltBlock : public AltMatrixBase<AltBlock<AltMatrix>>  {
    public:
        typedef typename AltMatrix::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        typedef const AltBlock<AltBlock<AltMatrix>> & Nested;
        
        AltBlock(AltMatrix &alt_matrix, Index row, Index col, Index height, Index width) :
        alt_matrix(alt_matrix), row(row), col(col), height(height), width(width),
        range(std::max(row, col), std::min(row+width, col+height) - std::max(row, col) + 1)
        {
        }
        
        
        bool isT1() const { return alt_matrix.isT1(); }
        
        Real squaredNorm() const {
            if (alt_matrix.isT1())
                return kqp::blockSquaredNorm(alt_matrix.t1(),row,col,height,width);
            else
                return kqp::blockSquaredNorm(alt_matrix.t2(),row,col,height,width);
        }
        
        Index rows() const { return width; }
        Index cols() const { return height; }
        
        // Assignement 

        template<typename Scalar, typename Derived, typename OtherDerived>
        void assign(const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Derived> &op1, 
                    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Derived> &op2, Index , Index ) {
            if (op1(0,0) != op2(0,0))
                KQP_THROW_EXCEPTION_F(not_implemented_exception, "Cannot assign a constant matrix to a constant matrix with different values (%g vs %g)", %op1(0,0) %op2(0,0));
        }

        
        template<typename Op, typename Derived, typename OtherDerived>
        void assign(const Eigen::CwiseNullaryOp<Op, Derived> &, const OtherDerived &, Index, Index) {
            KQP_THROW_EXCEPTION_F(not_implemented_exception, "Cannot assign a constant matrix to [%s]", % KQP_DEMANGLE(OtherDerived));
        }
        
        template<typename Scalar, typename OtherDerived>
        void assign(const Eigen::Identity<Scalar> &, const OtherDerived &, Index, Index) {
            KQP_THROW_EXCEPTION_F(not_implemented_exception, "Cannot assign an Id matrix to [%s]", % KQP_DEMANGLE(OtherDerived));
        }
        
        template<typename Derived, int Rows, int Cols, typename OtherDerived>
        void assign(Eigen::Matrix<Derived, Rows, Cols> &mTo, const Eigen::MatrixBase<OtherDerived> &mFrom, Index fromRow, Index fromCol) {
            mTo.block(row,col,height,width) = mFrom.derived().block(fromRow, fromCol, height, width);
        }
        
        template<typename Scalar, int Rows, int Cols, typename Derived>
        void assign(Eigen::Matrix<Derived, Rows, Cols> &, const Eigen::Identity<Scalar> &, Index , Index ) {
            KQP_THROW_EXCEPTION(not_implemented_exception, "Not implemented Matrix = Identity");
        }
        
        template<typename T> friend class AltBlock;
        
        template<typename Derived>
        AltBlock<AltMatrix>& assignTo(const AltBlock<Derived> &from) {
            if (from.height != this->height || from.width != this->width)
                KQP_THROW_EXCEPTION_F(out_of_bound_exception, "Block sizes differ in assignement (%d x %d != %d x %d)", %height %width %from.height %from.width);
            if (alt_matrix.isT1()) {
                if (from.alt_matrix.isT1()) this->assign(alt_matrix.t1(), from.alt_matrix.t1(), from.row, from.col);
                else this->assign(alt_matrix.t1(), from.alt_matrix.t2(), from.row, from.col);
            } else {
                if (from.alt_matrix.isT1()) this->assign(alt_matrix.t2(), from.alt_matrix.t1(), from.row, from.col);
                else this->assign(alt_matrix.t2(), from.alt_matrix.t2(), from.row, from.col);                
            }
            
            return *this;
        }
        
        
    private:
        AltMatrix &alt_matrix;
        Index row, col, height, width;
        std::pair<Index, Index> range;
    public:
        // Type dependent return
        
        auto t1() -> decltype(alt_matrix.m_t1.block(0,0,0,0)) { return alt_matrix.m_t1.block(row, col, height, width); };
        auto t2() -> decltype(alt_matrix.m_t2.block(0,0,0,0)) { return alt_matrix.m_t2.block(row, col, height, width); };
        auto t1() const -> decltype(alt_matrix.m_t1.block(0,0,0,0)) { return alt_matrix.m_t1.block(row, col, height, width); };
        auto t2() const -> decltype(alt_matrix.m_t2.block(0,0,0,0)) { return alt_matrix.m_t2.block(row, col, height, width); };
        

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
    // --- Expressions
    // ----
    
    
    // --- Forward declarations
    
    template<typename Op, typename Lhs, typename Rhs, int ProductOrder> class AltMatrixOp;
    template<typename Op, typename Lhs, typename Rhs, int ProductOrder> class AltArrayOp;
    
    //! Defines the expression type of a binary operator
    template<typename Op, typename Lhs, typename Rhs, int Side, bool isT1> struct ExprType;
    
    
    //! Multiplication operator
    struct MultOp {
        template<typename T1, typename T2> static inline auto apply(const T1 &t1, const T2 &t2) -> decltype(t1 * t2) { return t1 * t2; }
    };
    //! Difference operator
    struct MinusOp {
        template<typename T1, typename T2> static inline auto apply(const T1 &t1, const T2 &t2) -> decltype(t1 - t2) { return t1 - t2; }
    };

    
    //! Defines an expression type when the Alt matrix is on the left
    template<typename Op, typename _Lhs, typename _Rhs, bool isT1> struct ExprType<Op, _Lhs, _Rhs, Eigen::OnTheLeft, isT1> {
        const _Lhs &_lhs;
        typedef decltype(_lhs.t1()) LhsT1;
        typedef decltype(_lhs.t2()) LhsT2;
        typedef typename Eigen::internal::conditional<isT1, LhsT1, LhsT2>::type Lhs;
        const Lhs &lhs;
        
        typedef _Rhs Rhs;        
        const Rhs &rhs;
        
        typedef decltype(Op::apply(lhs, rhs)) Type;
    };
    
    //! Defines an expression type when the Alt matrix is on the right
    template<typename Op, typename _Lhs, typename _Rhs, bool isT1> struct ExprType<Op, _Lhs, _Rhs, Eigen::OnTheRight, isT1> {
        typedef _Lhs Lhs;
        const Lhs &lhs;
        
        const _Rhs &_rhs;
        typedef decltype(_rhs.t1()) RhsT1;
        typedef decltype(_rhs.t2()) RhsT2;
        typedef typename Eigen::internal::conditional<isT1, RhsT1, RhsT2>::type Rhs;
        const Rhs &rhs;
        
        typedef decltype(Op::apply(lhs, rhs)) Type;
    };
    
    
    //! Defines the expression type of a multiplication
    template<typename Op, typename _Lhs, typename _Rhs, int Side, bool isT1>
    struct AltExpression {
        typedef ExprType<Op, _Lhs,_Rhs,Side,isT1> Types;
        
        typedef typename Types::Type Expression;
        typedef typename Types::Lhs Lhs;
        typedef typename Types::Rhs Rhs;
        
        Expression expression;
        AltExpression (const Lhs &lhs, const Rhs &rhs) : expression(Op::apply(lhs, rhs)) {}
    };
    
    
    
    // --- Array wrapper
    
    template<typename Derived>
    class AltArrayBase {
    public:  
        const Derived& derived() const { return static_cast<const Derived&>(*this); }
        typename Eigen::internal::traits<Derived>::MatrixExpr matrix() const { return derived().matrix(); }
    };
    
    template<typename Derived>
    class AltArrayWrapper: public AltArrayBase<AltArrayWrapper<Derived>> {
        typename Eigen::internal::ref_selector<Derived>::type m_alt;
    public:  
        AltArrayWrapper(const Derived &alt) : m_alt(alt) {}
        bool isT1() const { return m_alt.isT1(); } 
        auto t1() const -> decltype(m_alt.t1().array()) { return m_alt.t1().array(); } 
        auto t2() const -> decltype(m_alt.t2().array()) { return m_alt.t2().array(); } 
        
        auto matrix() const -> decltype(m_alt) { return m_alt; }
        
        Index rows() const { 
            return m_alt.rows(); 
        }
        Index cols() const { return m_alt.cols(); }
        
    };
    
    //! Matrix expression out of an array expressio
    template<typename Derived>
    class AltMatrixWrapper: public AltMatrixBase<AltMatrixWrapper<Derived>> {
        typename Eigen::internal::ref_selector<Derived>::type m_expr;
    public:
        AltMatrixWrapper(const Derived &expr) : m_expr(expr) {}
        Index rows() const {
            return m_expr.rows(); 
        }
        Index cols() const { 
            return m_expr.cols(); 
        }
        
        bool isT1() const { return m_expr.isT1(); } 
        auto t1() const -> decltype(m_expr.t1().matrix()) { return m_expr.t1().matrix(); } 
        auto t2() const -> decltype(m_expr.t2().matrix()) { return m_expr.t2().matrix(); } 
        
        template<typename Dest> void evalTo(Dest& dest) const {
            if (isT1()) 
                dest = t1();
            else 
                dest = t2();
        }
        
    };
    
#define KQP_ALT_OP_LEFT(op, opname, Result, AltType, OtherType)\
    template<typename Derived, typename OtherDerived>\
    Result<opname, Derived, OtherDerived, Eigen::OnTheLeft>\
    operator op(const AltType<Derived> &lhs, const OtherType<OtherDerived> &rhs) {\
        return Result<opname, Derived, OtherDerived, Eigen::OnTheLeft>(lhs.derived(), rhs.derived());\
    };
#define KQP_ALT_OP_RIGHT(op, opname, Result, AltType, OtherType)\
    template<typename Derived, typename OtherDerived>\
    Result<opname, Derived, OtherDerived, Eigen::OnTheRight>\
    operator op(const OtherType<Derived> &lhs, const AltType<OtherDerived> &rhs) {\
        return Result<opname, Derived, OtherDerived, Eigen::OnTheRight>(lhs.derived(), rhs.derived());\
    };
    
#define KQP_ALT_OP(op, opname, Result, AltType, OtherType)\
    KQP_ALT_OP_LEFT(op, opname, Result, AltType, OtherType)\
    KQP_ALT_OP_RIGHT(op, opname, Result, AltType, OtherType)\
    KQP_ALT_OP_LEFT(op, opname, Result, AltType, AltType)

KQP_ALT_OP(-, MinusOp, AltArrayOp, AltArrayBase, Eigen::ArrayBase)
KQP_ALT_OP(*, MultOp,  AltArrayOp, AltArrayBase, Eigen::ArrayBase)

    template<typename Op, typename Lhs, typename Rhs, int Side>
    class AltArrayOp : Eigen::internal::no_assignment_operator, public AltArrayBase< AltArrayOp<Op, Lhs,Rhs,Side> >
    {
        typedef AltExpression<Op, Lhs, Rhs, Side, true> HolderT1;
        typedef AltExpression<Op, Lhs, Rhs, Side, false> HolderT2;
        
        typedef typename HolderT1::Expression ExprIfT1;
        typedef typename HolderT2::Expression  ExprIfT2;
        
        boost::shared_ptr<HolderT1> _t1;
        boost::shared_ptr<HolderT2> _t2;
        
        bool m_isT1;
        
    public:
        typedef AltArrayOp Nested;

        typedef typename kqp::scalar<Lhs>::type Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        // Initialisation when the Alt is on the left
        friend void initAltArrayOp(AltArrayOp<Op, Lhs, Rhs, Eigen::OnTheLeft> &op, const Lhs &lhs, const Rhs &rhs) {
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheLeft, true> HolderT1;
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheLeft, false> HolderT2;
            if ((op.m_isT1 = lhs.isT1()))  op._t1.reset(new HolderT1(lhs.t1(), rhs));
            else                           op._t2.reset(new HolderT2(lhs.t2(), rhs)); 
        }
        
        // Initialisation when the Alt is on the right
        friend void initAltArrayOp(AltArrayOp<Op, Lhs, Rhs, Eigen::OnTheRight> &op, const Lhs &lhs, const Rhs &rhs) {
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheRight, true> HolderT1;
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheRight, false> HolderT2;
            if ((op.m_isT1 = rhs.isT1()))  op._t1.reset(new HolderT1(lhs, rhs.t1()));
            else                           op._t2.reset(new HolderT2(lhs, rhs.t2())); 
        }
        
        
        AltArrayOp(const Lhs& lhs, const Rhs& rhs) {
            initAltArrayOp(*this, lhs, rhs);
        }
        
        bool isT1() const { return m_isT1; }
        
        inline const ExprIfT1 & t1() const { return _t1->expression; }
        inline const ExprIfT2 & t2() const { return _t2->expression; }
        
        AltMatrixWrapper<AltArrayOp> matrix() const { return AltMatrixWrapper<AltArrayOp>(*this); }
        
        Index rows() const { 
            return m_isT1 ? _t1->expression.rows() : _t2->expression.rows(); 
        }
        Index cols() const {return m_isT1 ? _t1->expression.cols() : _t2->expression.cols();  }
    };
    
    // --- Multiplication between an Alt matrix and anything
    
    template<typename Op, typename Lhs, typename Rhs, int Side>
    class AltMatrixOp : Eigen::internal::no_assignment_operator, public AltMatrixBase< AltMatrixOp<Op, Lhs,Rhs,Side> >
    {
        typedef AltExpression<Op, Lhs, Rhs, Side, true> HolderT1;
        typedef AltExpression<Op, Lhs, Rhs, Side, false> HolderT2;
        
        typedef typename HolderT1::Expression ExprIfT1;
        typedef typename HolderT2::Expression  ExprIfT2;
        
        boost::shared_ptr<HolderT1> _t1;
        boost::shared_ptr<HolderT2> _t2;
        
        bool m_isT1;
        
    public:
        typedef typename kqp::scalar<Lhs>::type Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        // Initialisation when the Alt is on the left
        friend void initAltMatrixOp(AltMatrixOp<Op, Lhs, Rhs, Eigen::OnTheLeft> &op, const Lhs &lhs, const Rhs &rhs) {
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheLeft, true> HolderT1;
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheLeft, false> HolderT2;
            if ((op.m_isT1 = lhs.isT1()))  op._t1.reset(new HolderT1(lhs.t1(), rhs));
            else                           op._t2.reset(new HolderT2(lhs.t2(), rhs)); 
        }
        
        // Initialisation when the Alt is on the right
        friend void initAltMatrixOp(AltMatrixOp<Op, Lhs, Rhs, Eigen::OnTheRight> &op, const Lhs &lhs, const Rhs &rhs) {
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheRight, true> HolderT1;
            typedef AltExpression<Op, Lhs, Rhs, Eigen::OnTheRight, false> HolderT2;
            if ((op.m_isT1 = rhs.isT1()))  op._t1.reset(new HolderT1(lhs, rhs.t1()));
            else                           op._t2.reset(new HolderT2(lhs, rhs.t2())); 
        }
        
        
        AltMatrixOp(const Lhs& lhs, const Rhs& rhs) {
            initAltMatrixOp(*this, lhs, rhs);
        }
        
       
        Index rows() const { return m_isT1 ? _t1->expression.rows() : _t2->expression.rows(); }
        Index cols() const {return m_isT1 ? _t1->expression.cols() : _t2->expression.cols();  }
        
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
    
    
    
    KQP_ALT_OP(*, MultOp, kqp::AltMatrixOp, kqp::AltMatrixBase, Eigen::EigenBase);
    
    // --- Scalar * AltMatrix
    template<class Derived>
    inline AltMatrixOp<MultOp,typename Eigen::internal::traits<Derived>::Scalar, Derived, Eigen::OnTheRight>
    operator*(typename Eigen::internal::traits<Derived>::Scalar alpha, const kqp::AltMatrixBase<Derived> &a) {
        return AltMatrixOp<MultOp,typename Eigen::internal::traits<Derived>::Scalar, Derived, Eigen::OnTheRight>(alpha, a.derived());
    }  
    
    
    
    
    // ---- Diagonal Wrapper
    template<typename Derived> struct DiagonalBlockWrapper {
        typename Derived::Nested m_value;
        Index startRow, startCol;
        Index rows, cols;
        DiagonalBlockWrapper(const Derived &value, Index startRow, Index startCol, Index blockRows, Index blockCols) 
        : m_value(value), startRow(startRow), startCol(startCol), rows(blockRows), cols(blockCols) {}
        const Derived &derived() const { return m_value; }
    };
    
    
   
    template<typename Lhs, typename Rhs, int side> struct DiagonalBlockWrapperDenseMult;
    
    template<typename Lhs, typename Rhs> struct DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheRight>
    : public Eigen::MatrixBase<DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheRight>> {
        DiagonalBlockWrapper<Lhs> m_lhs;
        typename Rhs::Nested m_rhs;
        typedef typename Eigen::internal::traits<DiagonalBlockWrapperDenseMult>::Scalar Scalar;
        
        Index zerosAbove, zerosLeft, first, size;
        
        DiagonalBlockWrapperDenseMult(const DiagonalBlockWrapper<Lhs>& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs) {
            zerosAbove = std::max(0l, m_lhs.startCol - m_lhs.startRow);
            zerosLeft = std::max(0l,  m_lhs.startRow - m_lhs.startCol);
            // First index of the diagonal
            first = std::max(m_lhs.startCol, m_lhs.startRow);
            // Number of values
            size = std::min(m_lhs.rows - zerosAbove, m_lhs.cols - zerosLeft);
        }
        
        
        Scalar coeff(Index row, Index col) const {            
            if (row < zerosAbove || row >= zerosAbove + size) return 0;
            return m_lhs.m_value[first+row-zerosAbove] * m_rhs(row+zerosLeft-zerosAbove, col);
            
        }
        
        template<typename Dest> inline void evalTo(Dest&) const {
            KQP_THROW_EXCEPTION(not_implemented_exception, "evalTo");
        }
        
        
        Index rows() const { return m_lhs.rows; }
        Index cols() const { return m_rhs.cols(); }
    };
    
    template<typename Lhs, typename Rhs> struct DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheLeft>
    : public Eigen::MatrixBase<DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheLeft>> {
        typename Lhs::Nested m_lhs;
        DiagonalBlockWrapper<Rhs> m_rhs;
        typedef typename Eigen::internal::traits<DiagonalBlockWrapperDenseMult>::Scalar Scalar;
        
        Index zerosAbove, zerosLeft, first, size;
        
        DiagonalBlockWrapperDenseMult(const Lhs& lhs, const DiagonalBlockWrapper<Rhs>& rhs) : m_lhs(lhs), m_rhs(rhs) {
            zerosAbove = std::max(0l, m_rhs.startCol - m_rhs.startRow);
            zerosLeft = std::max(0l,  m_rhs.startRow - m_rhs.startCol);
            // First index of the diagonal
            first = std::max(m_rhs.startCol, m_rhs.startRow);
            // Number of values
            size = std::min(m_rhs.rows - zerosAbove, m_rhs.cols - zerosLeft);
        }
        
        
        Scalar coeff(Index row, Index col) const {            
            if (col < zerosLeft || col >= zerosLeft + size) return 0;
            return  m_lhs(row, col+zerosAbove-zerosLeft) * m_rhs.m_value[first+col-zerosLeft];
            
        }
        
        template<typename Dest> inline void evalTo(Dest&) const {
            KQP_THROW_EXCEPTION(not_implemented_exception, "evalTo");
        }
        
        
        Index rows() const { return m_lhs.rows(); }
        Index cols() const { return m_rhs.cols; }
    };

    
    // --- DiagonalBlockWrapper * Dense
    template<class Lhs,class Rhs>
    DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheRight> operator*(const kqp::DiagonalBlockWrapper<Lhs> &a, const Eigen::MatrixBase<Rhs> &b) 
    {
        return DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheRight>(a, b.derived());
    }

    template<class Lhs,class Rhs>
    DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheLeft> operator*(const Eigen::MatrixBase<Lhs> &a, const kqp::DiagonalBlockWrapper<Rhs> &b) 
    {
        return DiagonalBlockWrapperDenseMult<Lhs,Rhs,Eigen::OnTheLeft>(a.derived(), b);
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
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
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
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
            };
        };
        
        // Adjoint of AltMatrix
        template<typename Derived>
        struct traits< kqp::Adjoint< kqp::AltMatrixBase<Derived> > > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<Derived>::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
            };
        };
        
        template<typename BaseT1, typename BaseT2>
        struct traits< kqp::Adjoint< kqp::AltMatrix<BaseT1, BaseT2> > > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits< kqp::AltMatrix<BaseT1, BaseT2> >::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
            };
        };
        template<typename UnaryOp, typename Derived>
        struct traits<kqp::AltCwiseUnaryOp<UnaryOp, Derived>> {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<Derived>::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
            };
        };
        
        template<typename Derived>
        struct traits<kqp::AltAsDiagonal<Derived>> {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<Derived>::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
            };
        };
        
        template<typename AltMatrix>
        struct traits<kqp::AltBlock<AltMatrix>> {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename traits<AltMatrix>::Scalar Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
            };
        };
        
        template<typename Derived>
        struct traits<kqp::AltArrayWrapper<Derived>> {
            typedef typename traits<Derived>::Scalar Scalar;
            typedef Derived MatrixExpr;
        };

        template<typename Op, typename Lhs, typename Rhs, int Side>
        struct traits<kqp::AltArrayOp<Op, Lhs, Rhs, Side>> {
            typedef typename scalar_product_traits<typename kqp::scalar<Lhs>::type, typename kqp::scalar<Rhs>::type>::ReturnType Scalar;
            typedef MatrixWrapper<kqp::AltArrayOp<Op, Lhs, Rhs, Side>> MatrixExpr;
            
            typedef kqp::Index Index;
            typedef kqp::AltMatrixStorage StorageKind;

            enum {
                Flags = 0,
            };

        };
        
        
        template<typename Derived>
           struct traits<kqp::AltMatrixWrapper<Derived>> {
               typedef typename traits<Derived>::Scalar Scalar;
               typedef kqp::AltMatrixStorage StorageKind;
               typedef typename traits<Derived>::Index Index;
           };
        
                
        template<typename Lhs, typename Rhs, int Side>
        struct traits<kqp::DiagonalBlockWrapperDenseMult<Lhs,Rhs,Side>> {
            typedef Dense StorageKind;
            typedef typename MatrixXd::Index Index;
            typedef typename scalar_product_traits<typename kqp::scalar<Lhs>::type, typename kqp::scalar<Rhs>::type>::ReturnType Scalar;
            typedef MatrixXpr XprKind;
            enum {
                Flags = 0,
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic,
                MaxRowsAtCompileTime = Dynamic,
                MaxColsAtCompileTime = Dynamic,
                CoeffReadCost = 1,
            };
        };
        
        
        // Alt * Eigen
        template<typename Op, typename Lhs, typename Rhs, int Side>
        struct traits< kqp::AltMatrixOp<Op,Lhs, Rhs, Side> > {
            typedef kqp::AltMatrixStorage StorageKind;
            typedef typename MatrixXd::Index Index;
            
            typedef typename scalar_product_traits<typename kqp::scalar<Lhs>::type, typename kqp::scalar<Rhs>::type>::ReturnType Scalar;
            enum {
                Flags = 0,
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic
            };
        };
        
        
    }


} 

#endif
