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

#include <iostream>
#include <ctime>

#define KQP_NO_EXTERN_TEMPLATE
#include <kqp/alt_matrix.hpp>

using namespace std;
using namespace kqp;
using namespace Eigen;

#define RANDOM_M(scalar, rows, cols) Eigen::Matrix<scalar,Dynamic,Dynamic>::Random(rows,cols).eval()
#define RANDOM_V(scalar, size) Eigen::Matrix<scalar,Dynamic,1>::Random(size).eval()
#define RANDOM_D(scalar, size) RANDOM_V(scalar,size).asDiagonal()


namespace {
    
    using Eigen::Matrix;
    using Eigen::Dynamic;

    using Eigen::SparseMatrix;
    using Eigen::ColMajor;
    
    // Dense or Identity matrix
    template<typename Scalar> struct AltDenseDiagonal {
        typedef AltMatrix< Eigen::Matrix<Scalar,Dynamic,Dynamic> , Eigen::DiagonalWrapper<Eigen::Matrix<Scalar,Dynamic,1> > > type;
    };    
    
    template<typename Derived>
    auto adjoint(const Eigen::MatrixBase<Derived> &m) -> decltype(m.adjoint()) {
        return m.adjoint();
    }
    
    template<typename Derived>
    const Eigen::DiagonalWrapper<Derived> & adjoint(const Eigen::DiagonalWrapper<Derived> &m)  {
        return m;
    }


    int test_alt_dense_matrix() {
        int code = 0;
        double error;
        typedef Eigen::Matrix<double,Dynamic,Dynamic> ScalarMatrix;

        Eigen::MatrixXd mA = Eigen::MatrixXd::Random(3,5);

        AltDense<double>::type mC = Eigen::Identity<double>(5,5);
        error  = (ScalarMatrix(mA * mC) - mA).squaredNorm();
        std::cerr << "Error: " << error << std::endl;        
        code |= error > EPSILON;
   
        Eigen::MatrixXd mB = Eigen::MatrixXd::Random(5,4);
        mC = mB;
        error  = (ScalarMatrix(mA * mC) - mA * mB).squaredNorm();
        std::cerr << "Error: " << error << std::endl;        
        code |= error > EPSILON;

        return code;
    }
    
    
    template<class Lhs, class Rhs>
    int test_pre_product(const Lhs &mA, const Rhs &mB) {
        typedef typename kqp::scalar<Lhs>::type Scalar;
        typedef Eigen::Matrix<Scalar,Dynamic,Dynamic> ScalarMatrix;
        
        std::cerr << "Alt(" << KQP_DEMANGLE(mA) << ") x " << KQP_DEMANGLE(mB) << " : ";
        typename AltDenseDiagonal< typename Lhs::Scalar >::type alt(mA);
        
        ScalarMatrix alt_mB = alt * mB;
        
        typename NumTraits<Scalar>::Real 
        error = (alt_mB - ScalarMatrix(mA * mB)).squaredNorm();        
        std::cerr << "Error: " << error << std::endl;        
        return error > EPSILON;
    }
    
    template<class Lhs, class Rhs>
    int test_post_product(const Lhs &mA, const Rhs &mB) {
        typedef typename kqp::scalar<Lhs>::type Scalar;
        typedef Eigen::Matrix<Scalar,Dynamic,Dynamic> ScalarMatrix;
        
        std::cerr << KQP_DEMANGLE(mA)  << " x Alt(" << KQP_DEMANGLE(mB)  << "): " ;
        typename AltDenseDiagonal<Scalar>::type alt(mB);
        
        ScalarMatrix mA_alt = mA * alt;
        typename NumTraits<Scalar>::Real 
        error = (mA_alt - ScalarMatrix(mA * mB)).squaredNorm(),
        error2 = (ScalarMatrix(mA * alt) - ScalarMatrix(mA * mB)).squaredNorm();
        std::cerr << "Error: " << error << " [" << error2 << "]" << std::endl;        
        return error > EPSILON;
        
    }
    
    
    template<class A, class B, class D>
    int test_pre_post_product(const A &a, const B &b, const D &d) {
        typedef Eigen::Matrix<typename A::Scalar, Dynamic, Dynamic> ScalarMatrix;
        typename AltDenseDiagonal< typename B::Scalar >::type alt_b(b);
        
        
        ScalarMatrix alt_m =  d * alt_b.adjoint() * a.adjoint() *  a * alt_b * d;
        
        ScalarMatrix m = d * b.adjoint() * a.adjoint() *  a * b * d;
        typename NumTraits<typename A::Scalar>::Real error = (m - alt_m).squaredNorm();
        
        std::cerr << KQP_DEMANGLE(d) << " x T(Alt(" << KQP_DEMANGLE(b)  << ")) x T(" << KQP_DEMANGLE(a)  << ")) x " 
        << KQP_DEMANGLE(a) << " x Alt(" << KQP_DEMANGLE(b) << " x " << KQP_DEMANGLE(d) << ": ";
        std::cerr << "Error: " << error << std::endl;        
        return error > EPSILON;
        
    }
    
    
    template<class A, class B, class D>
    int test_pre_post_product_2(const A &a, const B &b, const D &d) {
        typedef Eigen::Matrix<typename A::Scalar,Dynamic,Dynamic> ScalarMatrix;
        typename AltDenseDiagonal< typename B::Scalar >::type alt_b(b);
        
        ScalarMatrix m = d * b.adjoint() * a * b * d;
        ScalarMatrix alt_m =  d * alt_b.adjoint() *  a * alt_b * d;
        
        typename NumTraits<typename A::Scalar>::Real error = (m - alt_m).squaredNorm();
        
        std::cerr << KQP_DEMANGLE(d) << " x T(Alt(" << KQP_DEMANGLE(b)  << ")) x "
        << KQP_DEMANGLE(a) << " x Alt(" << KQP_DEMANGLE(b) << " x " << KQP_DEMANGLE(d) << ": ";
        std::cerr << "Error: " << error << std::endl;     
        return error > EPSILON;
        
    }
    
    
    
    template<class Lhs, class Rhs>
    int test_adjoint_pre_product(const Lhs &mA, const Rhs &mB) {
        typedef Eigen::Matrix<typename Lhs::Scalar, Dynamic, Dynamic> ScalarMatrix;
        
        std::cerr << "T(" << KQP_DEMANGLE(mB) << ") x " << "T(Alt(" << KQP_DEMANGLE(mA) << ")) : ";
        typename AltDenseDiagonal< typename Lhs::Scalar >::type alt(mA);
        
        ScalarMatrix t_alt_mB = adjoint(mB) * alt.adjoint();
        
        typename NumTraits<typename Lhs::Scalar>::Real 
        error = (t_alt_mB -  ScalarMatrix(adjoint(mB) * adjoint(mA))).squaredNorm();
        
        std::cerr << "Error: " << error << std::endl;        
        return error > EPSILON;
    }
    
    template<class Lhs, class Rhs>
    int test_adjoint_post_product(const Lhs &mA, const Rhs &mB) {
        typedef Eigen::Matrix<typename Lhs::Scalar,Dynamic,Dynamic> ScalarMatrix;
        
        std::cerr << KQP_DEMANGLE(mA)  << " x Alt(" << KQP_DEMANGLE(mB)  << "): " ;
        typename AltDenseDiagonal< typename Lhs::Scalar >::type alt(mB);
        
        ScalarMatrix mA_alt = mA * alt;
        typename NumTraits<typename Lhs::Scalar>::Real 
        error = (mA_alt - ScalarMatrix(mA * mB)).squaredNorm(),
        error2 = (ScalarMatrix(mA * alt) - ScalarMatrix(mA * mB)).squaredNorm();
        std::cerr << "Error: " << error << " [" << error2 << "]" << std::endl;        
        return error > EPSILON;
    }
    
    int test_block() {
        Eigen::MatrixXd m = RANDOM_M(double, 7, 7);
        typename AltDense<double>::type alt_m(m);
        
        double error = std::abs(m.block(2, 3, 2, 3).squaredNorm() - alt_m.block(2, 3, 2, 3).squaredNorm());
        std::cerr << "Block error [dense]: " << error << std::endl;
        
        auto d = RANDOM_D(double, 7);
        typename AltDenseDiagonal<double>::type alt_d(d);
        double error2 = std::abs(Eigen::MatrixXd(d).block(2, 3, 2, 3).squaredNorm() - alt_d.block(2, 3, 2, 3).squaredNorm());
        std::cerr << "Block error [diagonal]: " << error2 << std::endl;
        
        return error > EPSILON || error2 > EPSILON;
    }
    
    int test_block_pre_mult() {
        Eigen::MatrixXd m = RANDOM_M(double, 7, 7);
        typename AltDense<double>::type alt_m(m);
        
        Eigen::MatrixXd m2 = RANDOM_M(double, 4, 4);
        
        double error = (m.block(1,1,5,4) * m2 - Eigen::MatrixXd(alt_m.block(1,1,5,4) * m2)).squaredNorm();
        std::cerr << "AltDense block * Dense [Dense] error: " << error << std::endl;
        
        
        alt_m = Identity<double>(7,7);
        Eigen::MatrixXd id7 = Eigen::MatrixXd::Identity(7,7);
        double error2 = (id7.block(2,1,5,4) * m2 - Eigen::MatrixXd(alt_m.block(2,1,5,4) * m2)).squaredNorm();
        std::cerr << "AltDense block * Dense [Identity] error: " << error2 << std::endl;
        
        double error3 = (id7.block(1,2,3,4) * m2 - Eigen::MatrixXd(alt_m.block(1,2,3,4) * m2)).squaredNorm();
        std::cerr << "AltDense block * Dense [Identity/2] error: " << error3 << std::endl;
        
        return error > EPSILON  || error2 > EPSILON || error3 > EPSILON;
    }
    
    
    int test_block_post_mult() {
        Eigen::MatrixXd m = RANDOM_M(double, 7, 7);
        typename AltDense<double>::type alt_m(m);
        
        Eigen::MatrixXd m2 = RANDOM_M(double, 4, 4);
        
        double error = (m2 * m.block(1,1,4,5) - Eigen::MatrixXd(m2 * alt_m.block(1,1,4,5))).squaredNorm();
        std::cerr << "Dense * AltDense block [Dense] error: " << error << std::endl;
        
        
        alt_m = Identity<double>(7,7);
        Eigen::MatrixXd id7 = Eigen::MatrixXd::Identity(7,7);
        double error2 = (m2 * id7.block(2,1,4,5) - Eigen::MatrixXd(m2 * alt_m.block(2,1,4,5))).squaredNorm();
        std::cerr << "Dense * AltDense block [Identity] error: " << error2 << std::endl;
        
        double error3 = (m2 * id7.block(1,2,4,3) - Eigen::MatrixXd(m2 * alt_m.block(1,2,4,3))).squaredNorm();
        std::cerr << "Dense * AltDense block [Identity/2] error: " << error3 << std::endl;
        
        return error > EPSILON  || error2 > EPSILON || error3 > EPSILON;
    }
    
    
    template <typename Scalar>
    int test_unary() {
        Eigen::MatrixXd m = Eigen::MatrixXd::Random(5,5);
        typename kqp::AltDense<double>::type altm(m);
        
        altm.unaryExprInPlace(Eigen::internal::scalar_multiple_op<double>(2.));
        
        Eigen::Matrix<Scalar,Dynamic,Dynamic> m2 = altm;
        
        double error = (m2 - 2 * m).squaredNorm();
        std::cerr << "Unary error: " << error << std::endl;
        return error > EPSILON;
    }
    
    template<typename Scalar> SparseMatrix<Scalar, Eigen::ColMajor> getColMajorSparse(const Matrix<Scalar, Dynamic, Dynamic> &mat) {
        Matrix<Index, 1, Dynamic> countsPerCol((mat.array() >= 0).template cast<Index>().rowwise().sum());
       
        SparseMatrix<Scalar> s(mat.rows(), mat.cols());
        s.reserve(countsPerCol);

        for(Index i = 0; i < mat.rows(); i++)
            for(Index j = 0; j < mat.cols(); j++)
                    s.insert(i,j) = mat(i,j);
        
        return s;
    }

    
    template <typename Scalar>
    int test_sparse() {
        typedef Matrix<Scalar, Dynamic, Dynamic> ScalarMatrix;
        
        ScalarMatrix mat = RANDOM_M(Scalar,5,5);
        SparseMatrix<Scalar, ColMajor> sMat(getColMajorSparse(mat));

        ScalarMatrix mA = RANDOM_M(Scalar,4,5);
        typename kqp::AltDense<Scalar>::type altA(mA);

        double error = (mA * mat - ScalarMatrix(altA * sMat)).squaredNorm();
        std::cerr << "Sparse error: " << error << std::endl;

        return error > EPSILON;
    }
    
    template <typename Scalar>
    int test_array() {
        typedef Matrix<Scalar, Dynamic, Dynamic> ScalarMatrix;
        ScalarMatrix mat = RANDOM_M(Scalar,5,4);
        ScalarMatrix mat2 = RANDOM_M(Scalar,5,4);
        ScalarMatrix mat3 = RANDOM_M(Scalar,5,4);
        typename kqp::AltDense<Scalar>::type altMat(mat);
        typename kqp::AltDense<Scalar>::type altMat3(mat);

        
        ScalarMatrix res1 = ((mat2.array() - mat.array())).matrix();        
        ScalarMatrix altRes1 = ((mat2.array() - altMat.array())).matrix();
        
        double error1 = (res1 - altRes1).squaredNorm();
        std::cerr << "Array error[1]: " << error1 << std::endl;

        ScalarMatrix res2 = (mat3.array() * (mat2.array() - mat.array())).matrix();        
        ScalarMatrix altRes2 = (altMat3.array() * (mat2.array() - altMat.array())).matrix();

        double error2 = (res2 - altRes2).squaredNorm();
        std::cerr << "Array error[2]: " << error2 << std::endl;
        return error1 > EPSILON || error2 > EPSILON;
    }
    
    int test_rowwise() {
        typedef double Scalar;
        AltDense<Scalar>::type a = AltDense<Scalar>::Identity(10);
        MatrixXd b = MatrixXd::Identity(10,10);
        
        double error = (a.rowwise().squaredNorm() - b.rowwise().squaredNorm()).squaredNorm();
        std::cerr << "Row-wise error[identity]: " << error << std::endl;
        return error > EPSILON;  
    }
    
} // end <> ns



#define test_all(x, y) \
test_pre_product(x,y); \
test_post_product(x,y); \
test_adjoint_pre_product(x,y); \
test_adjoint_post_product(x,y);


    int main (int, const char **) {
        
        int code = 0;

        // Identity test

        code |= test_alt_dense_matrix();
        
        // Multiplication test
        
        code |= test_post_product(5.2, RANDOM_M(double,4,6));
        
        code |= test_all(RANDOM_M(double,4,6), RANDOM_M(double,6,4));
        
        code |= test_all(RANDOM_D(double,6), RANDOM_M(double,6,5));
        
        code |= test_all(RANDOM_M(double,5,10), RANDOM_D(double,10));
        
        code |= test_all(RANDOM_D(double, 5), RANDOM_D(double,5));
        
        code |= test_pre_post_product(RANDOM_M(double,10,5), RANDOM_M(double,5,4), RANDOM_D(double,4));
        
        code |= test_pre_post_product_2(RANDOM_M(double,5,5), RANDOM_M(double,5,4), RANDOM_D(double,4));
        
        // Block tests
        
        code |= test_block();
        
        code |= test_block_pre_mult();
        code |= test_block_post_mult();
        
        // Unary op
        
        code |= test_unary<double>();


        // Sparse
        
        code |= test_sparse<double>();
        
        // Row-wise
        
        code |= test_rowwise();
        
        // Array
        // FIXME: not working for the moment
//        code |= test_array<double>();

        return code;
    
}

