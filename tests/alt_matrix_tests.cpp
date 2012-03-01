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

#include <kqp/alt_matrix.hpp>

using namespace std;
using namespace kqp;
using namespace Eigen;

namespace {
    // Dense or Identity matrix
    template<typename Scalar> struct AltDenseDiagonal {
        typedef AltMatrix< Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> , Eigen::DiagonalWrapper<Eigen::Matrix<Scalar,Eigen::Dynamic,1> > > type;
    };    
    
    template<typename Derived>
    auto transpose(const Eigen::MatrixBase<Derived> &m) -> decltype(m.transpose()) {
        return m.transpose();
    }

    template<typename Derived>
    const Eigen::DiagonalWrapper<Derived> & transpose(const Eigen::DiagonalWrapper<Derived> &m)  {
        return m;
    }

    
    template<class Lhs, class Rhs>
    int test_pre_product(const Lhs &mA, const Rhs &mB) {
        typedef typename kqp::scalar<Lhs>::type Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
        std::cerr << "Alt(" << KQP_DEMANGLE(mA) << ") x " << KQP_DEMANGLE(mB) << " : ";
        typename AltDenseDiagonal< typename Lhs::Scalar >::type alt(mA);
        
        ScalarMatrix alt_mB = alt * mB;
        
        typename NumTraits<Scalar>::Real 
        error = (alt_mB - ScalarMatrix(mA * mB)).squaredNorm();        
        std::cerr << "Error: " << error << std::endl;        
        return error < EPSILON;
    }
    
    template<class Lhs, class Rhs>
    int test_post_product(const Lhs &mA, const Rhs &mB) {
        typedef typename kqp::scalar<Lhs>::type Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
        std::cerr << KQP_DEMANGLE(mA)  << " x Alt(" << KQP_DEMANGLE(mB)  << "): " ;
        typename AltDenseDiagonal<Scalar>::type alt(mB);
        
        ScalarMatrix mA_alt = mA * alt;
        typename NumTraits<Scalar>::Real 
        error = (mA_alt - ScalarMatrix(mA * mB)).squaredNorm(),
        error2 = (ScalarMatrix(mA * alt) - ScalarMatrix(mA * mB)).squaredNorm();
        std::cerr << "Error: " << error << " [" << error2 << "]" << std::endl;        
        return error < EPSILON;
 
    }
    
    
    template<class A, class B, class D>
    int test_pre_post_product(const A &a, const B &b, const D &d) {
        typedef Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        typename AltDenseDiagonal< typename B::Scalar >::type alt_b(b);
        
        
        ScalarMatrix alt_m =  d * alt_b.transpose() * a.transpose() *  a * alt_b * d;
        
        ScalarMatrix m = d * b.transpose() * a.transpose() *  a * b * d;
        typename NumTraits<typename A::Scalar>::Real error = (m - alt_m).squaredNorm();
        
        std::cerr << KQP_DEMANGLE(d) << " x T(Alt(" << KQP_DEMANGLE(b)  << ")) x T(" << KQP_DEMANGLE(a)  << ")) x " 
        << KQP_DEMANGLE(a) << " x Alt(" << KQP_DEMANGLE(b) << " x " << KQP_DEMANGLE(d) << ": ";
        std::cerr << "Error: " << error << std::endl;        
        return error < EPSILON;

    }
    
    
    template<class A, class B, class D>
    int test_pre_post_product_2(const A &a, const B &b, const D &d) {
        typedef Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        typename AltDenseDiagonal< typename B::Scalar >::type alt_b(b);
        
        ScalarMatrix m = d * b.transpose() * a * b * d;
        ScalarMatrix alt_m =  d * alt_b.transpose() *  a * alt_b * d;
        
        typename NumTraits<typename A::Scalar>::Real error = (m - alt_m).squaredNorm();
        
        std::cerr << KQP_DEMANGLE(d) << " x T(Alt(" << KQP_DEMANGLE(b)  << ")) x "
        << KQP_DEMANGLE(a) << " x Alt(" << KQP_DEMANGLE(b) << " x " << KQP_DEMANGLE(d) << ": ";
        std::cerr << "Error: " << error << std::endl;     
        return error < EPSILON;

    }
    

    
    template<class Lhs, class Rhs>
    int test_adjoint_pre_product(const Lhs &mA, const Rhs &mB) {
        typedef Eigen::Matrix<typename Lhs::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
        std::cerr << "T(" << KQP_DEMANGLE(mB) << ") x " << "T(Alt(" << KQP_DEMANGLE(mA) << ")) : ";
        typename AltDenseDiagonal< typename Lhs::Scalar >::type alt(mA);
        
        ScalarMatrix t_alt_mB = transpose(mB) * alt.transpose();
        
        typename NumTraits<typename Lhs::Scalar>::Real 
        error = (t_alt_mB -  ScalarMatrix(transpose(mB) * transpose(mA))).squaredNorm();
        
        std::cerr << "Error: " << error << std::endl;        
        return error < EPSILON;
    }
    
    template<class Lhs, class Rhs>
    int test_adjoint_post_product(const Lhs &mA, const Rhs &mB) {
        typedef Eigen::Matrix<typename Lhs::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
        std::cerr << KQP_DEMANGLE(mA)  << " x Alt(" << KQP_DEMANGLE(mB)  << "): " ;
        typename AltDenseDiagonal< typename Lhs::Scalar >::type alt(mB);
        
        ScalarMatrix mA_alt = mA * alt;
        typename NumTraits<typename Lhs::Scalar>::Real 
        error = (mA_alt - ScalarMatrix(mA * mB)).squaredNorm(),
        error2 = (ScalarMatrix(mA * alt) - ScalarMatrix(mA * mB)).squaredNorm();
        std::cerr << "Error: " << error << " [" << error2 << "]" << std::endl;        
        return error < EPSILON;
    }
}

#define RANDOM_M(scalar, rows, cols) Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(rows,cols).eval()
#define RANDOM_V(scalar, size) Eigen::Matrix<scalar, Eigen::Dynamic, 1>::Random(size).eval()
#define RANDOM_D(scalar, size) RANDOM_V(scalar,size).asDiagonal()


#define test_all(x, y) \
test_pre_product(x,y); \
test_post_product(x,y); \
test_adjoint_pre_product(x,y); \
test_adjoint_post_product(x,y);


namespace kqp {
    int altmatrix_test (std::deque<std::string> &) {
        
        int code = 0;
        
        code |= test_post_product(5.2, RANDOM_M(double,4,6));
        
        code |= test_all(RANDOM_M(double,4,6), RANDOM_M(double,6,4));
        
        code |= test_all(RANDOM_D(double,6), RANDOM_M(double,6,5));
        
        code |= test_all(RANDOM_M(double,5,10), RANDOM_D(double,10));
        
        code |= test_all(RANDOM_D(double, 5), RANDOM_D(double,5));
        
        code |= test_pre_post_product(RANDOM_M(double,10,5), RANDOM_M(double,5,4), RANDOM_D(double,4));
        
        code |= test_pre_post_product_2(RANDOM_M(double,5,5), RANDOM_M(double,5,4), RANDOM_D(double,4));
        
        // FIXME: add tests for blocks
        
        return code;
    }
}

