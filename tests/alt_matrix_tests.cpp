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
    // Test for product AltMatrix x Matrix
    template<class Lhs, class Rhs>
    int test_pre_product(const Lhs &mA, const Rhs &mB) {
        typedef Eigen::Matrix<typename Lhs::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
        std::cerr << "Alt(" << KQP_DEMANGLE(mA) << ") x " << KQP_DEMANGLE(mB) << " : ";
        kqp::AltMatrix<typename Lhs::Scalar> alt(mA);
        
        typename NumTraits<typename Lhs::Scalar>::Real 
        error = ((alt * mB).eval() - mA * mB).squaredNorm(),
        error2 = (ScalarMatrix(alt * mB) - mA * mB).squaredNorm();
        
        std::cerr << "Error: " << error << " [" << error2 << "]" << std::endl;        
        return error < EPSILON && error2 < EPSILON ? 0 : 1;
    }
    
    
    // Test for product Matrix x AltMatrix
    template<class Lhs, class Rhs>
    int test_post_product(const Lhs &mA, const Rhs &mB) {
        typedef Eigen::Matrix<typename Lhs::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        
        std::cerr << KQP_DEMANGLE(mA)  << " x Alt(" << KQP_DEMANGLE(mB)  << "): " ;
        kqp::AltMatrix<typename Lhs::Scalar> alt(mB);
        
        
        typename NumTraits<typename Lhs::Scalar>::Real 
        error = ((mA * alt).eval() - mA * mB).squaredNorm(),
        error2 = (ScalarMatrix(mA * alt) - mA * mB).squaredNorm();
        std::cerr << "Error: " << error << " [" << error2 << "]" << std::endl;        
        
        return error < EPSILON && error2 < EPSILON ? 0 : 1;
        
    }
    
    template<class A, class B, class D>
    int test_pre_post_product(const A &a, const B &b, const D &d) {
        typedef Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> ScalarMatrix;
        kqp::AltMatrix<typename B::Scalar> alt_b(b);
        
        ScalarMatrix m = d * b.transpose() * a.transpose() *  a * b * d;
        ScalarMatrix alt_m =  d * alt_b.transpose() * a.transpose() *  a * alt_b * d;
        
        typename NumTraits<typename A::Scalar>::Real error = (m - alt_m).squaredNorm();
        
        std::cerr << KQP_DEMANGLE(d) << " x T(Alt(" << KQP_DEMANGLE(b)  << ")) x T(" << KQP_DEMANGLE(a)  << ")) x " 
        << KQP_DEMANGLE(a) << " x Alt(" << KQP_DEMANGLE(b) << " x " << KQP_DEMANGLE(d) << ": ";
        std::cerr << "Error: " << error << std::endl;        
        
        return error < EPSILON;
    }
    
    
    void test_post_adjoint_product() {        
    }
    
    void test_pre_adjoint_product() {
    }
    
}

namespace kqp {
    int altmatrix_test (std::deque<std::string> &) {
        
#define RANDOM_M(scalar, rows, cols) Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(rows,cols).eval()
#define RANDOM_V(scalar, size) Eigen::Matrix<scalar, Eigen::Dynamic, 1>::Random(size).eval()
#define RANDOM_D(scalar, size) RANDOM_V(scalar,size).asDiagonal()
        
        
#define test_all(x, y) \
test_pre_product(x,y); \
test_post_product(x,y); 
        
        int code = 0;
        
        code |= test_all(RANDOM_M(double,4,6), RANDOM_M(double,6,4));
        
        code |= test_all(RANDOM_D(double,6), RANDOM_M(double,6,5));
        
        code |= test_all(RANDOM_M(double,5,10), RANDOM_D(double,10));
        
        //  test_all(RANDOM_D(double, 5), RANDOM_D(double,5));
        
        code |= test_pre_post_product(RANDOM_M(double,10,5), RANDOM_M(double,5,4), RANDOM_D(double,4));
        
        
        return code;
    }
}

