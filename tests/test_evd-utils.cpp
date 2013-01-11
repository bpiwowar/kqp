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
#include <string>
#include <deque>

#define KQP_NO_EXTERN_TEMPLATE

#include <kqp/evd_utils.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include "tests_utils.hpp"

DEFINE_LOGGER(logger, "kqp.test.probabilities");

namespace kqp {
    template<typename Scalar> int test_orthonormalization(bool positive = false) {
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        int dim = 10;
        int n = 5;
        int r = 5;
        
        ScalarMatrix mX = generateMatrix<Scalar>(dim, dim).leftCols(n);
        ScalarMatrix mY = ScalarMatrix::Random(n, r);
        RealVector mD = RealMatrix::Random(r,1);
        if (positive) 
            mD = mD.cwiseAbs();
        
        auto fs = DenseSpace<Scalar>::create(dim);
        
        Decomposition<Scalar> d(fs, typename Dense<Scalar>::SelfPtr(new Dense<Scalar>(mX)), mY, mD, false);
        Orthonormalize<Scalar>::run(d.fs, d.mX, d.mY, d.mD);
        ScalarMatrix inners = fs->k(d.mX, d.mY);
        
        Real error = (inners - Eigen::Identity<Scalar>(inners.rows(),inners.rows())).squaredNorm();
        std::cerr << "Orthonormalization [1] orthonormality error is " << error << std::endl;
        
        
        ScalarMatrix x = *d.mX->template as<Dense<Scalar>>() * d.mY * d.mD.asDiagonal() * d.mY.adjoint() * d.mX->template as<Dense<Scalar>>()->adjoint() ;
        ScalarMatrix y = mX * mY * mD.asDiagonal() * mY.adjoint() * mX.adjoint();
            
        Real error2 = (x - y ).squaredNorm();
        std::cerr << "Orthonormalization [2] reconstruction error is " << error2 << std::endl;
        
        Real threshold = mD.squaredNorm() * Eigen::NumTraits<Scalar>::epsilon();
        std::cerr << "Threshold is " << threshold << std::endl;
        return error > threshold || error2 > threshold ? 1 : 0;
    }
    
    int test_orthonormalization_real_positive(std::deque<std::string> &/*args*/) {
        return test_orthonormalization<double>(true);
    }
    
    int test_orthonormalization_real(std::deque<std::string> &/*args*/) {
        return test_orthonormalization<double>();
    }
    int test_orthonormalization_complex(std::deque<std::string> &/*args*/) {
        return test_orthonormalization<std::complex<double>>();
    }
}

#include "main-tests.inc"
DEFINE_TEST("orthonormalization/real-positive", test_orthonormalization_real_positive);
DEFINE_TEST("orthonormalization/real", test_orthonormalization_real);
DEFINE_TEST("orthonormalization/complex", test_orthonormalization_complex);

