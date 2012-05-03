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
#include <string>
#include <deque>

#include <kqp/probabilities.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include "tests_utils.hpp"

DEFINE_LOGGER(logger, "kqp.test.divergence");

namespace kqp {
    int test_orthonormalization(std::deque<std::string> &/*args*/) {
        int dim = 10;
        int n = 5;
        int r = 5;
        
        Eigen::MatrixXd mX = generateMatrix<double>(dim, dim).leftCols(n);
        Eigen::MatrixXd mY = Eigen::MatrixXd::Random(n, r);
        Eigen::VectorXd mD = Eigen::MatrixXd::Random(r,1);
        auto fs = DenseFeatureSpace<double>::create(dim);

        
        Density<double> d(fs, DenseMatrix<double>::create(mX), mY, mD, false);
        d.orthonormalize();
        Eigen::MatrixXd inners = fs.k(d.X(), d.Y());
        
        double error = (inners - Eigen::MatrixXd::Identity(inners.rows(),inners.rows())).squaredNorm();
        std::cerr << "Orthonormalization [1] error is " << error << std::endl;
        
        Eigen::MatrixXd x = *d.matrix()->as<DenseMatrix<double>>();
        Eigen::MatrixXd y = mX * mY * mD.asDiagonal();
        double error2 = (x * x.adjoint() - y * y.adjoint()).squaredNorm();
        std::cerr << "Orthonormalization [2] error is " << error2 << std::endl;

        double threshold = mD.squaredNorm() * EPSILON;
        std::cerr << "Threshold is " << threshold << std::endl;
        return error > threshold || error2 > threshold ? 1 : 0;
    }
}

#include "main-tests.inc"
DEFINE_TEST("orthonormalization", test_orthonormalization);

