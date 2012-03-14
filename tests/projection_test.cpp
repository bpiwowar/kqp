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

#include <deque>

#include <kqp/probabilities.hpp>
#include <kqp/trace.hpp>

#include <kqp/kernel_evd/dense_direct.hpp>

DEFINE_LOGGER(logger, "kqp.test.projection")

namespace kqp {
    namespace {
        /// Returns 0 
        int isApproxEqual(const std::string & name, const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
            double error = (a - b).squaredNorm(); 
            if (error > EPSILON) {
                KQP_LOG_ERROR_F(logger, "Error for %s is too big (%g)", %name %error);
                return 1;
            }
            return 0;
        }
        
        int isApproxEqual(const std::string & name, const Density< DenseMatrix<double> > &a, const Eigen::MatrixXd &b) {
            return isApproxEqual(name, a.matrix().get_matrix(), b);
        }
    }
    
    
    int simple_projection_test(std::deque<std::string> &/*args*/) {
        typedef DenseMatrix<double> FMatrix;
        typedef ftraits<FMatrix>::ScalarMatrix ScalarMatrix;
        
        Index dimension = 100;
        
        DenseDirectBuilder<double> kevd(dimension);
        
        for (int i = 0; i < 10; i++) {
            Eigen::VectorXd v = Eigen::VectorXd::Random(dimension);
            ((KernelEVD<FMatrix>&)kevd).add(FMatrix(v));
        }
        
        Event<FMatrix> subspace(kevd);
        
        FMatrix v(Eigen::VectorXd::Random(dimension));
        
        Density<FMatrix> v1 = subspace.project(Density<FMatrix>(v, true), false);
        Density<FMatrix> v2 = subspace.project(Density<FMatrix>(v, true), true);
        
        // Check that v1 . v2 = 0
        ScalarMatrix inners = v1.inners(v2);
        
        // Check that projecting in orthogonal subspaces leads to a null vector
        Density<FMatrix> v1_p = subspace.project(v1, true);        
        Density<FMatrix> v2_p = subspace.project(v2, false);
        
        // Check that v = v1 + v2
        return  inners.squaredNorm() < EPSILON 
        && v1_p.matrix().get_matrix().squaredNorm() < EPSILON
        && v2_p.matrix().get_matrix().squaredNorm() < EPSILON
        && (v1.matrix().get_matrix() + v2.matrix().get_matrix() - v.get_matrix()).squaredNorm() < EPSILON 
        ? 0 : 1;
        
    }
    

    
    int projection_test(std::deque<std::string> &/*args*/) {
        typedef DenseMatrix<double> FMatrix;
        typedef DenseDirectBuilder<double> DensityTracker;
        
        // From R script src/R/projections.R
        // generate(10,3,5,10)
        
#include "generated/projections_R.cpp"
        
        int code = 0;
        
        // Create sb and rho
        DensityTracker rhoTracker(dimension);
        
        for (int i = 0; i < rhoVectorsCount; i++) 
            rhoTracker.add(FMatrix(rhoVectors[i]));
        Density<FMatrix> rho(rhoTracker);
        
        DensityTracker sbTracker(dimension);
        for (int j = 0; j < sbVectorsCount; j++) 
            sbTracker.add(FMatrix(sbVectors[j]));
        
        // Strict event
        Event<FMatrix> sb(sbTracker);
        
        
        // Fuzzy event
        Event<FMatrix> sb_fuzzy(sbTracker);
        
        sb_fuzzy.multiplyBy(1. / std::sqrt(sb_fuzzy.squaredNorm()));
        
        // Projection
        code |= isApproxEqual("projection of rho onto A",
                              sb.project(rho, false), wanted_pRho);
        // Orthogonal projection
        code |=isApproxEqual("projection of rho onto orth A",
                             sb.project(rho, true), wanted_opRho);
        
        // Fuzzy projection
        code |=isApproxEqual("fuzzy projection of rho onto A",
                             sb_fuzzy.project(rho, false), wanted_fpRho);

        // Fuzzy orthogonal projection
        code |=isApproxEqual("fuzzy projection of rho onto orth A",
                             sb_fuzzy.project(rho, true), wanted_ofpRho);
        
        
        return code;
        
    }
}

