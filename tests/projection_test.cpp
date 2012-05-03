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

#include <kqp/feature_matrix/dense.hpp>
#include <kqp/kernel_evd/dense_direct.hpp>

DEFINE_LOGGER(logger, "kqp.test.projection")

namespace kqp {
    namespace {
        /// Returns 0 
        int isApproxEqual(const std::string & name, const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
            double error = (a - b).squaredNorm();
            double rel_error = error / b.squaredNorm(); 
            KQP_LOG_INFO_F(logger, "Error and relative errors for %s are %g and %g", %name %error %rel_error);
            if (rel_error > EPSILON) {
                KQP_LOG_ERROR_F(logger, "Relative error for %s is too big (%g)", %name %rel_error);
                return 1;
            }
            return 0;
        }
        
        int isApproxEqual(const std::string & name, const Density< double > &a, const Eigen::MatrixXd &b) {
            auto _a = a.matrix();
            auto m = dynamic_cast<DenseMatrix<double>&>(*_a);
            KQP_MATRIX(double) op = m.getMatrix() * m.getMatrix().transpose();
            return isApproxEqual(name, op, b);
        }
    }
    
    
    template<typename Scalar>
    const Matrix<Scalar,Dynamic,Dynamic> getMatrix(const Density<Scalar> &d) {
        return dynamic_cast<DenseMatrix<double>&>(*d.matrix()).getMatrix();
    }
    
    int simple_projection_test(std::deque<std::string> &/*args*/) {
        KQP_SCALAR_TYPEDEFS(double);
        
        Index dimension = 100;
        
        FSpace fs(new DenseFeatureSpace<double>(2));
        
        DenseDirectBuilder<double> kevd(dimension);
        
        for (int i = 0; i < 10; i++) {
            Eigen::VectorXd v = Eigen::VectorXd::Random(dimension);
            ((KernelEVD<double> &)kevd).add(DenseMatrix<double>::create(v));
        }
        
        Event<double> subspace(kevd);
        
        FeatureMatrix<double> v(DenseMatrix<double>::create(Eigen::VectorXd::Random(dimension)));
        
        Density<double> v1 = subspace.project(Density<double>(fs, v, true), false);
        Density<double> v2 = subspace.project(Density<double>(fs, v, true), true);
        
        // Check that v1 . v2 = 0
        ScalarMatrix inners = v1.inners(v2);
        
        // Check that projecting in orthogonal subspaces leads to a null vector
        Density<double> v1_p = subspace.project(v1, true);        
        Density<double> v2_p = subspace.project(v2, false);
        
        // Check that v = v1 + v2
        return  inners.squaredNorm() < EPSILON 
        && getMatrix(v1_p).squaredNorm() < EPSILON
        && getMatrix(v2_p).squaredNorm() < EPSILON
        && (getMatrix(v1) + getMatrix(v2) - v->as<DenseMatrix<double>>().getMatrix()).squaredNorm() < EPSILON 
        ? 0 : 1;
        
    }
    
    
    
    int projection_test(std::deque<std::string> &/*args*/) {
        KQP_SCALAR_TYPEDEFS(double);
        typedef DenseDirectBuilder<double> DensityTracker;
        
        // From R script src/R/projections.R
        // generate(10,3,5,10)
        
#include "generated/projections_R.inc"
        
        int code = 0;
        
        // Create sb and rho
        DensityTracker rhoTracker(dimension);
        
        for (int i = 0; i < rhoVectorsCount; i++) 
            rhoTracker.add(DenseMatrix<double>::create(rhoVectors[i]));
        Density<double> rho(rhoTracker);
        
        rho.normalize();
        
        DensityTracker sbTracker(dimension);
        for (int j = 0; j < sbVectorsCount; j++) 
            sbTracker.add(DenseMatrix<double>::create(sbVectors[j]));
        
        // Strict event
        Event<double> sb(sbTracker);
        
        
        // Fuzzy event
        Event<double> sb_fuzzy(sbTracker, true);
        
        sb_fuzzy.multiplyBy(1. / sb_fuzzy.trace());
        
        for(int i = 0; i < 2; i++) {
            sb.setUseLinearCombination(i == 0);
            sb_fuzzy.setUseLinearCombination(i == 0);
            
            KQP_LOG_INFO_F(logger, "=== Use linear combination = %s", %(i == 0 ? "yes" : "no"));
            
            // Projection
            Density<double> pRho = sb.project(rho, false);
            code |= isApproxEqual("projection of rho onto A", pRho.normalize(), wanted_pRho);
            
            // Orthogonal projection
            Density<double> opRho = sb.project(rho, true);
            code |=isApproxEqual("projection of rho onto orth A", opRho.normalize(), wanted_opRho);
            
            // Fuzzy projection
            Density<double> fpRho = sb_fuzzy.project(rho, false);
            code |=isApproxEqual("fuzzy projection of rho onto A", fpRho.normalize(), wanted_fpRho);
            
            // Fuzzy orthogonal projection
            Density<double> ofpRho = sb_fuzzy.project(rho, true);
            code |=isApproxEqual("fuzzy projection of rho onto orth A", ofpRho.normalize(), wanted_ofpRho);
        }
        
        return code;
        
    }
}

#include "main-tests.inc"
DEFINE_TEST("simple", simple_projection_test);
DEFINE_TEST("normal", projection_test);
