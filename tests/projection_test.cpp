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
#include "probabilities.hpp"
#include "kernel_evd/dense_direct.hpp"

DEFINE_LOGGER(logger, "kqp.test.projection")

namespace kqp {
    namespace {
        /// Returns 0 
        int isApproxEqual(const std::string & name, const Eigen::MatrixXd &a, const Eigen::Matrix &b) {
            double error = (a - b).squaredNorm(); 
            if (error > EPSILON) {
                KQP_LOG_ERROR_F(logger, "Error for %s is too big (%g)", %name %error);
                return 1;
            }
            return 0;
        }
    }
    
    
    int simple_projection_test(std::deque<std::string> &args) {
        typedef DenseMatrix<double> FMatrix;
        Index dimension = 100;
        
        DenseDirectBuilder<double> kevd(dimension);
        
        for (int i = 0; i < 10; i++) {
            Eigen::VectorXd v = Eigen::VectorXd::Random(dimension);
            kevd.add(FMatrix(v));
        }
        
        Event<FMatrix> subspace(kevd);
        
        FMatrix v(Eigen::VectorXd::Random(dimension));
        
        FMatrix v1 = subspace.project(v, false);
        FMatrix v2 = subspace.project(v, true);
        
        // Check that v1 . v2 = 0
        double inner = v1.get_matrix().col(0).inner(v2.get_matrix().col(0));
        
        // Check that projecting in orthogonal subspaces leads to a null vector
        FMatrix v1_p = subspace.project(v1, true);        
        FMatrix v2_p = subspace.project(v2, false);
        
        // Check that v = v1 + v2
        return  inner < EPSILON 
        && v1_p.col(0).squaredNorm() < EPISLON
        && v2_p.col(0).squaredNorm() < EPISLON
        && (v1 + v2 - v).squaredNorm() < EPSILON ? 0 : 1;
        
    }
    
    
    int projection_test() {
        typedef DenseMatrix<double> FMatrix;
        typedef DirectDensityBuilder<double> DensityTracker;
        
        // From R script src/R/projections.R
        
        // generate(10,3,5,10)
        
        Eigen::VectorXd rhoVectors[5];
        rhoVector[0] << -0.363676, -1.626673,
        -0.5991677, 0.3897943, -1.208076, -0.1842525,
        -1.371331, 0.2945451, 0.01874617, -0.2564784;
        rhoVector[1] <<  -2.119061, -0.1951504,
        -0.5963106, 0.08934727, -1.265198, -2.185287,
        -0.674866, 0.4829785, -0.9549439, 0.9255213;
        rhoVector[2] << -1.435514, 0.9685663,
        -0.324544, 0.3620872, -0.07794607, -0.651563,
        -1.759087, -1.379944, -1.853740, 0.1849260;
        rhoVector[3] << 0.5058193, -0.3012087,
        1.367954, -0.6776146, 0.6552276, -0.4006375, 2.137767,
        -0.02881534, -0.3345566, 0.2325252;
        rhoVector[4] << -0.8303227, 0.7356907,
        -0.4561763, -0.4812086, 1.216126, 0.5627448, -1.237594,
        1.066376, -1.246320, 0.3401156;
        
        Eigen::VectorXd sbVectors[3];
        sbVectors[0] << 0, 0, -1.051639,
        -0.4303878, -1.479827, 0, 0, 1.172706, 1.522586, 0;
        sbVectors[1] << 0, 2.106161, 0.5470484,
        0.7120943, -0.6458861, 0, 0, 0, 0.2273595, 0;
        sbVectors[2] <<  -0.4163547, 0, 0,
        1.155348, 0.5949573, 0, 0, 0, 0.06954478, -0.1914823;
        
        Eigen::MatrixXd wanted_fpRho(10, 10);
        wanted_fpRho << 0.002484600, 0.001264658, 0.008527885,
        -0.003111324, 0.007599663, 0, 0, -0.009143345,
        -0.01214978, 0.001142672, 0.001264658, 0.1912327,
        0.1497452, 0.1021027, 0.08037021, 0, 0, -0.1115959,
        -0.1244585, 0.0005816187, 0.008527885, 0.1497452,
        0.1902747, 0.08891785, 0.1549090, 0, 0, -0.1688076,
        -0.2044313, 0.003921991, -0.003111324, 0.1021027,
        0.08891785, 0.06869129, 0.06093878, 0, 0, -0.06958141,
        -0.07879957, -0.001430904, 0.007599663, 0.08037021,
        0.1549090, 0.06093878, 0.1531011, 0, 0, -0.1494643,
        -0.1866508, 0.0034951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -0.009143345, -0.1115959,
        -0.1688076, -0.06958141, -0.1494643, 0, 0, 0.1559187,
        0.1919180, -0.004205043, -0.01214978, -0.1244585,
        -0.2044313, -0.07879957, -0.1866508, 0, 0, 0.1919180,
        0.2377714, -0.005587706, 0.001142672, 0.0005816187,
        0.003921991, -0.001430904, 0.0034951, 0, 0,
        -0.004205043, -0.005587706, 0.0005255173;
        Eigen::MatrixXd wanted_pRho(10, 10);
        wanted_pRho << 0.01323191, 0.02199601, 0.03660284, -0.01663881,
        0.01781341, 0, 0, -0.03444576, -0.04455844, 0.00608538,
        0.02199601, 0.224301, 0.1773571, 0.0635406, 0.06737317,
        0, 0, -0.1328087, -0.1518933, 0.01011601, 0.03660284,
        0.1773571, 0.2023740, 0.02236447, 0.1132569, 0, 0,
        -0.1743023, -0.2132741, 0.01683372, -0.01663881,
        0.0635406, 0.02236447, 0.07005286, 0.01253747, 0, 0,
        -0.0065353, 0.001153288, -0.007652222, 0.01781341,
        0.06737317, 0.1132569, 0.01253747, 0.08863078, 0, 0,
        -0.1067815, -0.1343425, 0.00819242, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.03444576,
        -0.1328087, -0.1743023, -0.0065353, -0.1067815, 0, 0,
        0.1559019, 0.1938326, -0.01584167, -0.04455844,
        -0.1518933, -0.2132741, 0.001153288, -0.1343425, 0, 0,
        0.1938326, 0.2427089, -0.02049252, 0.00608538,
        0.01011601, 0.01683372, -0.007652222, 0.00819242, 0, 0,
        -0.01584167, -0.02049252, 0.002798679;
        Eigen::MatrixXd wanted_ofpRho(10, 10);
        wanted_ofpRho << 0.1677124, -0.02190436, 0.06881097, -0.01415170,
        0.05881613, 0.1098779, 0.1442587, -0.003188799,
        0.1201611, -0.05109014, -0.02190436, 0.07384375,
        -0.009091964, -0.01798430, 0.05015113, 0.01431349,
        -0.01790235, -0.01058087, -0.03984824, 0.01218393,
        0.06881097, -0.009091964, 0.05892167, -0.02172668,
        0.02909253, 0.01719707, 0.1167293, -0.003245499,
        0.0411081, -0.008159737, -0.01415170, -0.01798430,
        -0.02172668, 0.02323977, -0.0372862, -0.01144159,
        -0.04295253, -0.01711512, 0.004199208, -0.006981836,
        0.05881613, 0.05015113, 0.02909253, -0.0372862,
        0.09229287, 0.07498837, 0.05613238, 0.02186651,
        0.01116694, -0.008077133, 0.1098779, 0.01431349,
        0.01719707, -0.01144159, 0.07498837, 0.1264308,
        0.02923664, 0.01005658, 0.06186913, -0.04421775,
        0.1442587, -0.01790235, 0.1167293, -0.04295253,
        0.05613238, 0.02923664, 0.2552913, 0.007830574,
        0.1056560, -0.01202921, -0.003188799, -0.01058087,
        -0.003245499, -0.01711512, 0.02186651, 0.01005658,
        0.007830574, 0.06503791, 0.003377569, 0.01156652,
        0.1201611, -0.03984824, 0.0411081, 0.004199208,
        0.01116694, 0.06186913, 0.1056560, 0.003377569,
        0.1122451, -0.03718965, -0.05109014, 0.01218393,
        -0.008159737, -0.006981836, -0.008077133, -0.04421775,
        -0.01202921, 0.01156652, -0.03718965, 0.02498447;
        Eigen::MatrixXd wanted_opRho(10, 10);
        wanted_opRho << 0.1768022, -0.01659576, 0.0766019, 0.01088423,
        0.06801152, 0.1223934, 0.1427954, -0.00838297,
        0.1285431, -0.06075694, -0.01659576, 0.02201019,
        -0.02467433, -0.01781188, 0.02665924, 0.01628338,
        -0.02286712, 0.02185967, -0.01300314, 0.006724413,
        0.0766019, -0.02467433, 0.0691711, 0.01343174,
        0.014832, 0.01582331, 0.1190086, 0.004909633,
        0.06220671, -0.01684048, 0.01088423, -0.01781188,
        0.01343174, 0.01614296, -0.02680593, -0.01060353,
        0.0004223457, -0.02361138, 0.005972834, -0.007384271,
        0.06801152, 0.02665924, 0.014832, -0.02680593,
        0.08872135, 0.08124367, 0.07560228, 0.04615161,
        0.05335062, -0.01457797, 0.1223934, 0.01628338,
        0.01582331, -0.01060353, 0.08124367, 0.1423619,
        0.03292067, 0.01532028, 0.07509404, -0.05040112,
        0.1427954, -0.02286712, 0.1190086, 0.0004223457,
        0.07560228, 0.03292067, 0.2874598, 0.02189307,
        0.1389346, -0.02257789, -0.00838297, 0.02185967,
        0.004909633, -0.02361138, 0.04615161, 0.01532028,
        0.02189307, 0.06598815, -0.009252158, 0.01580156,
        0.1285431, -0.01300314, 0.06220671, 0.005972834,
        0.05335062, 0.07509404, 0.1389346, -0.009252158,
        0.1036325, -0.04005793, -0.06075694, 0.006724413,
        -0.01684048, -0.007384271, -0.01457797, -0.05040112,
        -0.02257789, 0.01580156, -0.04005793, 0.02770984 }, 10,
    0;
    
    int code = 0;
    
    // Create sb and rho
    DensityTracker rhoTracker;
    
    for (int i = 0; i < rhoVectors.size(); i++) {
        rhoTracker.add(rhoVectors[i]);
        Density rho(rhoTracker);
        
        DensityTracker sbTracker();
        sbTracker.removeMap();
        sbTracker.setSelector(new ThresholdSelector(1e-16, false));
        for (int j = 0; j < sbVectors.size(); j++) {
            sbTracker.add(FMatrix(sbVectors[j])));
            Subspace sb = new Subspace(sbTracker);
            sb.normalise();
            
            // Projection
            code |= isApproxEqual("projection of rho onto A",
                                  sb.projectLocal(rho, false, false, null).getMatrix(), wanted_pRho);
            // Orthogonal projection
            code |=isApproxEqual("projection of rho onto orth A",
                                 sb.projectLocal(rho, true, false, null).getMatrix(), wanted_opRho);
            
            // Fuzzy projection
            code |=isApproxEqual("fuzzy projection of rho onto A",
                                 sb.projectLocal(rho, false, true, null).getMatrix(), wanted_fpRho);
            code |=isApproxEqual("fuzzy projection of rho onto orth A",
                                 sb.projectLocal(rho, true, true, null).getMatrix(), wanted_ofpRho);
            return code;
            
        }
    }
