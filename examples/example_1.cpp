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

#include <kqp/feature_matrix/dense.hpp>
#include <kqp/kernel_evd/incremental.hpp>
#include <kqp/probabilities.hpp>

int main(int, const char**) {
    
    // --- Compute a density at random
    
    // Definitions
    using namespace kqp;
    typedef Eigen::Matrix<double,Dynamic,Dynamic> Matrix;
    int dim = 10;
    
    // Feature space
    DenseSpace<double> space(DenseSpace<double>::create(dim));
    
    // Creating an incremental builder
    IncrementalKernelEVD<double> kevd(space);
    
    // Add 10 vectors with $\alpha_i=1$
    for(int i = 0; i < 10; i++) {
        // Adds a random $\varphi_i$
        Matrix m = Matrix::Random(dim, 1);
        kevd.add(Dense<double>::create(m));
    }
    
    // Get the result $\rho \approx X Y D Y^\dagger X^\dagger$
    Decomposition<double> d = kevd.getDecomposition();

    // --- Compute a kEVD for a subspace
    
    IncrementalKernelEVD<double> kevd_event(space);
    for(int i = 0; i < 3; i++) {
        // Adds a random $\varphi_i$
        Matrix m = Matrix::Random(dim, 1);
        kevd_event.add(Dense<double>::create(m));
    }

    
    // --- Compute some probabilities
    
    // Setup densities and events
    Density<double> rho(kevd);
    rho.normalize();
    Event<double> event(kevd_event);
    
    // Compute the probability
    std::cout << "Probability = " << rho.probability(event) << std::endl;

    // Conditional probability
    Density<double> rho_cond = event.project(kevd).normalize(); 
    std::cout << "Entropy rho/E = " << rho_cond.entropy() << std::endl;
    
    // Conditional probability (orthogonal event)
    Density<double> rho_cond_orth = event.project(kevd, true).normalize();
    std::cout << "Entropy rho/not E = " << rho_cond_orth.entropy() << std::endl;
    
    return 0;
}
