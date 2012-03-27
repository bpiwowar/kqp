#include <kqp/feature_matrix/dense.hpp>
#include <kqp/kernel_evd/incremental.hpp>
#include <kqp/probabilities.hpp>

int main(int, const char**) {
    
    // --- Compute a density at random
    
    // Definitions
    using namespace kqp;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    int dim = 10;
    
    // Creating an incremental builder
    IncrementalKernelEVD<DenseMatrix<double>> kevd;
    
    // Add 10 vectors with $\alpha_i=1$
    for(int i = 0; i < 10; i++) {
        // Adds a random $\varphi_i$
        Matrix m = Matrix::Random(dim, 1);
        kevd.add(DenseMatrix<double>(m));
    }
    
    // Get the result $\rho \approx X Y D Y^\dagger X^\dagger$
    DenseMatrix<double> mX;
    typename AltDense<double>::type mY;
    Eigen::Matrix<double, Eigen::Dynamic,1>  mD;
    
    kevd.get_decomposition(mX, mY, mD);

    // --- Compute a kEVD for a subspace
    
    IncrementalKernelEVD<DenseMatrix<double>> kevd_event;
    for(int i = 0; i < 3; i++) {
        // Adds a random $\varphi_i$
        Matrix m = Matrix::Random(dim, 1);
        kevd_event.add(DenseMatrix<double>(m));
    }

    
    // --- Compute some probabilities
    
    // Setup densities and events
    Density<DenseMatrix<double>> rho(kevd);
    Event<DenseMatrix<double>> event(kevd_event);
    
    // Compute the probability
    std::cout << "Probability = " << rho.probability(event) << std::endl;

    // Conditional probability
    Density<DenseMatrix<double>> rho_cond = event.project(kevd).normalize(); 

    // Conditional probability (orthogonal event)
    Density<DenseMatrix<double>> rho_cond_orth = event.project(kevd, true).normalize();
    
    return 0;
}
