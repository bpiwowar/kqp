 #include <iostream>
#include <complex>

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include "kqp.h"
#include "probabilities.h"
#include "coneprog.h"
#include "test.h"

DEFINE_LOGGER(logger, "kqp.test.kqp-qp-solver")


namespace kqp {
    
    class KMult : public cvxopt::Matrix {
        Index n, r;
        const Eigen::MatrixXd &g;
    public:
        KMult(Index n, Index r, const Eigen::MatrixXd &g) : n(n), r(r), g(g) {}
        virtual void mult(const Eigen::VectorXd &x, Eigen::VectorXd &y, bool trans = false) const {
            y.resize(x.rows());
            for(int i = 0; i < r; i++) 
                if (trans)
                    y.segment(i*n, n) = g.adjoint() * x.segment(i*n,n);
            
                else                    
                    y.segment(i*n, n) = g * x.segment(i*n,n);
        }
        virtual Index rows() const { return g.rows(); }
        virtual Index cols() const { return g.cols(); }
        
    };
    
    /**
     * The matrix G
     */
    class QPConstraints : public cvxopt::Matrix {
        Index n, r;
    public:
        QPConstraints(Index n, Index r) : n(n), r(r) {}
        
        virtual void mult(const Eigen::VectorXd &x, Eigen::VectorXd &y, bool trans = false) const {
            // assumes y and x are two different things
            if (!trans) {
                y.resize(2*n*r);
                
                for(int i = 0; i < r; i++) {
                    y.segment(i*n, n) = - x.segment(i*n, n) - x.segment(n*r, n);
                    y.segment((i+r)*n, n) = x.segment(i*n, n) - x.segment(n*r, n);
                }
            } else {
                y.resize(n * (r+1));
                y.segment(n * r, n).array().setConstant(0);
               
                for(int i = 0; i < r; i++) {
                    y.segment(i*n, n) = - x.segment(i*n, n) - x.segment((i+r)*n, n);
                    y.segment(n * r, n) += x.segment(i*n, n) +  x.segment((i+r)*n, n);
                }
                
            }
            
            
        }
        virtual Index rows() const { return 2*n*r; }
        virtual Index cols() const { return n * (r+1); }
    };
    
    int kqp_qp_solver_test(int argc, const char **argv) {
        KQP_LOG_INFO(logger, "Starting qp solver tests");
        std::string name = argv[0];
        
        if (name == "kkt-solver") {
            int n = 2;
            int r = 2;
            Eigen::MatrixXd g(n,n);
            cvxopt::ScalingMatrix w;
            w.d.resize(2*r*n);
            Eigen::VectorXd x(n*(r+1)), y, z(2*n*r);
            x <<  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ;
            z <<  0.000000000000000e+00 ,  0.000000000000000e+00 ,  0.000000000000000e+00 ,  0.000000000000000e+00 ,  0.000000000000000e+00 ,  0.000000000000000e+00 ,  0.000000000000000e+00 ,  0.000000000000000e+00 ;
            g <<  1.000000000000000e+00 ,  0.000000000000000e+00 ,  0.000000000000000e+00 ,  1.000000000000000e+00 ;
            w.d.diagonal() <<  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ,  1.000000000000000e+00 ;
            
            // Solve
            KQP_KKTPreSolver kkt_presolver(g);
            boost::shared_ptr<cvxopt::KKTSolver> kktSolver(kkt_presolver.get(w));
            kktSolver->solve(x,y,z);
            
            
            // Solution
            Eigen::VectorXd s_x(n*(r+1)), s_z(2*n*r);
            s_x <<  3.333333333333334e-01 ,  3.333333333333334e-01 ,  3.333333333333334e-01 ,  3.333333333333334e-01 ,  2.500000000000002e-01 ,  2.500000000000002e-01 ;
            s_z <<  -5.833333333333335e-01 ,  -5.833333333333335e-01 ,  -5.833333333333335e-01 ,  -5.833333333333335e-01 ,  8.333333333333318e-02 ,  8.333333333333318e-02 ,  8.333333333333318e-02 ,  8.333333333333318e-02 ;
            
            std::cerr << "x - x* = " << x - s_x;
            std::cerr << "z - z* = " << z - s_z;
            
            std::cerr << "Average error (x): " << (x - s_x).norm() / (double)x.rows();
            std::cerr << "Average error (z): " << (z - s_z).norm() / (double)z.rows();
            return 0;
            
        } else if (name == "kk") {
            int n = 2;
            int r = 2;
            
            // Real solution is given by
            // (K + G' W^-2 G) x = a + G' W^-2 c
            
            // With
            // G' W^-2 G =
            // W_1^2+W_r+1^2               0
            //           ...               .
            //             W_r^2+W_2r^2    0
            // 0     ...        0      \sum W_i^2
            
            // Test of the KKT solver
            Eigen::MatrixXd g(n, n);
            g << 1, 0, 0, 1;
            KQP_KKTPreSolver kkt_presolver(g);

            cvxopt::ScalingMatrix w;
            w.d.resize(2*r*n);
            w.d.diagonal().setConstant(-1.);
            w.di = w.d.inverse();
            
            boost::shared_ptr<cvxopt::KKTSolver> kktSolver(kkt_presolver.get(w));
            
            Eigen::VectorXd x(n*r+n), y, z(2*n*r);
            
            x << 1, 2, 3, 4, 5, 6;
            z << 0,0,0,0, 0,0,0,0;
            
            kktSolver->solve(x,y,z);
            std::cerr << x;
            
            KQP_LOG_ASSERT(logger, std::abs(x[0] - 1./3.) < EPSILON, "x_1 = " << convert(x[0]) << "!= 1/3");
            KQP_LOG_ASSERT(logger, std::abs(x[1] - 2./3.) < EPSILON, "x_2 = " << convert(x[0]) << "!= 2/3");
            return 0;
        }
        
        if (name == "toy") {
            // --- Problem definition
            
            double lambda = 1;
            
            int n = 2;
            int r = 2;
            
            Eigen::MatrixXd gramMatrix(n, n);
            gramMatrix << 1, 0, 
                          0, 1;
            
            Eigen::MatrixXd alpha(n, r);
            alpha << 1, 0, 
                     0, 1;
            
            // --- Prepare the data structures
            
            Eigen::VectorXd c(n*r + n);
            for(int i = 0; i < r; i++)
                c.segment(i*n, n) = - 2 * gramMatrix * alpha.block(0, i, n, 1);
            c.segment(r*n,n).setConstant(lambda);
            
            
            // --- Solve
            kqp::cvxopt::ConeQPReturn result;
            
            QPConstraints G(n, r);
            KQP_KKTPreSolver kkt_presolver(gramMatrix);
            cvxopt::ConeQPOptions options;
            
            cvxopt::coneqp(KMult(n,r, gramMatrix), c, result, 
                           false, cvxopt::Dimensions(),
                            &G,0,0,0,
                            &kkt_presolver,
                            options
                            );
            std::cout << "Result " << result.x << std::endl;
            return 0;
        }
        
        
        BOOST_THROW_EXCEPTION(illegal_argument_exception()
                              << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
        
        return 0;
    }
    
}