#include <iostream>
#include <complex>

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
        
        
        if (name == "toy") {
            // --- Problem definition
            
            double lambda = 1;
            
            int n = 2;
            int r = 2;
            
            Eigen::MatrixXd gramMatrix(n, n);
            gramMatrix << 1, 0, 0, 1;
            
            Eigen::MatrixXd alpha(n, r);
            alpha << .5, .2, 
            .1, 1.;
            
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