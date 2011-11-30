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
    
    /**
     Class that knows how to multiply by
      
     \f$ \left(\begin{array}{ccc}
     K\\
     & \ddots\\
     &  & K
     \end{array}\right)
     \f$
     
     */
    class KMult : public cvxopt::Matrix {
        Index n, r;
        const Eigen::MatrixXd &g;
    public:
        KMult(Index n, Index r, const Eigen::MatrixXd &g) : n(n), r(r), g(g) {}
        
        virtual void mult(const Eigen::VectorXd &x, Eigen::VectorXd &y, bool trans = false) const {
            y.resize(x.rows());
            y.tail(n).setZero();
            for(int i = 0; i < r; i++) 
                if (trans)
                    y.segment(i*n, n).noalias() = g.adjoint() * x.segment(i*n,n);
            
                else                    
                    y.segment(i*n, n).noalias() = g * x.segment(i*n,n);
        }
        virtual Index rows() const { return g.rows(); }
        virtual Index cols() const { return g.cols(); }
        
    };
    
    /**
     * Multiplying with matrix G 
     * <pre>
     * -Id           -Id
     *     ...       -Id
     *          -Id  -Id
     *  Id           -Id
     *     ...       -Id
     *           Id  -Id
     * </pre>
     * of size 2nr x n (r+1)
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
                    y.segment(i*n, n) = - x.segment(i*n, n) + x.segment((i+r)*n, n);
                    y.segment(n * r, n) -= x.segment(i*n, n) +  x.segment((i+r)*n, n);
                }
                
            }
            
            
        }
        virtual Index rows() const { return 2*n*r; }
        virtual Index cols() const { return n * (r+1); }
    };
    
    /// Solve a QP system
    void solve_qp(int n, int r, double lambda, const Eigen::MatrixXd &gramMatrix, const Eigen::MatrixXd &alpha, kqp::cvxopt::ConeQPReturn &result) {
        Eigen::VectorXd c(n*r + n);
        for(int i = 0; i < r; i++) {
            c.segment(i*n, n) = - 2 * gramMatrix * alpha.block(0, i, n, 1);
        }
        c.segment(r*n,n).setConstant(lambda);
        KQP_LOG_DEBUG(logger, "q = " << convert(c.adjoint()));
        
        QPConstraints G(n, r);
        KQP_KKTPreSolver kkt_presolver(gramMatrix);
        cvxopt::ConeQPOptions options;
        
        cvxopt::coneqp(KMult(n,r, gramMatrix), c, result, 
                       false, cvxopt::Dimensions(),
                       &G,0,0,0,
                       &kkt_presolver,
                       options
                       );

    }
    
#include "generated/kkt_solver.cpp"
#include "generated/qp_solver.cpp"
    
    int kqp_qp_solver_test(int argc, const char **argv) {
        KQP_LOG_INFO(logger, "Starting qp solver tests");
        std::string name = argv[0];
        
        if (name == "kkt-solver-simple") 
            return kkt_test_simple();

        if (name == "kkt-solver-diagonal-g")
            return kkt_test_diagonal_g();

        if (name == "kkt-solver-diagonal-d")
            return kkt_test_diagonal_d();
        
        if (name == "kkt-solver-random")
            return kkt_test_random();

        
        
        if (name == "simple") 
            return qp_test_simple();
        
        if (name == "random") 
            return qp_test_random();
        
        BOOST_THROW_EXCEPTION(illegal_argument_exception()
                              << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
        
        return 0;
    }
    
}