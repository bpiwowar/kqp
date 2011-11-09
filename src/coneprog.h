#ifndef _KQP_CONEPROG_H_
#define _KQP_CONEPROG_H_

#include <vector>
#include <Eigen/Core>

namespace kqp {
    namespace cvxopt {
        /** Problem dimensions */
        struct Dimensions {
            /// Quadrant \f$\mathbb{R}^{+l}\f$
            int l;
            /// A list with the dimensions of the second-order cones (positive integers)
            std::vector<int> q;
            /// a list with the dimensions of the positive semidefinite cones (nonnegative integers)
            std::vector<int> s;
            
        };
        
        /// Options to be used for the coneq 
        struct ConeQPOptions {
            /** Use corrections or not */
            bool useCorrection;
            
            /** Debug flag */
            bool DEBUG;
            
            /** Use Mehrotra correction or not. */
            bool correction;
            
            /** Show progress */
            bool show_progress;
            
            /** Maximum number of iterations */
            int maxiters;
            
            double abstol;
            double reltol;
            double feastol;
            
            int refinement;
            
            ConeQPOptions();
        };
        
        
        struct ConeQPInitVals {
            Eigen::VectorXd x;
            Eigen::VectorXd  y;
            Eigen::VectorXd  s;
            Eigen::VectorXd z;
        };
        
        
        struct ConeQPReturn : public ConeQPInitVals {
            
            std::string status;
            
            double gap;
            double relative_gap;
            double primal_objective;
            double dual_objective;
            
            double primal_infeasibility;
            double dual_infeasibility;
            
            double primal_slack;
            double dual_slack;
            
            int iterations;
            
            ConeQPReturn();
            
        };
        
        /*
         Nesterov-Todd scaling matrix
         - W['dnl']: positive vector
         - W['dnli']: componentwise inverse of W['dnl']
         - W['d']: positive vector
         - W['di']: componentwise inverse of W['d']
         - W['v']: lists of 2nd order cone vectors with unit hyperbolic norms
         - W['beta']: list of positive numbers
         - W['r']: list of square matrices 
         - W['rti']: list of square matrices.  rti[k] is the inverse transpose
         of r[k].
         */
        struct ScalingMatrix {
            Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> d, di, dnl, dnli;
            
            std::vector<double> beta;
            std::vector<Eigen::MatrixXd> r, rti;
            std::vector<Eigen::MatrixXd> v;
        };
        
        
        // The KKTSolver
        class KKTSolver {
        public:
            virtual void solve(Eigen::VectorXd &x, Eigen::VectorXd &y, Eigen::VectorXd & z) const = 0;  
        };
        
        /// The KKT solver
        class KKTPreSolver {
        public:
            KKTSolver *get(const ScalingMatrix &w) {
                return 0;
            };
        };
        
        
    }
    
}

#endif
