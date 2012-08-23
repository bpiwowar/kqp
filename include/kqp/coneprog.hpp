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

#ifndef _KQP_CONEPROG_H_
#define _KQP_CONEPROG_H_

#include <vector>

#include <kqp/kqp.hpp>
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
            
            inline Dimensions() : l(-1) {}
        };
        
        
        /// Options to be used for the coneq 
        template<typename Scalar>
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
            
            Scalar abstol;
            Scalar reltol;
            Scalar feastol;
            
            int refinement;
            
            ConeQPOptions();
        };
        
        /// Status of the result
        enum Status {
            /// Converged
            OPTIMAL,
            
            // Singular KKT matrix
            SINGULAR_KKT_MATRIX,
            
            // Not converged
            NOT_CONVERGED
        };
        
        template<typename Scalar>
        struct ConeQPReturn {
            KQP_VECTOR(Scalar) x;
            KQP_VECTOR(Scalar) y;
            KQP_VECTOR(Scalar) s;
            KQP_VECTOR(Scalar) z;
            
            Status status;
            
            Scalar gap;
            Scalar relative_gap;
            Scalar primal_objective;
            Scalar dual_objective;
            
            Scalar primal_infeasibility;
            Scalar dual_infeasibility;
            
            Scalar primal_slack;
            Scalar dual_slack;
            
            int iterations;
            
            ConeQPReturn();
            
        };
        
        
        template<typename Scalar>
        class QPMatrix {
        public:    
            //! Computes Q * x and stores the result in y (y might be the same as x)
            virtual void mult(const KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &y, bool adjoint = false) const = 0;
            virtual  Eigen::MatrixXd::Index rows() const = 0;
            virtual  Eigen::MatrixXd::Index cols() const = 0;
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
         - W['rti']: list of square matrices.  rti[k] is the inverse adjoint
         of r[k].
         */
        template<typename Scalar>
        struct ScalingMatrix {
            KQP_VECTOR(Scalar) d, di, dnl, dnli;
            
            std::vector<Scalar> beta;
            std::vector<KQP_MATRIX(Scalar)> r, rti;
            std::vector<KQP_MATRIX(Scalar)> v;
        };
        
        
        // The KKTSolver
        template<typename Scalar>
        class KKTSolver {
        public:
            virtual ~KKTSolver() {}
            virtual void solve(KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &y, KQP_VECTOR(Scalar) & z) const = 0;  
        };
        
        /// The KKT solver
        template<typename Scalar>
        class KKTPreSolver {
        public:
            virtual KKTSolver<Scalar> *get(const ScalingMatrix<Scalar> &w) = 0;
        };
        
        
        /**  Solves a pair of primal and dual convex quadratic cone programs
         
         Adapted from cvxopt - restriction to the first order cones (commented the non first order parters for latter inclusion)
         
         minimize    \f$ \frac{1}{2} x^\top P x + q^\top*x \f$
         subject to  \f$ Gx + s = h\f$
         \f$ A*x = b\f$
         \f$s >= 0\f$
         
         maximize    -(1/2)*(q + G'*z + A'*y)' * pinv(P) * (q + G'*z + A'*y)
         - h'*z - b'*y 
         subject to  q + G'*z + A'*y in range(P)
         z >= 0.
         
         The inequalities are with respect to a cone C defined as the Cartesian
         product of N + M + 1 cones:
         
         C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.
         
         The first cone C_0 is the nonnegative orthant of dimension ml.  
         The next N cones are 2nd order cones of dimension mq[0], ..., mq[N-1].
         The second order cone of dimension m is defined as
         
         { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.
         
         The next M cones are positive semidefinite cones of order ms[0], ...,
         ms[M-1] >= 0.  
         
         
         
         @param P is a dense or sparse 'd' matrix of size (n,n) with the lower 
         triangular part of the Hessian of the objective stored in the 
         lower triangle.  Must be positive semidefinite.
         
         @param q is a dense 'd' matrix of size (n,1).
         
         dims is a dictionary with the dimensions of the components of C.  
         It has three fields.
         -dims.l = ml, the dimension of the nonnegative orthant C_0.
         (ml >= 0.)
         - dims['q'] = mq = [ mq[0], mq[1], ..., mq[N-1] ], a list of N 
         integers with the dimensions of the second order cones 
         C_1, ..., C_N.  (N >= 0 and mq[k] >= 1.)
         - dims['s'] = ms = [ ms[0], ms[1], ..., ms[M-1] ], a list of M  
         integers with the orders of the semidefinite cones 
         C_{N+1}, ..., C_{N+M}.  (M >= 0 and ms[k] >= 0.)
         The default value of dims = {'l': G.size[0], 'q': [], 's': []}.
         
         G is a dense or sparse 'd' matrix of size (K,n), where
         
         K = ml + mq[0] + ... + mq[N-1] + ms[0]**2 + ... + ms[M-1]**2.
         
         Each column of G describes a vector 
         
         v = ( v_0, v_1, ..., v_N, vec(v_{N+1}), ..., vec(v_{N+M}) ) 
         
         in V = R^ml x R^mq[0] x ... x R^mq[N-1] x S^ms[0] x ... x S^ms[M-1]
         stored as a column vector
         
         [ v_0; v_1; ...; v_N; vec(v_{N+1}); ...; vec(v_{N+M}) ].
         
         Here, if u is a symmetric matrix of order m, then vec(u) is the 
         matrix u stored in column major order as a vector of length m**2.
         We use BLAS unpacked 'L' storage, i.e., the entries in vec(u) 
         corresponding to the strictly upper triangular entries of u are 
         not referenced.
         
         h is a dense 'd' matrix of size (K,1), representing a vector in V,
         in the same format as the columns of G.
         
         A is a dense or sparse 'd' matrix of size (p,n).   The default
         value is a sparse 'd' matrix of size (0,n).
         
         b is a dense 'd' matrix of size (p,1).  The default value is a 
         dense 'd' matrix of size (0,1).
         
         initvals is a dictionary with optional primal and dual starting 
         points initvals['x'], initvals['s'], initvals['y'], initvals['z'].
         - initvals['x'] is a dense 'd' matrix of size (n,1).   
         - initvals['s'] is a dense 'd' matrix of size (K,1), representing
         a vector that is strictly positive with respect to the cone C.  
         - initvals['y'] is a dense 'd' matrix of size (p,1).  
         - initvals['z'] is a dense 'd' matrix of size (K,1), representing
         a vector that is strictly positive with respect to the cone C.
         A default initialization is used for the variables that are not
         specified in initvals.
         
         It is assumed that rank(A) = p and rank([P; A; G]) = n.
         
         The other arguments are normally not needed.  They make it possible
         to exploit certain types of structure, as described below.
         
         
         Output arguments.
         
         Returns a dictionary with keys 'status', 'x', 's', 'z', 'y',
         'primal objective', 'dual objective', 'gap', 'relative gap', 
         'primal infeasibility', 'dual infeasibility', 'primal slack',
         'dual slack', 'iterations'. 
         
         The 'status' field has values 'optimal' or 'unknown'.  'iterations'
         is the number of iterations taken.
         
         If the status is 'optimal', 'x', 's', 'y', 'z' are an approximate 
         solution of the primal and dual optimality conditions   
         
         G*x + s = h,  A*x = b  
         P*x + G'*z + A'*y + q = 0 
         s >= 0,  z >= 0
         s'*z = 0.
         
         If the status is 'unknown', 'x', 'y', 's', 'z' are the last 
         iterates before termination.  These satisfy s > 0 and z > 0, 
         but are not necessarily feasible.
         
         The values of the other fields are defined as follows.
         
         - 'primal objective': the primal objective (1/2)*x'*P*x + q'*x.
         
         - 'dual objective': the dual objective 
         
         L(x,y,z) = (1/2)*x'*P*x + q'*x + z'*(G*x - h) + y'*(A*x-b).
         
         - 'gap': the duality gap s'*z.  
         
         - 'relative gap': the relative gap, defined as 
         
         gap / -primal objective 
         
         if the primal objective is negative, 
         
         gap / dual objective
         
         if the dual objective is positive, and NULL otherwise.
         
         - 'primal infeasibility': the residual in the primal constraints,
         defined as the maximum of the residual in the inequalities 
         
         || G*x + s + h || / max(1, ||h||) 
         
         and the residual in the equalities 
         
         || A*x - b || / max(1, ||b||).
         
         
         - 'dual infeasibility': the residual in the dual constraints,
         defined as 
         
         || P*x + G'*z + A'*y + q || / max(1, ||q||).
         
         
         - 'primal slack': the smallest primal slack, sup {t | s >= t*e }, 
         where 
         
         e = ( e_0, e_1, ..., e_N, e_{N+1}, ..., e_{M+N} )
         
         is the identity vector in C.  e_0 is an ml-vector of ones, 
         e_k, k = 1,..., N, is the unit vector (1,0,...,0) of length
         mq[k], and e_k = vec(I) where I is the identity matrix of order
         ms[k].
         
         - 'dual slack': the smallest dual slack, sup {t | z >= t*e }.
         
         If the exit status is 'optimal', then the primal and dual
         infeasibilities are guaranteed to be less than 
         solvers.options['feastol'] (default 1e-7).  The gap is less than 
         solvers.options['abstol'] (default 1e-7) or the relative gap is 
         less than solvers.options['reltol'] (default 1e-6).
         
         Termination with status 'unknown' indicates that the algorithm 
         failed to find a solution that satisfies the specified tolerances.
         In some cases, the returned solution may be fairly accurate.  If 
         the primal and dual infeasibilities, the gap, and the relative gap
         are small, then x, y, s, z are close to optimal.  
         
         
         Advanced usage.
         
         Three mechanisms are provided to express problem structure.
         
         1.  The user can provide a customized routine for solving linear 
         equations (`KKT systems')
         
         [ P   A'  G'    ] [ ux ]   [ bx ]
         [ A   0   0     ] [ uy ] = [ by ].
         [ G   0   -W'*W ] [ uz ]   [ bz ]
         
         W is a scaling matrix, a block diagonal mapping
         
         W*u = ( W0*u_0, ..., W_{N+M}*u_{N+M} )
         
         defined as follows.
         
         - For the 'l' block (W_0):
         
         W_0 = diag(d),
         
         with d a positive vector of length ml.
         
         - For the 'q' blocks (W_{k+1}, k = 0, ..., N-1):
         
         W_{k+1} = beta_k * ( 2 * v_k * v_k' - J )
         
         where beta_k is a positive scalar, v_k is a vector in R^mq[k]
         with v_k[0] > 0 and v_k'*J*v_k = 1, and J = [1, 0; 0, -I].
         
         - For the 's' blocks (W_{k+N}, k = 0, ..., M-1):
         
         W_k * u = vec(r_k' * mat(u) * r_k)
         
         where r_k is a nonsingular matrix of order ms[k], and mat(x) is
         the inverse of the vec operation.
         
         The optional argument kktsolver is a Python function that will be
         called as g = kktsolver(W).  W is a dictionary that contains
         the parameters of the scaling:
         
         - W['d'] is a positive 'd' matrix of size (ml,1).
         - W['di'] is a positive 'd' matrix with the elementwise inverse of
         W['d'].
         - W['beta'] is a list [ beta_0, ..., beta_{N-1} ]
         - W['v'] is a list [ v_0, ..., v_{N-1} ]
         - W['r'] is a list [ r_0, ..., r_{M-1} ]
         - W['rti'] is a list [ rti_0, ..., rti_{M-1} ], with rti_k the
         inverse of the adjoint of r_k.
         
         The call g = kktsolver(W) should return a function g that solves 
         the KKT system by g(x, y, z).  On entry, x, y, z contain the 
         righthand side bx, by, bz.  On exit, they contain the solution,
         with uz scaled, the argument z contains W*uz.  In other words, 
         on exit x, y, z are the solution of
         
         [ P   A'  G'*W^{-1} ] [ ux ]   [ bx ]
         [ A   0   0         ] [ uy ] = [ by ].
         [ G   0   -W'       ] [ uz ]   [ bz ]
         
         
         2.  The linear operators P*u, G*u and A*u can be specified 
         by providing Python functions instead of matrices.  This can only 
         be done in combination with 1. above, i.e., it requires the 
         kktsolver argument.
         
         If P is a function, the call P(u, v, alpha, beta) should evaluate 
         the matrix-vectors product
         
         v := alpha * P * u + beta * v.
         
         The arguments u and v are required.  The other arguments have 
         default values alpha = 1.0, beta = 0.0. 
         
         If G is a function, the call G(u, v, alpha, beta, trans) should 
         evaluate the matrix-vector products
         
         v := alpha * G * u + beta * v  if trans is 'N'
         v := alpha * G' * u + beta * v  if trans is 'T'.
         
         The arguments u and v are required.  The other arguments have
         default values alpha = 1.0, beta = 0.0, trans = 'N'.
         
         If A is a function, the call A(u, v, alpha, beta, trans) should
         evaluate the matrix-vectors products
         
         v := alpha * A * u + beta * v if trans is 'N'
         v := alpha * A' * u + beta * v if trans is 'T'.
         
         The arguments u and v are required.  The other arguments
         have default values alpha = 1.0, beta = 0.0, trans = 'N'.
         
         
         3.  Instead of using the default representation of the primal 
         variable x and the dual variable y as one-column 'd' matrices, 
         we can represent these variables and the corresponding parameters 
         q and b by arbitrary Python objects (matrices, lists, dictionaries,
         etc).  This can only be done in combination with 1. and 2. above,
         i.e., it requires a user-provided KKT solver and an operator 
         description of the linear mappings.   It also requires the 
         arguments xnewcopy, xdot, xscal, xaxpy, ynewcopy, ydot, yscal, 
         yaxpy.  These arguments are functions defined as follows.
         
         If X is the vector space of primal variables x, then:
         - xnewcopy(u) creates a new copy of the vector u in X.
         - xdot(u, v) returns the inner product of two vectors u and v in X.
         - xscal(alpha, u) computes u := alpha*u, where alpha is a scalar
         and u is a vector in X.
         - xaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar 
         alpha and two vectors u and v in X.
         If this option is used, the argument q must be in the same format
         as x, the argument P must be a Python function, the arguments A 
         and G must be Python functions or NULL, and the argument 
         kktsolver is required.
         
         If Y is the vector space of primal variables y:
         - ynewcopy(u) creates a new copy of the vector u in Y.
         - ydot(u, v) returns the inner product of two vectors u and v in Y.
         - yscal(alpha, u) computes u := alpha*u, where alpha is a scalar
         and u is a vector in Y.
         - yaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar 
         alpha and two vectors u and v in Y.
         If this option is used, the argument b must be in the same format
         as y, the argument A must be a Python function or NULL, and the 
         argument kktsolver is required.
         
         */
        template<typename Scalar>
        void coneqp(const QPMatrix<Scalar> &P, KQP_VECTOR(Scalar) &q,
                    ConeQPReturn<Scalar> &result,
                    bool initVals = false,
                    Dimensions dims = Dimensions(), 
                    const QPMatrix<Scalar> *G = NULL, KQP_VECTOR(Scalar)* h = NULL, 
                    KQP_MATRIX(Scalar) *A = NULL, KQP_VECTOR(Scalar) *b = NULL,
                    KKTPreSolver<Scalar>* kktpresolver = NULL, 
                    ConeQPOptions<Scalar> options = ConeQPOptions<Scalar>());
        
    }
    
}

#endif
