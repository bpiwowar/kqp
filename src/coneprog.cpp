/**
 * Based on cvxopt 1.1.3
 *
 * (c) 2011 B. Piwowarski 
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <boost/scoped_ptr.hpp>
#include <boost/format.hpp>
#include <algorithm>
#include <numeric>

#include "Eigen/Core"


#include "kqp.hpp"
#include "coneprog.hpp"

DEFINE_LOGGER(logger, "kqp.coneqp")

#define KQP_NOT_IMPLEMENTED BOOST_THROW_EXCEPTION(kqp::not_implemented_exception())


namespace kqp { namespace cvxopt {
    
    typedef std::vector<int>::iterator IntIterator;
    
    template<typename Scalar>
    ConeQPOptions<Scalar>::ConeQPOptions() :
    useCorrection(true),
    DEBUG(false),
    correction(true),
    show_progress(true),
    maxiters(100),
    abstol(1e-7),
    reltol(1e-6),
    feastol(1e-7),
    refinement(-1)
    {
        
    }
    
    template<typename Scalar>
    ConeQPReturn<Scalar>::ConeQPReturn() :
    gap(0), relative_gap(0), primal_objective(0), dual_objective(0), primal_infeasibility(0), dual_infeasibility(0), primal_slack(0), dual_slack(0), iterations(0) {}
    
    
    
    /// Inner product of two vectors in S.
    template<typename Scalar>
    Scalar sdot(const KQP_VECTOR(Scalar) &x, const KQP_VECTOR(Scalar) &y, const Dimensions &dims, size_t mnl = 0) {
        size_t ind = mnl + dims.l;
        for(std::vector<int>::const_iterator k = dims.q.begin(); k != dims.q.end(); k++)
            ind += *k;
        
        
        Scalar a = x.head(ind).dot(y.head(ind));
        
        for(size_t i = 0; i < dims.s.size(); i++) { 
            int m = dims.s[i];
            KQP_NOT_IMPLEMENTED;
            /*
             a +=
             blas.dot(x, y, offsetx = ind, offsety = ind, incx = m+1, 
             incy = m+1, n = m);
             
             for(int i = 1; i < m; i++) 
             a += 2.0 * blas.dot(x, y, incx = m+1, incy = m+1, 
             offsetx = ind+j, offsety = ind+j, n = m-j);
             ind += m * m;
             
             */
        }
        return a;
    }
    
    /// Returns the norm of a vector in S
    template<typename Scalar>
    Scalar snrm2(const KQP_VECTOR(Scalar) &x, const Dimensions &dims, size_t mnl = 0) {
        return std::sqrt(sdot(x, x, dims, mnl));
    }
    
    /**
     Returns min {t | x + t*e >= 0}, where e is defined as follows
     
     - For the nonlinear and 'l' blocks: e is the vector of ones.
     - For the 'q' blocks: e is the first unit vector.
     - For the 's' blocks: e is the identity matrix.
     
     When called with the argument sigma, also returns the eigenvalues 
     (in sigma) and the eigenvectors (in x) of the 's' components of x.
     */
    template<typename Scalar>
    Scalar max_step(KQP_VECTOR(Scalar) &x, const Dimensions &dims, int mnl = 0, const KQP_VECTOR(Scalar) * sigma = NULL) {
        std::vector<Scalar> t;
        int ind = mnl + dims.l;
        if (ind > 0)
            t.push_back(- x.topRows(ind).minCoeff() );
        
        if (!dims.q.empty()) {
            KQP_NOT_IMPLEMENTED;
            //    for(size_t i = 0; i < dims.q.size(); i++) { int m = dims.q[i];
            //        if (m > 0) 
            //            t += [ blas.nrm2(x, offset = ind + 1, n = m-1) - x[ind] ];
            //        ind += m;
            //    }
            //    
        }
        
        if (!dims.s.empty()) {
            KQP_NOT_IMPLEMENTED;
            //    if (sigma == NULL) {
            //        Q = matrix(0.0, (max(dims['s']), max(dims['s'])));
            //        w = matrix(0.0, (max(dims['s']),1));
            //    }
            
            //    int ind2 = 0;
            //    for(int m : dims.s) {
            //        if sigma is None:
            //            blas.copy(x, Q, offsetx = ind, n = m**2)
            //            lapack.syevr(Q, w, range = 'I', il = 1, iu = 1, n = m, ldA = m)
            //            if m:  t += [ -w[0] ]
            //        else:            
            //            lapack.syevd(x, sigma, jobz = 'V', n = m, ldA = m, offsetA = 
            //                ind, offsetW = ind2)
            //            if m:  t += [ -sigma[ind2] ] 
            //        ind += m*m
            //        ind2 += m
            //    }
            
        }
        
        if (!t.empty()) return *std::max_element(t.begin(), t.end());
        return 0.0;
    }
    
    /*
     Applies Nesterov-Todd scaling or its inverse.
     
     Computes 
     
     x := W*x        (trans is 'N', inverse = 'N')  
     x := W^T*x      (trans is 'T', inverse = 'N')  
     x := W^{-1}*x   (trans is 'N', inverse = 'I')  
     x := W^{-T}*x   (trans is 'T', inverse = 'I'). 
     
     x is a dense 'd' matrix.
     
     
     The 'dnl' and 'dnli' entries are optional, and only present when the 
     function is called from the nonlinear solver.
     
     */
    template <typename Scalar, int ColsAtCompileTime>
    void scale(Eigen::Matrix<Scalar, Eigen::Dynamic, ColsAtCompileTime> &x, const ScalingMatrix<Scalar> &W, bool trans = false, bool inverse = false) {
        
        size_t ind = 0;
        
        // Scaling for nonlinear component xk is xk := dnl .* xk; inverse 
        // scaling is xk ./ dnl = dnli .* xk, where dnl = W['dnl'], 
        // dnli = W['dnli'].
        
        if (W.dnl.rows() > 0) {
            const KQP_VECTOR(Scalar) &w = inverse ? W.dnli : W.dnl;        
            x.block(0, 0, w.rows(), x.cols()) = w.asDiagonal() * x.block(0, 0, w.rows(), x.cols());
            ind += w.rows();
        }
        
        // Scaling for linear 'l' component xk is xk := d .* xk; inverse 
        // scaling is xk ./ d = di .* xk, where d = W['d'], di = W['di'].
        
        const KQP_VECTOR(Scalar) &w = inverse ? W.di : W.d;
        x.block(ind, 0, w.rows(), x.cols()).noalias() = w.asDiagonal() * x.block(ind, 0, w.rows(), x.cols());
        ind += w.rows();
        
        
        // Scaling for 'q' component is 
        //
        //     xk := beta * (2*v*v' - J) * xk
        //         = beta * (2*v*(xk'*v)' - J*xk)
        //
        // where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
        //
        // Inverse scaling is
        //
        //     xk := 1/beta * (2*J*v*v'*J - J) * xk
        //         = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk). 
        
        if (!W.beta.empty()) {
            KQP_NOT_IMPLEMENTED;
            
            /*
             
             w = matrix(0.0, (x.size[1], 1));
             
             for k in xrange(len(W['v'])) {
             v = W['v'][k]
             m = v.size[0]
             if inverse == 'I':  
             blas.scal(-1.0, x, offset = ind, inc = x.size[0])
             blas.gemv(x, v, w, trans = 'T', m = m, n = x.size[1], offsetA = 
             ind, ldA = x.size[0])
             blas.scal(-1.0, x, offset = ind, inc = x.size[0])
             blas.ger(v, w, x, alpha = 2.0, m = m, n = x.size[1], ldA = 
             x.size[0], offsetA = ind)
             if inverse == 'I': 
             blas.scal(-1.0, x, offset = ind, inc = x.size[0])
             a = 1.0 / W['beta'][k] 
             else:
             a = W['beta'][k] 
             for i in xrange(x.size[1]):
             blas.scal(a, x, n = m, offset = ind + i*x.size[0])
             ind += m
             }
             */
        }
        
        // Scaling for 's' component xk is
        //
        //     xk := vec( r' * mat(xk) * r )  if trans = 'N'
        //     xk := vec( r * mat(xk) * r' )  if trans = 'T'.
        //
        // r is kth element of W['r'].
        //
        // Inverse scaling is
        //
        //     xk := vec( rti * mat(xk) * rti' )  if trans = 'N'
        //     xk := vec( rti' * mat(xk) * rti )  if trans = 'T'.
        //
        // rti is kth element of W['rti'].
        
        if (!W.r.empty()) {
            KQP_NOT_IMPLEMENTED;
            /*
             maxn = max( [0] + [ r.size[0] for r in W['r'] ] )
             a = matrix(0.0, (maxn, maxn))
             for k in xrange(len(W['r'])) {
             bool t;
             
             if (inverse == 'N') {
             r = W['r'][k];
             t = !trans;
             }else {
             r = W['rti'][k];
             t = trans;
             }
             
             n = r.size[0]
             for i in xrange(x.size[1]) {
             
             // scale diagonal of xk by 0.5
             blas.scal(0.5, x, offset = ind + i*x.size[0], inc = n+1, n = n)
             
             // a = r*tril(x) (t is 'N') or a = tril(x)*r  (t is 'T')
             blas.copy(r, a)
             if t == 'N':   
             blas.trmm(x, a, side = 'R', m = n, n = n, ldA = n, ldB = n,
             offsetA = ind + i*x.size[0])
             else:    
             blas.trmm(x, a, side = 'L', m = n, n = n, ldA = n, ldB = n,
             offsetA = ind + i*x.size[0])
             
             // x := (r*a' + a*r')  if t is 'N'
             // x := (r'*a + a'*r)  if t is 'T'
             blas.syr2k(r, a, x, trans = t, n = n, k = n, ldB = n, ldC = n,
             offsetC = ind + i*x.size[0])
             }
             ind += n**2
             
             }
             */
        }
        
    }
    
    
    
    template<typename Scalar>
    void res(const QPMatrix<Scalar> &P, const KQP_MATRIX(Scalar) &A, const QPMatrix<Scalar> &G, const KQP_VECTOR(Scalar) & ux, const KQP_VECTOR(Scalar) &  uy, const KQP_VECTOR(Scalar) & uz,const KQP_VECTOR(Scalar) &  us,  KQP_VECTOR(Scalar) & vx,  KQP_VECTOR(Scalar) & vy,  KQP_VECTOR(Scalar) & vz, KQP_VECTOR(Scalar) &  vs, const ScalingMatrix<Scalar> &W, const KQP_VECTOR(Scalar) & lmbda) {
        
        // Evaluates residual in Newton equations:
        // 
        //      [ vx ]    [ vx ]   [ 0     ]   [ P  A'  G' ]   [ ux        ]
        //      [ vy ] := [ vy ] - [ 0     ] - [ A  0   0  ] * [ uy        ]
        //      [ vz ]    [ vz ]   [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]
        //
        //      vs := vs - lmbda o (uz + us).
        
        // vx := vx - P*ux - A'*uy - G'*W^{-1}*uz
        KQP_MATRIX(Scalar) wz3 = uz;
        scale(wz3, W, false, true);
        
        KQP_VECTOR(Scalar) Pux, Gtuz;
        P.mult(ux, Pux);
        G.mult(uz, Gtuz);
        vx -= Pux - A.transpose() * uy - Gtuz;
        
        
        // vy := vy - A*ux
        vy -= A * ux;
        
        // vz := vz - G*ux - W'*us
        KQP_MATRIX(Scalar) ws3 = ux;
        scale(ws3, W, true, false);
        KQP_VECTOR(Scalar) Gux;
        G.mult(ux, Gux);
        vz -= Gux - ws3;
        
        // vs := vs - lmbda o (uz + us)
        vs.array() -= lmbda.array() * (uz + us).array();
    }
    
    /**
     Converts lower triangular matrix to symmetric.  
     Fills in the upper triangular part of the symmetric matrix stored in 
     x[offset : offset+n*n] using 'L' storage.
     */
    //KQP_MATRIX(Scalar) symm(const KQP_MATRIX(Scalar) &x) {
    //    if (n <= 1) return x;
    //    return x.selfadjointView<Eigen::Lower>();
    //    for i in xrange(n-1):
    //        blas.copy(x, x, offsetx = offset + i*(n+1) + 1, offsety = 
    //            offset + (i+1)*(n+1) - 1, incy = n, n = n-i-1)
    //}
    
    /**
     The inverse product x := (y o\ x), when the 's' components of y are 
     diagonal.
     */
    template<typename Scalar>
    void sinv(KQP_VECTOR(Scalar) &x, const KQP_VECTOR(Scalar) &y, const Dimensions &dims, int mnl = 0) {
        
        // For the nonlinear and 'l' blocks:  
        // 
        //     yk o\ xk = yk .\ xk.
        
        x.topRows(mnl + dims.l).array() /= y.topRows(mnl + dims.l).array();
        
        // blas.tbsv(y, x, n = mnl + dims.l, k = 0, ldA = 1)
        
        // For the 'q' blocks: 
        //
        //                        [ l0   -l1'              ]  
        //     yk o\ xk = 1/a^2 * [                        ] * xk
        //                        [ -l1  (a*I + l1*l1')/l0 ]
        //
        // where yk = (l0, l1) and a = l0^2 - l1'*l1.
        
        int ind = mnl + dims.l;
        for(size_t i = 0; i < dims.q.size(); i++) { int m = dims.q[i];
            KQP_NOT_IMPLEMENTED;
            //        aa = jnrm2(y, n = m, offset = ind)**2
            //        cc = x[ind]
            //        dd = blas.dot(y, x, offsetx = ind+1, offsety = ind+1, n = m-1)
            //        x[ind] = cc * y[ind] - dd
            //        blas.scal(aa / y[ind], x, n = m-1, offset = ind+1)
            //        blas.axpy(y, x, alpha = dd/y[ind] - cc, n = m-1, offsetx = ind+1, 
            //            offsety = ind+1)
            //        blas.scal(1.0/aa, x, n = m, offset = ind)
            //        ind += m
        }
        
        // For the 's' blocks:
        //
        //     yk o\ xk =  xk ./ gamma
        //
        // where gammaij = .5 * (yk_i + yk_j).
        
        int ind2 = ind;
        for(size_t i = 0; i < dims.s.size(); i++) { int m = dims.s[i];
            KQP_NOT_IMPLEMENTED;
            //        for j in xrange(m):
            //            u = 0.5 * ( y[ind2+j:ind2+m] + y[ind2+j] )
            //            blas.tbsv(u, x, n = m-j, k = 0, ldA = 1, offsetx = ind + 
            //                j*(m+1))  
            //        ind += m*m
            //        ind2 += m
        }
    }
    
    
    template<typename Scalar>
    class f4_no_ir_class {
        const ScalingMatrix<Scalar> &W;
        const KKTSolver<Scalar> &solver;
        const Dimensions &dims; 
        const KQP_VECTOR(Scalar) &lmbda;
    public:
        f4_no_ir_class(const ScalingMatrix<Scalar> &W, const KKTSolver<Scalar> &solver, const Dimensions &dims, const KQP_VECTOR(Scalar) &lmbda):
        W(W), solver(solver), dims(dims), lmbda(lmbda) {}
        
        // f4_no_ir(x, y, z, s) solves
        // 
        //     [ 0     ]   [ P  A'  G' ]   [ ux        ]   [ bx ]
        //     [ 0     ] + [ A  0   0  ] * [ uy        ] = [ by ]
        //     [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]   [ bz ]
        //
        //     lmbda o (uz + us) = bs.
        //
        // On entry, x, y, z, s contain bx, by, bz, bs.
        // On exit, they contain ux, uy, uz, us.
        void operator()(KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &y, KQP_VECTOR(Scalar) &z, KQP_VECTOR(Scalar) &s) const {
            // Solve 
            //
            //     [ P A' G'   ] [ ux        ]    [ bx                    ]
            //     [ A 0  0    ] [ uy        ] =  [ by                    ]
            //     [ G 0 -W'*W ] [ W^{-1}*uz ]    [ bz - W'*(lmbda o\ bs) ]
            //
            //     us = lmbda o\ bs - uz.
            //
            // On entry, x, y, z, s  contains bx, by, bz, bs. 
            // On exit they contain x, y, z, s.
            
            // s := lmbda o\ s 
            //    = lmbda o\ bs
            sinv(s, lmbda, dims);
            
            // z := z - W'*s 
            //    = bz - W'*(lambda o\ bs)
            KQP_MATRIX(Scalar) ws3 = s;
            scale(ws3, W, true, false);
            z = z - ws3;
            
            // Solve for ux, uy, uz
            solver.solve(x, y, z);
            
            // s := s - z 
            //    = lambda o\ bs - uz.
            s = s - z;
        }
    };
    
    template<typename Scalar>
    class f4_class {
        const int refinement;
        const bool DEBUG;
        const f4_no_ir_class<Scalar> &f4_no_ir;
        const ScalingMatrix<Scalar>& W;
        const KQP_VECTOR(Scalar) &lmbda;
        const Dimensions &dims;
        const KQP_MATRIX(Scalar) &A;
        const QPMatrix<Scalar> &P, &G;
        
    public:
        f4_class(int refinement, bool DEBUG, const f4_no_ir_class<Scalar> &f4_no_ir, const ScalingMatrix<Scalar> &W, const KQP_VECTOR(Scalar) &lmbda, const Dimensions &dims,
                 const QPMatrix<Scalar> &P, const KQP_MATRIX(Scalar) &A, const QPMatrix<Scalar> &G) 
        : refinement(refinement), DEBUG(DEBUG), f4_no_ir(f4_no_ir), W(W), lmbda(lmbda), dims(dims), P(P), A(A), G(G)
        {
            
        }
        
        void operator()(KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &y, KQP_VECTOR(Scalar) &z, KQP_VECTOR(Scalar) &s) {
            
            KQP_VECTOR(Scalar) wx, wy, wz, ws;
            KQP_VECTOR(Scalar) wx2, wy2, wz2, ws2;
            
            if (refinement > 0|| DEBUG) {
                wx = x;
                wy = y;
                wz = z;
                ws = s;
            }
            
            f4_no_ir(x, y, z, s);   
            
            for(int i = 0; i < refinement; i++) {
                wx2 = wx;
                wy2 = wy;
                wz2 = wz;
                ws2 = ws;
                res(P, A, G, x, y, z, s, wx2, wy2, wz2, ws2, W, lmbda);
                f4_no_ir(wx2, wy2, wz2, ws2);
                x += wx2;
                y += wy2;
                z += wz2;
                s += ws2;
            }
            
            if (DEBUG) {
                res(P, A, G, x, y, z, s, wx, wy, wz, ws, W, lmbda);
                std::cerr << "KKT residuals:" << std::endl;
                std::cerr << boost::format("    'x': %e") % wx.norm()  << std::endl;
                std::cerr << boost::format("    'y': %e") % wy.norm() << std::endl;
                std::cerr << boost::format("    'z': %e") % snrm2(wz, dims) << std::endl;
                std::cerr << boost::format("    's': %e") % snrm2(ws, dims) << std::endl;
            }
        }
    };
    
    
    /**
     Returns the Nesterov-Todd scaling W at points s and z, and stores the 
     scaled variable in lmbda. 
     
     W * z = W^{-T} * s = lmbda. 
     */
    template<typename Scalar>
    void compute_scaling(ScalingMatrix<Scalar> &W, const KQP_VECTOR(Scalar) &s, const KQP_VECTOR(Scalar) &z, KQP_VECTOR(Scalar) &lmbda, const Dimensions &dims, int mnl = -1) {
        
        
        // For the nonlinear block:
        //
        //     W['dnl'] = sqrt( s[:mnl] ./ z[:mnl] )
        //     W['dnli'] = sqrt( z[:mnl] ./ s[:mnl] )
        //     lambda[:mnl] = sqrt( s[:mnl] .* z[:mnl] )
        
        if (mnl < 0) {
            mnl = 0;
        } else {
            W.dnl.diagonal() = (s.segment(0, mnl).array() / z.segment(0, mnl).array()).sqrt();
            W.dnli = W.dnl.inverse();
            lmbda.segment(0,mnl) = (s.segment(0, mnl).array() * z.segment(0, mnl).array()).sqrt();
        }
        
        // For the 'l' block: 
        //
        //     W['d'] = sqrt( sk ./ zk )
        //     W['di'] = sqrt( zk ./ sk )
        //     lambdak = sqrt( sk .* zk )
        //
        // where sk and zk are the firstdims.l entries of s and z.
        // lambda_k is stored in the firstdims.l positions of lmbda.
        
        long m = dims.l;
        W.d.diagonal() = (s.segment(mnl, m).array() / z.segment(mnl, m).array()).sqrt();
        W.di = W.d.inverse();
        
        lmbda.segment(mnl, m) = (s.segment(mnl, m).array() * z.segment(mnl, m).array()).sqrt();
        
        
        if (!dims.q.empty() || !dims.s.empty())
            // see below
            KQP_NOT_IMPLEMENTED;
        /*
         // For the 'q' blocks, compute lists 'v', 'beta'.
         //
         // The vector v[k] has unit hyperbolic norm: 
         // 
         //     (sqrt( v[k]' * J * v[k] ) = 1 with J = [1, 0; 0, -I]).
         // 
         // beta[k] is a positive scalar.
         //
         // The hyperbolic Householder matrix H = 2*v[k]*v[k]' - J
         // defined by v[k] satisfies 
         // 
         //     (beta[k] * H) * zk  = (beta[k] * H) \ sk = lambda_k
         //
         // where sk = s[indq[k]:indq[k+1]], zk = z[indq[k]:indq[k+1]].
         //
         // lambda_k is stored in lmbda[indq[k]:indq[k+1]].
         
         ind = mnl +dims.l
         W['v'] = [ matrix(0.0, (k,1)) for k in dims['q'] ]
         W['beta'] = len(dims['q']) * [ 0.0 ] 
         
         for k in xrange(len(dims['q'])):
         m = dims['q'][k]
         v = W['v'][k]
         
         // a = sqrt( sk' * J * sk )  where J = [1, 0; 0, -I]
         aa = jnrm2(s, offset = ind, n = m)
         
         // b = sqrt( zk' * J * zk )
         bb = jnrm2(z, offset = ind, n = m) 
         
         // beta[k] = ( a / b )**1/2
         W['beta'][k] = math.sqrt( aa / bb )
         
         // c = sqrt( (sk/a)' * (zk/b) + 1 ) / sqrt(2)    
         cc = math.sqrt( ( blas.dot(s, z, n = m, offsetx = ind, offsety = 
         ind) / aa / bb + 1.0 ) / 2.0 )
         
         // vk = 1/(2*c) * ( (sk/a) + J * (zk/b) )
         blas.copy(z, v, offsetx = ind, n = m)
         blas.scal(-1.0/bb, v)
         v[0] *= -1.0 
         blas.axpy(s, v, 1.0/aa, offsetx = ind, n = m)
         blas.scal(1.0/2.0/cc, v)
         
         // v[k] = 1/sqrt(2*(vk0 + 1)) * ( vk + e ),  e = [1; 0]
         v[0] += 1.0
         blas.scal(1.0/math.sqrt(2.0 * v[0]), v)
         
         // To get the scaled variable lambda_k
         // 
         //     d =  sk0/a + zk0/b + 2*c
         //     lambda_k = [ c; 
         //                  (c + zk0/b)/d * sk1/a + (c + sk0/a)/d * zk1/b ]
         //     lambda_k *= sqrt(a * b)
         
         lmbda[ind] = cc
         dd = 2*cc + s[ind]/aa + z[ind]/bb
         blas.copy(s, lmbda, offsetx = ind+1, offsety = ind+1, n = m-1) 
         blas.scal((cc + z[ind]/bb)/dd/aa, lmbda, n = m-1, offset = ind+1)
         blas.axpy(z, lmbda, (cc + s[ind]/aa)/dd/bb, n = m-1, offsetx = 
         ind+1, offsety = ind+1)
         blas.scal(math.sqrt(aa*bb), lmbda, offset = ind, n = m)
         
         ind += m
         
         
         // For the 's' blocks: compute two lists 'r' and 'rti'.
         //
         //     r[k]' * sk^{-1} * r[k] = diag(lambda_k)^{-1}
         //     r[k]' * zk * r[k] = diag(lambda_k)
         //
         // where sk and zk are the entries inds[k] : inds[k+1] of
         // s and z, reshaped into symmetric matrices.
         //
         // rti[k] is the inverse of r[k]', so 
         //
         //     rti[k]' * sk * rti[k] = diag(lambda_k)^{-1}
         //     rti[k]' * zk^{-1} * rti[k] = diag(lambda_k).
         //
         // The vectors lambda_k are stored in 
         // 
         //     lmbda[dims.l + dimsq : -1 ]
         
         W['r'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
         W['rti'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
         work = matrix(0.0, (max( [0] + dims['s'] )**2, 1))
         Ls = matrix(0.0, (max( [0] + dims['s'] )**2, 1))
         Lz = matrix(0.0, (max( [0] + dims['s'] )**2, 1))
         
         ind2 = ind
         for k in xrange(len(dims['s'])):
         m = dims['s'][k]
         r, rti = W['r'][k], W['rti'][k]
         
         // Factor sk = Ls*Ls'; store Ls in ds[inds[k]:inds[k+1]].
         blas.copy(s, Ls, offsetx = ind2, n = m**2) 
         lapack.potrf(Ls, n = m, ldA = m)
         
         // Factor zs[k] = Lz*Lz'; store Lz in dz[inds[k]:inds[k+1]].
         blas.copy(z, Lz, offsetx = ind2, n = m**2) 
         lapack.potrf(Lz, n = m, ldA = m)
         
         // SVD Lz'*Ls = U*diag(lambda_k)*V'.  Keep U in work. 
         for i in xrange(m): 
         blas.scal(0.0, Ls, offset = i*m, n = i)
         blas.copy(Ls, work, n = m**2)
         blas.trmm(Lz, work, transA = 'T', ldA = m, ldB = m, n = m, m = m) 
         lapack.gesvd(work, lmbda, jobu = 'O', ldA = m, m = m, n = m, 
         offsetS = ind)
         
         // r = Lz^{-T} * U 
         blas.copy(work, r, n = m*m)
         blas.trsm(Lz, r, transA = 'T', m = m, n = m, ldA = m)
         
         // rti = Lz * U 
         blas.copy(work, rti, n = m*m)
         blas.trmm(Lz, rti, m = m, n = m, ldA = m)
         
         // r := r * diag(sqrt(lambda_k))
         // rti := rti * diag(1 ./ sqrt(lambda_k))
         for i in xrange(m):
         a = math.sqrt( lmbda[ind+i] )
         blas.scal(a, r, offset = m*i, n = m)
         blas.scal(1.0/a, rti, offset = m*i, n = m)
         
         ind += m
         ind2 += m*m
         */
    }
    
    /**
     Updates the Nesterov-Todd scaling matrix W and the scaled variable 
     lmbda so that on exit
     
     W * zt = W^{-T} * st = lmbda.
     
     On entry, the nonlinear, 'l' and 'q' components of the arguments s 
     and z contain W^{-T}*st and W*zt, i.e, the new iterates in the current 
     scaling.
     
     The 's' components contain the factors Ls, Lz in a factorization of 
     the new iterates in the current scaling, W^{-T}*st = Ls*Ls',   
     W*zt = Lz*Lz'.
     */
    template<typename Scalar>
    void update_scaling(const Dimensions &dims, ScalingMatrix<Scalar> &W, KQP_VECTOR(Scalar) &lmbda, KQP_VECTOR(Scalar) &s, KQP_VECTOR(Scalar) &z) {
        // Nonlinear and 'l' blocks
        //
        //    d :=  d .* sqrt( s ./ z )
        //    lmbda := lmbda .* sqrt(s) .* sqrt(z)
        
        int mnl = W.dnl.rows();
        
        int ml = W.d.rows();
        int m = mnl + ml;
        s.segment(0,m) = s.segment(0,m).cwiseSqrt();
        z.segment(0,m) = z.segment(0,m).cwiseSqrt();
        
        // d := d .* s .* z 
        if (mnl > 0) {
            W.dnl.diagonal() = W.dnl.diagonal().segment(0,mnl).array() * s.segment(0, mnl).array() / z.segment(0, mnl).array();
            W.dnli = W.dnl.inverse();
        }
        
        W.d.diagonal() = W.d.diagonal().array() * s.segment(mnl, ml).array() / z.segment(0, ml).array();
        W.di = W.d.inverse();
        
        // lmbda := s .* z
        
        lmbda = s.topRows(m).array() * z.topRows(m).array();
        
        if (!dims.q.empty() || !dims.s.empty())
            // see below
            KQP_NOT_IMPLEMENTED;
        
        /*
         // 'q' blocks.
         // 
         // Let st and zt be the new variables in the old scaling:
         //
         //     st = s_k,   zt = z_k
         //
         // and a = sqrt(st' * J * st),  b = sqrt(zt' * J * zt).
         //
         // 1. Compute the hyperbolic Householder transformation 2*q*q' - J 
         //    that maps st/a to zt/b.
         // 
         //        c = sqrt( (1 + st'*zt/(a*b)) / 2 ) 
         //        q = (st/a + J*zt/b) / (2*c). 
         //
         //    The new scaling point is 
         //
         //        wk := betak * sqrt(a/b) * (2*v[k]*v[k]' - J) * q 
         //
         //    with betak = W['beta'][k].
         // 
         // 3. The scaled variable:
         //
         //        lambda_k0 = sqrt(a*b) * c
         //        lambda_k1 = sqrt(a*b) * ( (2vk*vk' - J) * (-d*q + u/2) )_1
         //
         //    where 
         //
         //        u = st/a - J*zt/b 
         //        d = ( vk0 * (vk'*u) + u0/2 ) / (2*vk0 *(vk'*q) - q0 + 1).
         //
         // 4. Update scaling
         //   
         //        v[k] := wk^1/2 
         //              = 1 / sqrt(2*(wk0 + 1)) * (wk + e).
         //        beta[k] *=  sqrt(a/b)
         
         ind = m
         for k in xrange(len(W['v'])):
         
         v = W['v'][k]
         m = len(v)
         
         // ln = sqrt( lambda_k' * J * lambda_k )
         ln = jnrm2(lmbda, n = m, offset = ind) 
         
         // a = sqrt( sk' * J * sk ) = sqrt( st' * J * st ) 
         // s := s / a = st / a
         aa = jnrm2(s, offset = ind, n = m)
         blas.scal(1.0/aa, s, offset = ind, n = m)
         
         // b = sqrt( zk' * J * zk ) = sqrt( zt' * J * zt )
         // z := z / a = zt / b
         bb = jnrm2(z, offset = ind, n = m) 
         blas.scal(1.0/bb, z, offset = ind, n = m)
         
         // c = sqrt( ( 1 + (st'*zt) / (a*b) ) / 2 )
         cc = math.sqrt( ( 1.0 + blas.dot(s, z, offsetx = ind, offsety = 
         ind, n = m) ) / 2.0 )
         
         // vs = v' * st / a 
         vs = blas.dot(v, s, offsety = ind, n = m) 
         
         // vz = v' * J *zt / b
         vz = jdot(v, z, offsety = ind, n = m) 
         
         // vq = v' * q where q = (st/a + J * zt/b) / (2 * c)
         vq = (vs + vz ) / 2.0 / cc
         
         // vu = v' * u  where u =  st/a - J * zt/b 
         vu = vs - vz  
         
         // lambda_k0 = c
         lmbda[ind] = cc
         
         // wk0 = 2 * vk0 * (vk' * q) - q0 
         wk0 = 2 * v[0] * vq - ( s[ind] + z[ind] ) / 2.0 / cc 
         
         // d = (v[0] * (vk' * u) - u0/2) / (wk0 + 1)
         dd = (v[0] * vu - s[ind]/2.0 + z[ind]/2.0) / (wk0 + 1.0)
         
         // lambda_k1 = 2 * v_k1 * vk' * (-d*q + u/2) - d*q1 + u1/2
         blas.copy(v, lmbda, offsetx = 1, offsety = ind+1, n = m-1)
         blas.scal(2.0 * (-dd * vq + 0.5 * vu), lmbda, offset = ind+1, 
         n = m-1)
         blas.axpy(s, lmbda, 0.5 * (1.0 - dd/cc), offsetx = ind+1, offsety 
         = ind+1, n = m-1)
         blas.axpy(z, lmbda, 0.5 * (1.0 + dd/cc), offsetx = ind+1, offsety
         = ind+1, n = m-1)
         
         // Scale so that sqrt(lambda_k' * J * lambda_k) = sqrt(aa*bb).
         blas.scal(math.sqrt(aa*bb), lmbda, offset = ind, n = m)
         
         // v := (2*v*v' - J) * q 
         //    = 2 * (v'*q) * v' - (J* st/a + zt/b) / (2*c)
         blas.scal(2.0 * vq, v)
         v[0] -= s[ind] / 2.0 / cc
         blas.axpy(s, v,  0.5/cc, offsetx = ind+1, offsety = 1, n = m-1)
         blas.axpy(z, v, -0.5/cc, offsetx = ind, n = m)
         
         // v := v^{1/2} = 1/sqrt(2 * (v0 + 1)) * (v + e)
         v[0] += 1.0
         blas.scal(1.0 / math.sqrt(2.0 * v[0]), v)
         
         // beta[k] *= ( aa / bb )**1/2
         W['beta'][k] *= math.sqrt( aa / bb )
         
         ind += m
         
         
         // 's' blocks
         // 
         // Let st, zt be the updated variables in the old scaling:
         // 
         //     st = Ls * Ls', zt = Lz * Lz'.
         //
         // where Ls and Lz are the 's' components of s, z.
         //
         // 1.  SVD Lz'*Ls = Uk * lambda_k^+ * Vk'.
         //
         // 2.  New scaling is 
         //
         //         r[k] := r[k] * Ls * Vk * diag(lambda_k^+)^{-1/2}
         //         rti[k] := r[k] * Lz * Uk * diag(lambda_k^+)^{-1/2}.
         //
         
         work = matrix(0.0, (max( [0] + [r.size[0] for r in W['r']])**2, 1))
         ind = mnl + ml + sum([ len(v) for v in W['v'] ])
         ind2, ind3 = ind, 0
         for k in xrange(len(W['r'])):
         r, rti = W['r'][k], W['rti'][k]
         m = r.size[0]
         
         // r := r*sk = r*Ls
         blas.gemm(r, s, work, m = m, n = m, k = m, ldB = m, ldC = m,
         offsetB = ind2)
         blas.copy(work, r, n = m**2)
         
         // rti := rti*zk = rti*Lz
         blas.gemm(rti, z, work, m = m, n = m, k = m, ldB = m, ldC = m,
         offsetB = ind2)
         blas.copy(work, rti, n = m**2)
         
         // SVD Lz'*Ls = U * lmbds^+ * V'; store U in sk and V' in zk.
         blas.gemm(z, s, work, transA = 'T', m = m, n = m, k = m, ldA = m,
         ldB = m, ldC = m, offsetA = ind2, offsetB = ind2)
         lapack.gesvd(work, lmbda, jobu = 'A', jobvt = 'A', m = m, n = m, 
         ldA = m, U = s, Vt = z, ldU = m, ldVt = m, offsetS = ind, 
         offsetU = ind2, offsetVt = ind2)
         
         // r := r*V
         blas.gemm(r, z, work, transB = 'T', m = m, n = m, k = m, ldB = m,
         ldC = m, offsetB = ind2)
         blas.copy(work, r, n = m**2)
         
         // rti := rti*U
         blas.gemm(rti, s, work, n = m, m = m, k = m, ldB = m, ldC = m,
         offsetB = ind2)
         blas.copy(work, rti, n = m**2)
         
         // r := r*lambda^{-1/2}; rti := rti*lambda^{-1/2}
         for i in xrange(m):    
         a = 1.0 / math.sqrt(lmbda[ind+i])
         blas.scal(a, r, offset = m*i, n = m)
         blas.scal(a, rti, offset = m*i, n = m)
         
         ind += m
         ind2 += m*m
         ind3 += m
         */
    }
    
    /**
     The product x := y o y.   The 's' components of y are diagonal and
     only the diagonals of x and y are stored.     
     */
    template<typename Scalar>
    void ssqr(KQP_VECTOR(Scalar) &x, const KQP_VECTOR(Scalar) &y, const Dimensions &dims, int mnl = 0) {
        x = y.array() * y.array();
        int ind = mnl + dims.l;
        
        if (!dims.q.empty() || !dims.s.empty())
            // see below
            KQP_NOT_IMPLEMENTED;
        
        /*
         for m in dims['q']:
         x[ind] = blas.nrm2(y, offset = ind, n = m)**2
         blas.scal(2.0*y[ind], x, n = m-1, offset = ind+1)
         ind += m 
         blas.tbmv(y, x, n = sum(dims['s']), k = 0, ldA = 1, offsetA = ind, 
         offsetx = ind) 
         */
    }
    
    
    /**
     The product x := (y o x).  If diag is true, the 's' part of y is 
     diagonal and only the diagonal is stored.
     */
    template<typename Scalar>
    void sprod(KQP_VECTOR(Scalar) &x, const KQP_VECTOR(Scalar) &y, const Dimensions &dims, int mnl = 0, bool diag = false) {
        
        // For the nonlinear and 'l' blocks:  
        //
        //     yk o xk = yk .* xk.
        
        int n = mnl +dims.l;
        x.topRows(n).array() *= y.topRows(n).array();
        
        if (!dims.q.empty() || !dims.s.empty())
            // see below
            KQP_NOT_IMPLEMENTED;
        
        /*
         // For 'q' blocks: 
         //
         //               [ l0   l1'  ]
         //     yk o xk = [           ] * xk
         //               [ l1   l0*I ] 
         //
         // where yk = (l0, l1).
         
         ind = mnl +dims.l
         for m in dims['q']:
         dd = blas.dot(x, y, offsetx = ind, offsety = ind, n = m)
         blas.scal(y[ind], x, offset = ind+1, n = m-1)
         blas.axpy(y, x, alpha = x[ind], n = m-1, offsetx = ind+1, offsety 
         = ind+1)
         x[ind] = dd
         ind += m
         
         
         // For the 's' blocks:
         //
         //    yk o sk = .5 * ( Yk * mat(xk) + mat(xk) * Yk )
         // 
         // where Yk = mat(yk) if diag is 'N' and Yk = diag(yk) if diag is 'D'.
         
         if diag is 'N':
         maxm = max([0] + dims['s'])
         A = matrix(0.0, (maxm, maxm))
         
         for m in dims['s']:
         blas.copy(x, A, offsetx = ind, n = m*m)
         
         // Write upper triangular part of A and yk.
         for i in xrange(m-1):
         symm(A, m)
         symm(y, m, offset = ind)
         
         // xk = 0.5 * (A*yk + yk*A)
         blas.syr2k(A, y, x, alpha = 0.5, n = m, k = m, ldA = m,  ldB = 
         m, ldC = m, offsetB = ind, offsetC = ind)
         
         ind += m*m
         
         else:
         ind2 = ind
         for m in dims['s']:
         for j in xrange(m):
         u = 0.5 * ( y[ind2+j:ind2+m] + y[ind2+j] )
         blas.tbmv(u, x, n = m-j, k = 0, ldA = 1, offsetx = 
         ind + j*(m+1))  
         ind += m*m
         ind2 += m
         */
        
    }
    
    /*
     Evaluates
     
     x := H(lambda^{1/2}) * x   (inverse is 'N')
     x := H(lambda^{-1/2}) * x  (inverse is 'I').
     
     H is the Hessian of the logarithmic barrier.
     */
    template<typename Scalar>
    void scale2(const KQP_VECTOR(Scalar) &lmbda, KQP_VECTOR(Scalar) &x, const Dimensions &dims, int mnl = 0, bool inverse = false) {
        
        // For the nonlinear and 'l' blocks, 
        //
        //     xk := xk ./ l   (inverse is 'N')
        //     xk := xk .* l   (inverse is 'I')
        //
        // where l is lmbda[:mnl+dims.l].
        
        if (!inverse)
            x.topRows(mnl + dims.l).array() /= lmbda.topRows(mnl + dims.l).array();
        else
            x.topRows(mnl + dims.l).array() *= lmbda.topRows(mnl + dims.l).array();
        
        
        if (!dims.q.empty() || !dims.s.empty())
            // see below
            KQP_NOT_IMPLEMENTED;
        /*
         // For 'q' blocks, if inverse is 'N',
         //
         //     xk := 1/a * [ l'*J*xk;  
         //         xk[1:] - (xk[0] + l'*J*xk) / (l[0] + 1) * l[1:] ].
         //
         // If inverse is 'I',
         //
         //     xk := a * [ l'*xk; 
         //         xk[1:] + (xk[0] + l'*xk) / (l[0] + 1) * l[1:] ].
         //
         // a = sqrt(lambda_k' * J * lambda_k), l = lambda_k / a.
         
         ind = mnl +dims.l
         for m in dims['q']:
         a = jnrm2(lmbda, n = m, offset = ind)
         if inverse == 'N':
         lx = jdot(lmbda, x, n = m, offsetx = ind, offsety = ind)/a
         else:
         lx = blas.dot(lmbda, x, n = m, offsetx = ind, offsety = ind)/a
         x0 = x[ind]
         x[ind] = lx
         c = (lx + x0) / (lmbda[ind]/a + 1) / a 
         if inverse == 'N':  c *= -1.0
         blas.axpy(lmbda, x, alpha = c, n = m-1, offsetx = ind+1, offsety =
         ind+1)
         if inverse == 'N': a = 1.0/a 
         blas.scal(a, x, offset = ind, n = m)
         ind += m
         
         
         // For the 's' blocks, if inverse is 'N',
         //
         //     xk := vec( diag(l)^{-1/2} * mat(xk) * diag(k)^{-1/2}).
         //
         // If inverse is 'I',
         //
         //     xk := vec( diag(l)^{1/2} * mat(xk) * diag(k)^{1/2}).
         //
         // where l is kth block of lambda.
         // 
         // We scale upper and lower triangular part of mat(xk) because the
         // inverse operation will be applied to nonsymmetric matrices.
         
         ind2 = ind
         for k in xrange(len(dims['s'])):
         m = dims['s'][k]
         for j in xrange(m):
         c = math.sqrt(lmbda[ind2+j]) * base.sqrt(lmbda[ind2:ind2+m])
         if inverse == 'N':  
         blas.tbsv(c, x, n = m, k = 0, ldA = 1, offsetx = ind + j*m)
         else:
         blas.tbmv(c, x, n = m, k = 0, ldA = 1, offsetx = ind + j*m)
         ind += m*m
         ind2 += m
         */
        
    }
    
    template<typename Scalar>
    void coneqp(const QPMatrix<Scalar> &P, KQP_VECTOR(Scalar) &q,
                ConeQPReturn<Scalar> &result,
                bool initVals,
                Dimensions dims,
                const QPMatrix<Scalar> *_G, KQP_VECTOR(Scalar)* _h, 
                KQP_MATRIX(Scalar) *_A, KQP_VECTOR(Scalar) *_b,
                KKTPreSolver<Scalar>* kktpresolver, 
                ConeQPOptions<Scalar> options) {
        
        const Scalar STEP = 0.99;
        const Scalar EXPON = 3;
        
        if (options.maxiters < 1)
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("Option maxiters must be a positive integer"));
        
        
        if (options.reltol <= 0.0 && options.abstol <= 0.0)
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("at least one of options['reltol'] and options['abstol'] must be positive"));
        
        if (options.feastol <= 0.0)
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("options['feastol'] must be a positive scalar"));
        
        
        // -- Choose the KKT Solver if not set
        if (kktpresolver == NULL) 
            if (dims.q.size() > 0 || dims.s.size() > 0 )
                //kktsolver = 'chol';
                BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("chol solver not avaible"));
            else 
                //kktsolver = 'chol2';
                BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("chol2 solver not avaible"));
        
        
        if (q.cols() != 1)
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("'q' must be a matrix with one column"));
        
        //    if (P.rows() != q.rows() || P.cols() != q.rows())
        //        BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("'P' must be a 'd' matrix of size (%d, %d)"));
        
        /*
         def fP(x, y, alpha = 1.0, beta = 0.0):
         base.symv(P, x, y, alpha = alpha, beta = beta)
         else:
         fP = P
         
         */
        
        if (_h && _h->cols() != 1)
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("'h' must be a 'd' matrix with one column"));
        
        if (dims.l == -1) {
            dims.l = _h ? _h->rows() : (_G ? _G->rows() : 0);
        }
        
        /*
         if (dims.l < 0) 
         raise TypeError("'dims.l' must be a nonnegative integer")
         if [ k for k in dims['q'] if type(k) is not int or k < 1 ]:
         raise TypeError("'dims['q']' must be a list of positive integers")
         if [ k for k in dims['s'] if type(k) is not int or k < 0 ]:
         raise TypeError("'dims['s']' must be a list of nonnegative " \
         "integers")
         */
        
        if (options.refinement == -1) {
            options.refinement = (dims.q.size() > 0 || dims.s.size() > 0) ? 1 : 0;
        }
        
        
        int cdim = dims.l;
        int dimsq = 0, dimss = 0;
        for(IntIterator i = dims.q.begin(); i != dims.q.end(); i++) { cdim += *i; dimsq += *i; }
        for(IntIterator i = dims.s.begin(); i != dims.q.end(); i++) { cdim += *i * *i; dimss += *i; }
        
        boost::shared_ptr<KQP_VECTOR(Scalar)> __h;
        KQP_VECTOR(Scalar) &h = _h ? *_h : *(__h = boost::shared_ptr<KQP_VECTOR(Scalar)>(new KQP_VECTOR(Scalar)(cdim)));
        if (!_h) h.setZero(cdim);
        if (h.rows() != cdim)
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message((boost::format("'h' must be a 'd' matrix of size (%d,1)") %cdim).str()));
        
        // Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
        std::vector<int> indq;
        
        indq.push_back(dims.l);
        
        for(IntIterator k = dims.q.begin(); k != dims.q.end(); k++)
            indq.push_back(indq.back() + *k);        
        
        
        // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
        std::vector<int> inds;
        if (!indq.empty()) inds.push_back(indq.back());
        
        for(IntIterator k = dims.s.begin(); k != dims.s.end(); k++)
            inds.push_back(inds.back() + *k * *k);
        
        boost::shared_ptr<KQP_MATRIX(Scalar)> __G;
        const QPMatrix<Scalar> &G = *_G; // _G ? *_G : *(__G = boost::shared_ptr<Matrix>(new  *(cdim, q.rows())));
        //    if (G.rows() != cdim || G.cols() != q.rows())
        //        BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message((boost::format("'G' must be a 'd' matrix of size (%d, %d)") % cdim % q.rows()).str()));
        
        
        boost::shared_ptr<KQP_MATRIX(Scalar)> __A;
        
        KQP_MATRIX(Scalar) &A = _A ? *_A : *(__A = boost::shared_ptr<KQP_MATRIX(Scalar)>(new KQP_MATRIX(Scalar)(0, q.rows())));
        
        if (A.cols() != q.rows())
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message((boost::format("'A' must be a 'd' matrix with %d columns") % q.rows()).str()));
        
        boost::shared_ptr<KQP_VECTOR(Scalar)> __b;
        KQP_VECTOR(Scalar) &b = _b ? *_b : *(__b = boost::shared_ptr<KQP_VECTOR(Scalar)>(new KQP_VECTOR(Scalar)(A.rows())));
        if (!_b) b.setZero(b.rows());
        
        if (b.rows() != A.rows())
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message((boost::format("'b' must have length %d") % A.rows()).str()));
        
        KQP_VECTOR(Scalar) ws3(cdim);
        KQP_VECTOR(Scalar) wz3(cdim);
        
        // HERE was res (moved to cvxopt_res)
        
        
        Scalar resx0 = std::max(1.0, q.norm());
        Scalar resy0 = std::max(1.0, b.norm());
        
        Scalar resz0 = std::max(1.0, snrm2(h, dims));
        
        ConeQPReturn<Scalar> &r = result;
        KQP_VECTOR(Scalar) &x = r.x, &y = r.y, &z = r.z, &s = r.s;
        
        // --- No inequalities in QP
        if (cdim == 0) {
            
            // Solve
            //
            //     [ P  A' ] [ x ]   [ -q ]
            //     [       ] [   ] = [    ].
            //     [ A  0  ] [ y ]   [  b ]
            
            boost::shared_ptr<KKTSolver<Scalar> > f3;
            try {
                ScalingMatrix<Scalar> w;
                KQP_LOG_DEBUG(logger, "Constructing a KKT solver");
                f3 = boost::shared_ptr<KKTSolver<Scalar> >(kktpresolver->get(w));
            } catch(...) {
                BOOST_THROW_EXCEPTION(arithmetic_exception() << errinfo_message("Rank(A) < p or Rank([P; A; G]) < n"));
            }
            
            x = -q;
            y = b;
            
            f3->solve(x, y, z);
            
            
            // dres = || P*x + q + A'*y || / resx0 
            KQP_VECTOR(Scalar) rx;
            P.mult(x, rx);
            Scalar pcost = 0.5 * (x.dot(rx) + x.dot(q));
            
            rx += A.transpose() * y;
            
            //Scalar dres = rx.norm() / resx0;
            
            // pres = || A*x - b || / resy0
            
            // KQP_VECTOR(Scalar) ry = A * x - b;
            //Scalar pres = ry.norm() / resy0;
            
            Scalar relgap;
            if (pcost == 0.0) relgap = NAN;
            else  relgap = 0.0;
            
            r.status = "optimal";      
            r.primal_objective = r.dual_objective = pcost;
            return;
        }
        
        
        // --- We have inequalities in the QP
        
        x = q;
        y = b;
        s = KQP_VECTOR(Scalar)(cdim);
        z = KQP_VECTOR(Scalar)(cdim);
        
        if (!initVals) {
            
            // Factor
            //
            //     [ P   A'  G' ] 
            //     [ A   0   0  ].
            //     [ G   0  -I  ]
            
            ScalingMatrix<Scalar> W; 
            W.d.setIdentity(dims.l);
            W.di = W.d;
            
            for(IntIterator m = dims.q.begin(); m != dims.q.end(); m++) 
                W.v.push_back(KQP_MATRIX(Scalar)(*m,1));
            
            W.beta.push_back(dims.q.size());
            for(typename std::vector<KQP_MATRIX(Scalar)>::iterator v = W.v.begin();  v != W.v.end(); v++) 
                (*v)(0,0) = 1.0;
            
            for(size_t i = 0; i < dims.s.size(); i++) { int m = dims.s[i];
                W.r.push_back(KQP_MATRIX(Scalar)(m,m));
                W.r.back().setIdentity();
                W.rti.push_back(KQP_MATRIX(Scalar)(m,m));
                W.rti.back().setIdentity();
            }
            
            boost::shared_ptr<KKTSolver<Scalar> > f;
            
            try {
                f = boost::shared_ptr<KKTSolver<Scalar> >(kktpresolver->get(W));   
            } catch(arithmetic_exception &e) {
                BOOST_THROW_EXCEPTION(arithmetic_exception() << errinfo_message("Rank(A) < p or Rank([P; A; G]) < n"));
            }
            
            
            // Solve
            //
            //     [ P   A'  G' ]   [ x ]   [ -q ]
            //     [ A   0   0  ] * [ y ] = [  b ].
            //     [ G   0  -I  ]   [ z ]   [  h ]
            
            x = - q; 
            y = b; 
            z = h;
            
            try {
                KQP_LOG_DEBUG(logger, "x=" << x.adjoint());
                KQP_LOG_DEBUG(logger, "z=" << z.adjoint());
                
                f->solve(x, y, z);
                KQP_LOG_DEBUG(logger, "x=" << x.adjoint());
                KQP_LOG_DEBUG(logger, "z=" << z.adjoint());
            } catch(arithmetic_exception &e) {
                BOOST_THROW_EXCEPTION(arithmetic_exception() << errinfo_message("Rank(A) < p or Rank([P; A; G]) < n"));
            }
            
            s = -z;
            
            
            // Ensures s is positive
            Scalar nrms = snrm2(s, dims);
            Scalar ts = max_step(s, dims);
            if (ts >= -1e-8 * std::max(nrms, 1.0)) {  
                Scalar a = 1.0 + ts;
                s.segment(0, dims.l).array() += a;
                
                for(int i = 0; i < indq.size() - 1; i++)
                    s[indq[i]] += a;
                
                int ind = dims.l + dimsq;
                for(size_t i = 0; i < dims.s.size(); i++) { 
                    int m = dims.s[i];
                    for(int i = ind; i < ind + m * m; i += m+1) s[i] += a;
                    ind += m * m;
                }
                
            }
            
            // ensures z is positive
            Scalar nrmz = snrm2(z, dims);
            Scalar tz = max_step(z, dims);
            if (tz >= -1e-8 * std::max(nrmz, 1.0)) {
                Scalar a = 1.0 + tz;
                z.segment(0, dims.l).array() += a;
                
                for(int i = 0; i < indq.size() - 1; i++) z[indq[i]] += a;
                int ind = dims.l + dimsq;
                for(size_t i = 0; i < dims.s.size(); i++) { 
                    int m = dims.s[i];
                    for(int i = ind; i < ind + m * m; i += m+1) 
                        z[i] += a;
                    ind += m * m;
                }
                
            }
            
            
        } else {
            // We have initialised values
            
            if (result.x.rows() > 0) x = result.x; 
            else x.setZero();
            
            if (result.s.rows()) {
                s = result.s;
                // ts = min{ t | s + t*e >= 0 }
                if (max_step(s, dims) >= 0.)
                    BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("initial s is not positive"));
                
            } else { 
                for(int i = 0; i < dims.l; i++) s[i] = 1.;
                
                int ind = dims.l;
                for(size_t i = 0; i < dims.q.size(); i++) {
                    int m = dims.q[i];
                    s[ind] = 1.0;
                    ind += m;
                }
                
                for(size_t i = 0; i < dims.s.size(); i++) { 
                    int m = dims.s[i];
                    for(int i = ind; i < ind+m*m; i += m+1) 
                        s[i] = 1.;
                    ind += m * m;
                }
            }
            
            if (result.y.rows()) y = result.y; else y.setZero();
            
            if (result.z.rows()) {
                z = result.z;
                // tz = min{ t | z + t*e >= 0 }
                if (max_step(z, dims) >= 0.)
                    BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("initial z is not positive"));
            } else {
                for(int i = 0; i < dims.l; i++) z[i] = 1.;
                int ind = dims.l;
                for(size_t i = 0; i < dims.q.size(); i++) { int m = dims.q[i];
                    z[ind] = 1.0;
                    ind += m;
                }
                
                for(size_t i = 0; i < dims.s.size(); i++) { int m = dims.s[i];
                    for(int i = ind; i < ind+m*m; i += m+1) 
                        z[i] = 1.;
                    ind += m * m;
                }
            }
        }
        
        
        // --- Start the optimisation
        
        KQP_VECTOR(Scalar) rx = q, ry = b, rz = KQP_VECTOR(Scalar)(cdim);
        KQP_VECTOR(Scalar) dx = x, dy = y;
        KQP_VECTOR(Scalar) dz = KQP_VECTOR(Scalar)(cdim), ds = KQP_VECTOR(Scalar)(cdim);
        
        KQP_VECTOR(Scalar) lmbda = KQP_VECTOR(Scalar)(dims.l + dimsq + dimss);
        KQP_VECTOR(Scalar) lmbdasq = lmbda;
        KQP_VECTOR(Scalar) sigs(dimss);
        KQP_VECTOR(Scalar) sigz(dimss);
        
        // Makes some references for fast access
        std::string &status = r.status;
        Scalar &pcost = r.primal_objective, &dcost = r.dual_objective, &dres = r.dual_infeasibility, &pres = r.primal_infeasibility;
        Scalar &gap = r.gap, &relgap = r.relative_gap;
        
        
        KQP_LOG_INFO(logger, boost::format("% 10s% 12s% 10s% 8s% 7s") % pcost % dcost % gap % pres % dres);
        
        gap = sdot(s, z, dims);
        
        int &iters = r.iterations;
        ScalingMatrix<Scalar> W;
        for(iters = 0; iters <= options.maxiters; iters++) {
            KQP_LOG_DEBUG(logger, "========= ITERATION " << convert(iters));
            KQP_LOG_DEBUG(logger, "x=" << convert(x.adjoint()));
            KQP_LOG_DEBUG(logger, "z=" << convert(z.adjoint()));
            KQP_LOG_DEBUG(logger, "s=" << convert(s.adjoint()));
            KQP_LOG_DEBUG(logger, "h=" << convert(h.adjoint()));
            
            // f0 = (1/2)*x'*P*x + q'*x + r and  rx = P*x + q + A'*y + G'*z.
            KQP_VECTOR(Scalar) rx;
            P.mult(x, rx);
            rx += q;
            Scalar f0 = 0.5 * ( x.dot(rx) + x.dot(q));
            
            KQP_VECTOR(Scalar) Gz;
            G.mult(z, Gz, true);
            
            rx += A.adjoint() * y + Gz;
            Scalar resx = rx.norm();
            
            // ry = A*x - b
            KQP_VECTOR(Scalar) ry = A * x - b;
            Scalar resy = ry.norm();
            
            // rz = s + G*x - h
            KQP_VECTOR(Scalar) rz;
            G.mult(x, rz);
            rz += s - h;
            Scalar resz = snrm2(rz, dims);
            
            
            
            // Statistics for stopping criteria.
            
            // pcost = (1/2)*x'*P*x + q'*x 
            // dcost = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h)
            //       = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h+s) - z'*s
            //       = (1/2)*x'*P*x + q'*x + y'*ry + z'*rz - gap
            pcost = f0;
            dcost = f0 + y.dot(ry) + sdot(z, rz, dims) - gap;
            if (pcost < 0.0)
                relgap = gap / -pcost;
            else if (dcost > 0.0)
                relgap = gap / dcost;
            else
                relgap = NAN;
            pres = std::max(resy/resy0, resz/resz0);
            dres = resx/resx0;
            
            
            KQP_LOG_INFO(logger, boost::format("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e") % iters % pcost % dcost % gap % pres % dres);
            
            if (( pres <= options.feastol && dres <= options.feastol && ( gap <= options.abstol || (!std::isnan(relgap) && relgap <= options.reltol) )) || iters == options.maxiters) {
                int ind = dims.l  + dimsq;
                for(size_t i = 0; i < dims.s.size(); i++) { int m = dims.s[i];
                    KQP_NOT_IMPLEMENTED;
                    //                misc.symm(s, m, ind);
                    //                misc.symm(z, m, ind);
                    ind += m * m;
                }
                Scalar ts = max_step(s, dims);
                Scalar tz = max_step(z, dims);
                if (iters == options.maxiters) {
                    if (options.show_progress)
                        std::cerr << "Terminated (maximum number of iterations reached)." << std::endl;
                    status = "unknown";
                }
                else {
                    if (options.show_progress)
                        std::cerr << "Optimal solution found." << std::endl;
                    status = "optimal";
                }
                r.dual_slack = -tz;
                r.primal_slack = -ts;
                return;
            }
            
            // Compute initial scaling W and scaled iterates:  
            //
            //     W * z = W^{-T} * s = lambda.
            // 
            // lmbdasq = lambda o lambda.
            
            if (iters == 0) {
                compute_scaling(W, s, z, lmbda, dims);
                KQP_LOG_DEBUG(logger, "lmbda=" << convert(lmbda.adjoint()));
            }
            
            KQP_LOG_DEBUG(logger, "scaling(d)=" << convert(W.d.diagonal().adjoint()));
            
            ssqr(lmbdasq, lmbda, dims);
            KQP_LOG_DEBUG(logger, "lmbdasq=" << convert(lmbdasq.adjoint()));
            
            
            // f3(x, y, z) solves
            //
            //    [ P   A'  G'    ] [ ux        ]   [ bx ]
            //    [ A   0   0     ] [ uy        ] = [ by ].
            //    [ G   0   -W'*W ] [ W^{-1}*uz ]   [ bz ]
            //
            // On entry, x, y, z containg bx, by, bz.
            // On exit, they contain ux, uy, uz.
            
            boost::shared_ptr<KKTSolver<Scalar> > f3; 
            try { 
                f3 = boost::shared_ptr<KKTSolver<Scalar> >(kktpresolver->get(W)); 
            } catch(arithmetic_exception &) {
                if (iters == 0)
                    BOOST_THROW_EXCEPTION(arithmetic_exception() << errinfo_message("Rank(A) < p or Rank([P; A; G]) < n"));
                else {
                    int ind = dims.l + dimsq;
                    for(size_t i = 0; i < dims.s.size(); i++) { int m = dims.s[i];
                        KQP_NOT_IMPLEMENTED;
                        //                    misc.symm(s, m, ind)
                        //                    misc.symm(z, m, ind)
                        //                    ind += m**2
                    }
                    r.primal_slack = -max_step(s, dims);
                    r.dual_slack = -max_step(z, dims);
                    std::cerr << "Terminated (singular KKT matrix).";
                    status = "unknown";
                    return;
                }
            }
            
            f4_no_ir_class<Scalar> f4_no_ir(W, *f3, dims, lmbda);
            
            
            KQP_VECTOR(Scalar) wx, wy, wz, ws;
            KQP_VECTOR(Scalar) wx2, wy2, wz2, ws2;
            
            if (iters == 0) {
                if (options.refinement > 0 || options.DEBUG) {               
                    wx = q;
                    wy = b;
                    wz = KQP_VECTOR(Scalar)(cdim);
                    ws = KQP_VECTOR(Scalar)(cdim);
                }
                if (options.refinement > 0) {
                    wx2 = q;
                    wy2 = b;
                    wz2 = KQP_VECTOR(Scalar)(cdim);
                    ws2 = KQP_VECTOR(Scalar)(cdim);
                }
            }
            
            f4_class<Scalar> f4(options.refinement, options.DEBUG, f4_no_ir, W, lmbda, dims, P, A, G);
            
            Scalar mu = gap / (dims.l + dims.q.size() + dimss);
            Scalar sigma = 0.;
            Scalar eta = 0.0;
            
            Scalar step = 0.0;
            
            for(size_t i = 0; i <= 1; i++) {
                
                // Solve
                //
                //     [ 0     ]   [ P  A' G' ]   [ dx        ]
                //     [ 0     ] + [ A  0  0  ] * [ dy        ] = -(1 - eta) * r
                //     [ W'*ds ]   [ G  0  0  ]   [ W^{-1}*dz ]
                //
                //     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e (i=0)
                //     lmbda o (dz + ds) = -lmbda o lmbda - dsa o dza 
                //                         + sigma*mu*e (i=1) where dsa, dza
                //                         are the solution for i=0. 
                
                // ds = -lmbdasq + sigma * mu * e  (if i is 0)
                //    = -lmbdasq - dsa o dza + sigma * mu * e  (if i is 1), 
                //     where ds, dz are solution for i is 0.
                ds.setZero();
                if (options.correction && i == 1)  
                    ds += -ws3;
                
                ds.topRows(dims.l + dimsq) += -lmbdasq.topRows(dims.l+dimsq);
                ds.topRows(dims.l).array() += sigma*mu;
                int ind =dims.l;
                for(size_t j = 0; j < dims.q.size(); j++) { int m = dims.q[j];
                    ds[ind] += sigma*mu;
                    ind += m;
                }
                int ind2 = ind;
                for(size_t j = 0; j < dims.s.size(); j++) { int m = dims.s[j];
                    KQP_NOT_IMPLEMENTED;
                    //                blas.axpy(lmbdasq, ds, n = m, offsetx = ind2, offsety =  
                    //                    ind, incy = m + 1, alpha = -1.0)
                    //                ds[ind : ind + m*m : m+1] += sigma*mu
                    //                ind += m*m
                    //                ind2 += m
                }
                
                
                // (dx, dy, dz) := -(1 - eta) * (rx, ry, rz)
                dx = -(1-eta) * rx;
                dy = -(1-eta) * ry;
                dz = -(1-eta) * rz;
                
                try {
                    KQP_LOG_DEBUG(logger, "[dx]=" << convert(dx.adjoint()));
                    KQP_LOG_DEBUG(logger, "[dz]=" << convert(dz.adjoint()));
                    KQP_LOG_DEBUG(logger, "[ds]=" << convert(ds.adjoint()));
                    f4(dx, dy, dz, ds); 
                }
                catch(arithmetic_exception &) {
                    if (iters == 0)
                        BOOST_THROW_EXCEPTION(arithmetic_exception() << errinfo_message("Rank(A) < p or Rank([P; A; G]) < n"));
                    
                    ind =dims.l + dimsq;
                    for(size_t j = 0; j < dims.s.size(); j++) { int m = dims.s[j];
                        KQP_NOT_IMPLEMENTED;
                        //                        misc.symm(s, m, ind)
                        //                        misc.symm(z, m, ind)
                        //                        ind += m**2
                    }
                    r.primal_slack = -max_step(s, dims);
                    r.dual_slack = -max_step(z, dims);
                    std::cerr << "Terminated (singular KKT matrix)." << std::endl;
                    status = "unknown";
                    return;
                    
                }
                
                Scalar dsdz = sdot(ds, dz, dims);
                
                // Save ds o dz for Mehrotra NULL
                if (options.correction && i == 0) {               
                    ws3 = ds;
                    sprod(ws3, dz, dims);
                }
                
                // Maximum steps to boundary.  
                // 
                // If i is 1, also compute eigenvalue decomposition of the 
                // 's' blocks in ds,dz.  The eigenvectors Qs, Qz are stored in 
                // dsk, dzk.  The eigenvalues are stored in sigs, sigz.
                
                scale2(lmbda, ds, dims);
                scale2(lmbda, dz, dims);
                Scalar ts, tz;
                if (i == 0) {
                    ts = max_step(ds, dims);
                    tz = max_step(dz, dims);
                } else {
                    ts = max_step(ds, dims, 0, &sigs);
                    tz = max_step(dz, dims, 0, &sigz);
                }
                
                
                Scalar t = std::max(std::max(0.0, ts), tz);
                if (t == 0)
                    step = 1.0;
                else
                    if (i == 0)
                        step = std::min(1.0, 1.0 / t);
                    else
                        step = std::min(1.0, STEP / t);
                if (i == 0) {
                    sigma = std::pow(std::min(1.0, std::max(0.0, 
                                                            1.0 - step + dsdz/gap * step*step)), EXPON);
                    eta = 0.0;
                }
            }
            
            
            KQP_LOG_DEBUG(logger, " STEP " << convert(step));
            KQP_LOG_DEBUG(logger, "   dx=" << convert(dx.adjoint()));
            KQP_LOG_DEBUG(logger, "   dy=" << convert(dx.adjoint()));
            
            x += step * dx;
            y += step * dy;
            
            
            // We will now replace the 'l' and 'q' blocks of ds and dz with 
            // the updated iterates in the current scaling.
            // We also replace the 's' blocks of ds and dz with the factors 
            // Ls, Lz in a factorization Ls*Ls', Lz*Lz' of the updated variables
            // in the current scaling.
            
            // ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
            // dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
            ds.topRows(dims.l + dimsq) *= step;
            dz.topRows(dims.l + dimsq) *= step;
            int ind =dims.l;
            ds.topRows(ind).array() += 1.0;
            dz.topRows(ind).array() += 1.0;
            for(IntIterator m = dims.q.begin(); m != dims.q.end(); m++) {
                ds[ind] += 1.0;
                dz[ind] += 1.0;
                ind += *m;
            }
            
            // ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
            //
            // This replaced the 'l' and 'q' components of ds and dz with the
            // updated iterates in the current scaling.
            // The 's' components of ds and dz are replaced with
            //
            //     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
            //     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
            // 
            scale2(lmbda, ds, dims, false, true);
            scale2(lmbda, dz, dims, false, true);
            
            // sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
            // sigz := ( e + step*sigz ) ./ lmabda for 's' blocks.
            sigs = (1.0 + step * sigs.array()) / lmbda.segment(dims.l + dimsq, dimss).array();
            sigz = (1.0 + step * sigz.array()) / lmbda.segment(dims.l + dimsq, dimss).array();
            
            
            // dsk := Ls = dsk * sqrt(sigs).
            // dzk := Lz = dzk * sqrt(sigz).
            int ind2 = dims.l + dimsq;
            int ind3 = 0;
            for(int k = 0; k < dims.s.size(); k++) {
                KQP_NOT_IMPLEMENTED;
                //            m = dims['s'][k]
                //            for i in xrange(m):
                //                blas.scal(math.sqrt(sigs[ind3+i]), ds, offset = ind2 + m*i,
                //                    n = m)
                //                blas.scal(math.sqrt(sigz[ind3+i]), dz, offset = ind2 + m*i,
                //                    n = m)
                //            ind2 += m*m
                //            ind3 += m
            }
            
            
            // Update lambda and scaling.
            update_scaling(dims, W, lmbda, ds, dz);
            
            
            // Unscale s, z (unscaled variables are used only to compute 
            // feasibility residuals).
            
            s.topRows(dims.l + dimsq) = lmbda.topRows(dims.l + dimsq);
            ind =dims.l + dimsq;
            ind2 = ind;
            for(IntIterator m = dims.s.begin(); m != dims.s.end(); m++) {
                KQP_NOT_IMPLEMENTED;
                //            blas.scal(0.0, s, offset = ind2)
                //            blas.copy(lmbda, s, offsetx = ind, offsety = ind2, n = m, 
                //                incy = m+1)
                //            ind += m
                //            ind2 += m*m
            }
            
            scale(s, W, true, false);
            
            z.topRows(dims.l + dimsq) = lmbda.topRows(dims.l + dimsq);
            ind =dims.l + dimsq;
            ind2 = ind;
            for(IntIterator m = dims.s.begin(); m != dims.s.end(); m++) {
                KQP_NOT_IMPLEMENTED;
                //            blas.scal(0.0, z, offset = ind2)
                //            blas.copy(lmbda, z, offsetx = ind, offsety = ind2, n = m, 
                //                incy = m+1)
                //            ind += m
                //            ind2 += m*m
            }
            scale(z, W, false, true);
            
            gap = lmbda.squaredNorm();
            
        }
    }
    
    
    
    
    /*
     
     
     
     
     if use_C:
     pack = misc_solvers.pack
     else:
     def pack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0):
     """
     Copy x to y using packed storage.
     
     The vector x is an element of S, with the 's' components stored in 
     unpacked storage.  On return, x is copied to y with the 's' components
     stored in packed storage and the off-diagonal entries scaled by 
     sqrt(2).
     """
     
     nlq = mnl +dims.l + dimsq
     blas.copy(x, y, n = nlq, offsetx = offsetx, offsety = offsety)
     iu, ip = offsetx + nlq, offsety + nlq
     for n in dims['s']:
     for k in xrange(n):
     blas.copy(x, y, n = n-k, offsetx = iu + k*(n+1), offsety = ip)
     y[ip] /= math.sqrt(2)
     ip += n-k
     iu += n**2 
     np = sum([ n*(n+1)/2 for n in dims['s'] ])
     blas.scal(math.sqrt(2.0), y, n = np, offset = offsety+nlq)
     
     
     if use_C:
     pack2 = misc_solvers.pack2
     else:
     def pack2(x, dims, mnl = 0):
     """
     In-place version of pack(), which also accepts matrix arguments x.  
     The columns of x are elements of S, with the 's' components stored in
     unpacked storage.  On return, the 's' components are stored in packed
     storage and the off-diagonal entries are scaled by sqrt(2).
     """
     
     if not dims['s']: return
     iu = mnl +dims.l + dimsq
     ip = iu
     for n in dims['s']:
     for k in xrange(n):
     x[ip, :] = x[iu + (n+1)*k, :]
     x[ip + 1 : ip+n-k, :] = x[iu + (n+1)*k + 1: iu + n*(k+1), :] \
     * math.sqrt(2.0)
     ip += n - k
     iu += n**2 
     np = sum([ n*(n+1)/2 for n in dims['s'] ])
     
     
     if use_C:
     unpack = misc_solvers.unpack
     else:
     def unpack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0):
     """
     The vector x is an element of S, with the 's' components stored
     in unpacked storage and off-diagonal entries scaled by sqrt(2).
     On return, x is copied to y with the 's' components stored in 
     unpacked storage.
     """
     
     nlq = mnl +dims.l + dimsq
     blas.copy(x, y, n = nlq, offsetx = offsetx, offsety = offsety)
     iu, ip = offsety+nlq, offsetx+nlq
     for n in dims['s']:
     for k in xrange(n):
     blas.copy(x, y, n = n-k, offsetx = ip, offsety = iu+k*(n+1))
     y[iu+k*(n+1)] *= math.sqrt(2)
     ip += n-k
     iu += n**2 
     nu = sum([ n**2 for n in dims['s'] ])
     blas.scal(1.0/math.sqrt(2.0), y, n = nu, offset = offsety+nlq)
     
     
     
     
     def sdot2(x, y):
     """
     Inner product of two block-diagonal symmetric dense 'd' matrices.
     
     x and y are square dense 'd' matrices, or lists of N square dense 'd' 
     matrices.
     """
     
     a = 0.0
     if type(x) is matrix:
     n = x.size[0]
     a += blas.dot(x, y, incx=n+1, incy=n+1, n=n)
     for j in xrange(1,n):
     a += 2.0 * blas.dot(x, y, incx=n+1, incy=n+1, offsetx=j,
     offsety=j, n=n-j)
     
     else:
     for k in xrange(len(x)):
     n = x[k].size[0]
     a += blas.dot(x[k], y[k], incx=n+1, incy=n+1, n=n)
     for j in xrange(1,n):
     a += 2.0 * blas.dot(x[k], y[k], incx=n+1, incy=n+1, 
     offsetx=j, offsety=j, n=n-j)
     return a
     
     
     
     
     
     if use_C:
     trisc = misc_solvers.trisc
     else:
     def trisc(x, dims, offset = 0): 
     """
     Sets upper triangular part of the 's' components of x equal to zero
     and scales the strictly lower triangular part by 2.0.
     """ 
     
     m =dims.l + dimsq + sum([ k**2 for k in dims['s'] ]) 
     ind = offset +dims.l + dimsq
     for mk in dims['s']:
     for j in xrange(1, mk):  
     blas.scal(0.0, x, n = mk-j, inc = mk, offset = 
     ind + j*(mk + 1) - 1) 
     blas.scal(2.0, x, offset = ind + mk*(j-1) + j, n = mk-j) 
     ind += mk**2
     
     
     if use_C:
     triusc = misc_solvers.triusc
     else:
     def triusc(x, dims, offset = 0):
     """
     Scales the strictly lower triangular part of the 's' components of x 
     by 0.5.
     """ 
     
     m =dims.l + dimsq + sum([ k**2 for k in dims['s'] ]) 
     ind = offset +dims.l + dimsq
     for mk in dims['s']:
     for j in xrange(1, mk):  
     blas.scal(0.5, x, offset = ind + mk*(j-1) + j, n = mk-j) 
     ind += mk**2
     
     
     def sgemv(A, x, y, dims, trans = 'N', alpha = 1.0, beta = 0.0, n = None, 
     offsetA = 0, offsetx = 0, offsety = 0): 
     """
     KQP_MATRIX(Scalar)-vector multiplication.
     
     A is a matrix or spmatrix of size (m, n) where 
     
     N =dims.l + dimsq + sum( k**2 for k in dims['s'] ) 
     
     representing a mapping from R^n to S.  
     
     If trans is 'N': 
     
     y := alpha*A*x + beta * y   (trans = 'N').
     
     x is a vector of length n.  y is a vector of length N.
     
     If trans is 'T':
     
     y := alpha*A'*x + beta * y  (trans = 'T').
     
     x is a vector of length N.  y is a vector of length n.
     
     The 's' components in S are stored in unpacked 'L' storage.
     """
     
     m =dims.l + dimsq + sum([ k**2 for k in dims['s'] ]) 
     if n is None: n = A.size[1]
     if trans == 'T' and alpha:  trisc(x, dims, offsetx)
     base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta, m = m,
     n = n, offsetA = offsetA, offsetx = offsetx, offsety = offsety)
     if trans == 'T' and alpha: triusc(x, dims, offsetx)
     
     
     def jdot(x, y, n = None, offsetx = 0, offsety = 0):
     """
     Returns x' * J * y, where J = [1, 0; 0, -I].
     """
     
     if n is None: 
     if len(x) != len(y): raise ValueError("x and y must have the "\
     "same length")
     n = len(x)
     return x[offsetx] * y[offsety] - blas.dot(x, y, n = n-1, 
     offsetx = offsetx + 1, offsety = offsety + 1) 
     
     
     def jnrm2(x, n = None, offset = 0):
     """
     Returns sqrt(x' * J * x) where J = [1, 0; 0, -I], for a vector
     x in a second order cone. 
     """
     
     if n is None:  n = len(x)
     a = blas.nrm2(x, n = n-1, offset = offset+1)
     return math.sqrt(x[offset] - a) * math.sqrt(x[offset] + a)
     
     
     
     
     
     
     
     
     def kkt_ldl(G, dims, A, mnl = 0):
     """
     Solution of KKT equations by a dense LDL factorization of the 
     3 x 3 system.
     
     Returns a function that (1) computes the LDL factorization of
     
     [ H           A'   GG'*W^{-1} ] 
     [ A           0    0          ],
     [ W^{-T}*GG   0   -I          ] 
     
     given H, Df, W, where GG = [Df; G], and (2) returns a function for 
     solving 
     
     [ H     A'   GG'   ]   [ ux ]   [ bx ]
     [ A     0    0     ] * [ uy ] = [ by ].
     [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
     
     H is n x n,  A is p x n, Df is mnl x n, G is N x n where
     N =dims.l + dimsq + sum( k**2 for k in dims['s'] ).
     """
     
     p, n = A.size
     ldK = n + p + mnl +dims.l + dimsq + sum([ k*(k+1)/2 for k 
     in dims['s'] ])
     K = matrix(0.0, (ldK, ldK))
     ipiv = matrix(0, (ldK, 1))
     u = matrix(0.0, (ldK, 1))
     g = matrix(0.0, (mnl + G.size[0], 1))
     
     def factor(W, H = None, Df = None):
     
     blas.scal(0.0, K)
     if H is not None: K[:n, :n] = H
     K[n:n+p, :n] = A
     for k in xrange(n):
     if mnl: g[:mnl] = Df[:,k]
     g[mnl:] = G[:,k]
     scale(g, W, trans = 'T', inverse = 'I')
     pack(g, K, dims, mnl, offsety = k*ldK + n + p)
     K[(ldK+1)*(p+n) :: ldK+1]  = -1.0
     lapack.sytrf(K, ipiv)
     
     def solve(x, y, z):
     
     // Solve
     //
     //     [ H          A'   GG'*W^{-1} ]   [ ux   ]   [ bx        ]
     //     [ A          0    0          ] * [ uy   [ = [ by        ]
     //     [ W^{-T}*GG  0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]
     //
     // and return ux, uy, W*uz.
     //
     // On entry, x, y, z contain bx, by, bz.  On exit, they contain
     // the solution ux, uy, W*uz.
     
     blas.copy(x, u)
     blas.copy(y, u, offsety = n)
     scale(z, W, trans = 'T', inverse = 'I') 
     pack(z, u, dims, mnl, offsety = n + p)
     lapack.sytrs(K, ipiv, u)
     blas.copy(u, x, n = n)
     blas.copy(u, y, offsetx = n, n = p)
     unpack(u, z, dims, mnl, offsetx = n + p)
     
     return solve
     
     return factor
     
     
     def kkt_ldl2(G, dims, A, mnl = 0):
     """
     Solution of KKT equations by a dense LDL factorization of the 2 x 2 
     system.
     
     Returns a function that (1) computes the LDL factorization of
     
     [ H + GG' * W^{-1} * W^{-T} * GG   A' ]
     [                                     ]
     [ A                                0  ]
     
     given H, Df, W, where GG = [Df; G], and (2) returns a function for 
     solving 
     
     [ H    A'   GG'   ]   [ ux ]   [ bx ]
     [ A    0    0     ] * [ uy ] = [ by ].
     [ GG   0   -W'*W  ]   [ uz ]   [ bz ]
     
     H is n x n,  A is p x n, Df is mnl x n, G is N x n where
     N =dims.l + dimsq + sum( k**2 for k in dims['s'] ).
     """
     
     p, n = A.size
     ldK = n + p 
     K = matrix(0.0, (ldK, ldK))
     if p: ipiv = matrix(0, (ldK, 1))
     g = matrix(0.0, (mnl + G.size[0], 1))
     u = matrix(0.0, (ldK, 1))
     
     def factor(W, H = None, Df = None):
     
     blas.scal(0.0, K)
     if H is not None: K[:n, :n] = H
     K[n:,:n] = A
     for k in xrange(n):
     if mnl: g[:mnl] = Df[:,k]
     g[mnl:] = G[:,k]
     scale(g, W, trans = 'T', inverse = 'I')
     scale(g, W, inverse = 'I')
     if mnl: base.gemv(Df, g, K, trans = 'T', beta = 1.0, n = n-k, 
     offsetA = mnl*k, offsety = (ldK + 1)*k)
     sgemv(G, g, K, dims, trans = 'T', beta = 1.0, n = n-k,
     offsetA = G.size[0]*k, offsetx = mnl, offsety = 
     (ldK + 1)*k)
     if p: lapack.sytrf(K, ipiv)
     else: lapack.potrf(K)
     
     def solve(x, y, z):
     
     // Solve
     //
     //     [ H + GG' * W^{-1} * W^{-T} * GG    A' ]   [ ux ]   
     //     [                                      ] * [    ] 
     //     [ A                                 0  ]   [ uy ]   
     //
     //         [ bx + GG' * W^{-1} * W^{-T} * bz ]
     //     =   [                                 ]
     //         [ by                              ]
     //
     // and return x, y, W*z = W^{-T} * (GG*x - bz).
     
     blas.copy(z, g)
     scale(g, W, trans = 'T', inverse = 'I')
     scale(g, W, inverse = 'I')
     if mnl: 
     base.gemv(Df, g, u, trans = 'T')
     beta = 1.0
     else: 
     beta = 0.0
     sgemv(G, g, u, dims, trans = 'T', offsetx = mnl, beta = beta)
     blas.axpy(x, u)
     blas.copy(y, u, offsety = n)
     if p: lapack.sytrs(K, ipiv, u)
     else: lapack.potrs(K, u)
     blas.copy(u, x, n = n)
     blas.copy(u, y, offsetx = n, n = p)
     if mnl: base.gemv(Df, x, z, alpha = 1.0, beta = -1.0)
     sgemv(G, x, z, dims, alpha = 1.0, beta = -1.0, offsety = mnl)
     scale(z, W, trans = 'T', inverse = 'I')
     
     return solve
     
     return factor
     
     
     def kkt_chol(G, dims, A, mnl = 0):
     """
     Solution of KKT equations by reduction to a 2 x 2 system, a QR 
     factorization to eliminate the equality constraints, and a dense 
     Cholesky factorization of order n-p. 
     
     Computes the QR factorization
     
     A' = [Q1, Q2] * [R; 0]
     
     and returns a function that (1) computes the Cholesky factorization 
     
     Q_2^T * (H + GG^T * W^{-1} * W^{-T} * GG) * Q2 = L * L^T, 
     
     given H, Df, W, where GG = [Df; G], and (2) returns a function for 
     solving 
     
     [ H    A'   GG'    ]   [ ux ]   [ bx ]
     [ A    0    0      ] * [ uy ] = [ by ].
     [ GG   0    -W'*W  ]   [ uz ]   [ bz ]
     
     H is n x n,  A is p x n, Df is mnl x n, G is N x n where
     N =dims.l + dimsq + sum( k**2 for k in dims['s'] ).
     """
     
     p, n = A.size
     cdim = mnl +dims.l + dimsq + sum([ k**2 for k in 
     dims['s'] ])
     cdim_pckd = mnl +dims.l + dimsq + sum([ k*(k+1)/2 for k 
     in dims['s'] ])
     
     // A' = [Q1, Q2] * [R; 0]  (Q1 is n x p, Q2 is n x n-p).
     if type(A) is matrix: 
     QA = A.T
     else: 
     QA = matrix(A.T)
     tauA = matrix(0.0, (p,1))
     lapack.geqrf(QA, tauA)
     
     Gs = matrix(0.0, (cdim, n))
     K = matrix(0.0, (n,n)) 
     bzp = matrix(0.0, (cdim_pckd, 1))
     yy = matrix(0.0, (p,1))
     
     def factor(W, H = None, Df = None):
     
     // Compute 
     //
     //     K = [Q1, Q2]' * (H + GG' * W^{-1} * W^{-T} * GG) * [Q1, Q2]
     //
     // and take the Cholesky factorization of the 2,2 block
     //
     //     Q_2' * (H + GG^T * W^{-1} * W^{-T} * GG) * Q2.
     
     // Gs = W^{-T} * GG in packed storage.
     if mnl: 
     Gs[:mnl, :] = Df
     Gs[mnl:, :] = G
     scale(Gs, W, trans = 'T', inverse = 'I')
     pack2(Gs, dims, mnl)
     
     // K = [Q1, Q2]' * (H + Gs' * Gs) * [Q1, Q2].
     blas.syrk(Gs, K, k = cdim_pckd, trans = 'T')
     if H is not None: K[:,:] += H
     symm(K, n)
     lapack.ormqr(QA, tauA, K, side = 'L', trans = 'T')
     lapack.ormqr(QA, tauA, K, side = 'R')
     
     // Cholesky factorization of 2,2 block of K.
     lapack.potrf(K, n = n-p, offsetA = p*(n+1))
     
     def solve(x, y, z):
     
     // Solve
     //
     //     [ 0          A'  GG'*W^{-1} ]   [ ux   ]   [ bx        ]
     //     [ A          0   0          ] * [ uy   ] = [ by        ]
     //     [ W^{-T}*GG  0   -I         ]   [ W*uz ]   [ W^{-T}*bz ]
     //
     // and return ux, uy, W*uz.
     //
     // On entry, x, y, z contain bx, by, bz.  On exit, they contain
     // the solution ux, uy, W*uz.
     //
     // If we change variables ux = Q1*v + Q2*w, the system becomes 
     // 
     //     [ K11 K12 R ]   [ v  ]   [Q1'*(bx+GG'*W^{-1}*W^{-T}*bz)]
     //     [ K21 K22 0 ] * [ w  ] = [Q2'*(bx+GG'*W^{-1}*W^{-T}*bz)]
     //     [ R^T 0   0 ]   [ uy ]   [by                           ]
     // 
     //     W*uz = W^{-T} * ( GG*ux - bz ).
     
     // bzp := W^{-T} * bz in packed storage 
     scale(z, W, trans = 'T', inverse = 'I')
     pack(z, bzp, dims, mnl)
     
     // x := [Q1, Q2]' * (x + Gs' * bzp)
     //    = [Q1, Q2]' * (bx + Gs' * W^{-T} * bz)
     blas.gemv(Gs, bzp, x, beta = 1.0, trans = 'T', m = cdim_pckd)
     lapack.ormqr(QA, tauA, x, side = 'L', trans = 'T')
     
     // y := x[:p] 
     //    = Q1' * (bx + Gs' * W^{-T} * bz)
     blas.copy(y, yy)
     blas.copy(x, y, n = p)
     
     // x[:p] := v = R^{-T} * by 
     blas.copy(yy, x)
     lapack.trtrs(QA, x, uplo = 'U', trans = 'T', n = p)
     
     // x[p:] := K22^{-1} * (x[p:] - K21*x[:p])
     //        = K22^{-1} * (Q2' * (bx + Gs' * W^{-T} * bz) - K21*v)
     blas.gemv(K, x, x, alpha = -1.0, beta = 1.0, m = n-p, n = p,
     offsetA = p, offsety = p)
     lapack.potrs(K, x, n = n-p, offsetA = p*(n+1), offsetB = p)
     
     // y := y - [K11, K12] * x
     //    = Q1' * (bx + Gs' * W^{-T} * bz) - K11*v - K12*w
     blas.gemv(K, x, y, alpha = -1.0, beta = 1.0, m = p, n = n)
     
     // y := R^{-1}*y
     //    = R^{-1} * (Q1' * (bx + Gs' * W^{-T} * bz) - K11*v 
     //      - K12*w)
     lapack.trtrs(QA, y, uplo = 'U', n = p)
     
     // x := [Q1, Q2] * x
     lapack.ormqr(QA, tauA, x, side = 'L')
     
     // bzp := Gs * x - bzp.
     //      = W^{-T} * ( GG*ux - bz ) in packed storage.
     // Unpack and copy to z.
     blas.gemv(Gs, x, bzp, alpha = 1.0, beta = -1.0, m = cdim_pckd)
     unpack(bzp, z, dims, mnl)
     
     return solve
     
     return factor
     
     
     def kkt_chol2(G, dims, A, mnl = 0):
     """
     Solution of KKT equations by reduction to a 2 x 2 system, a sparse 
     or dense Cholesky factorization of order n to eliminate the 1,1 
     block, and a sparse or dense Cholesky factorization of order p.
     Implemented only for problems with no second-order or semidefinite
     cone constraints.
     
     Returns a function that (1) computes Cholesky factorizations of
     the matrices 
     
     S = H + GG' * W^{-1} * W^{-T} * GG,  
     K = A * S^{-1} *A'
     
     or (if K is singular in the first call to the function), the matrices
     
     S = H + GG' * W^{-1} * W^{-T} * GG + A' * A,  
     K = A * S^{-1} * A',
     
     given H, Df, W, where GG = [Df; G], and (2) returns a function for 
     solving 
     
     [ H     A'   GG'   ]   [ ux ]   [ bx ]
     [ A     0    0     ] * [ uy ] = [ by ].
     [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
     
     H is n x n,  A is p x n, Df is mnl x n, G isdims.l x n.
     """
     
     if dims['q'] or dims['s']:
     raise ValueError("kktsolver option 'kkt_chol2' is implemented "\
     "only for problems with no second-order or semidefinite cone "\
     "constraints")
     p, n = A.size
     ml =dims.l
     F = {'firstcall': True, 'singular': False}
     
     def factor(W, H = None, Df = None):
     
     if F['firstcall']:
     if type(G) is matrix: 
     F['Gs'] = matrix(0.0, G.size) 
     else:
     F['Gs'] = spmatrix(0.0, G.I, G.J, G.size) 
     if mnl:
     if type(Df) is matrix:
     F['Dfs'] = matrix(0.0, Df.size) 
     else: 
     F['Dfs'] = spmatrix(0.0, Df.I, Df.J, Df.size) 
     if (mnl and type(Df) is matrix) or type(G) is matrix or \
     type(H) is matrix:
     F['S'] = matrix(0.0, (n,n))
     F['K'] = matrix(0.0, (p,p))
     else:
     F['S'] = spmatrix([], [], [], (n,n), 'd')
     F['Sf'] = None
     if type(A) is matrix:
     F['K'] = matrix(0.0, (p,p))
     else:
     F['K'] = spmatrix([], [], [], (p,p), 'd')
     
     // Dfs = Wnl^{-1} * Df 
     if mnl: base.gemm(spmatrix(W['dnli'], range(mnl), range(mnl)), Df, 
     F['Dfs'], partial = True)
     
     // Gs = Wl^{-1} * G.
     base.gemm(spmatrix(W['di'], range(ml), range(ml)), G, F['Gs'], 
     partial = True)
     
     if F['firstcall']:
     base.syrk(F['Gs'], F['S'], trans = 'T') 
     if mnl: 
     base.syrk(F['Dfs'], F['S'], trans = 'T', beta = 1.0)
     if H is not None: 
     F['S'] += H
     try:
     if type(F['S']) is matrix: 
     lapack.potrf(F['S']) 
     else:
     F['Sf'] = cholmod.symbolic(F['S'])
     cholmod.numeric(F['S'], F['Sf'])
     except ArithmeticError:
     F['singular'] = True 
     if type(A) is matrix and type(F['S']) is spmatrix:
     F['S'] = matrix(0.0, (n,n))
     base.syrk(F['Gs'], F['S'], trans = 'T') 
     if mnl:
     base.syrk(F['Dfs'], F['S'], trans = 'T', beta = 1.0)
     base.syrk(A, F['S'], trans = 'T', beta = 1.0) 
     if H is not None:
     F['S'] += H
     if type(F['S']) is matrix: 
     lapack.potrf(F['S']) 
     else:
     F['Sf'] = cholmod.symbolic(F['S'])
     cholmod.numeric(F['S'], F['Sf'])
     F['firstcall'] = False
     
     else:
     base.syrk(F['Gs'], F['S'], trans = 'T', partial = True)
     if mnl: base.syrk(F['Dfs'], F['S'], trans = 'T', beta = 1.0, 
     partial = True)
     if H is not None:
     F['S'] += H
     if F['singular']:
     base.syrk(A, F['S'], trans = 'T', beta = 1.0, partial = 
     True) 
     if type(F['S']) is matrix: 
     lapack.potrf(F['S']) 
     else:
     cholmod.numeric(F['S'], F['Sf'])
     
     if type(F['S']) is matrix: 
     // Asct := L^{-1}*A'.  Factor K = Asct'*Asct.
     if type(A) is matrix: 
     Asct = A.T
     else: 
     Asct = matrix(A.T)
     blas.trsm(F['S'], Asct)
     blas.syrk(Asct, F['K'], trans = 'T')
     lapack.potrf(F['K'])
     
     else:
     // Asct := L^{-1}*P*A'.  Factor K = Asct'*Asct.
     if type(A) is matrix:
     Asct = A.T
     cholmod.solve(F['Sf'], Asct, sys = 7)
     cholmod.solve(F['Sf'], Asct, sys = 4)
     blas.syrk(Asct, F['K'], trans = 'T')
     lapack.potrf(F['K']) 
     else:
     Asct = cholmod.spsolve(F['Sf'], A.T, sys = 7)
     Asct = cholmod.spsolve(F['Sf'], Asct, sys = 4)
     base.syrk(Asct, F['K'], trans = 'T')
     Kf = cholmod.symbolic(F['K'])
     cholmod.numeric(F['K'], Kf)
     
     def solve(x, y, z):
     
     // Solve
     //
     //     [ H          A'  GG'*W^{-1} ]   [ ux   ]   [ bx        ]
     //     [ A          0   0          ] * [ uy   ] = [ by        ]
     //     [ W^{-T}*GG  0   -I         ]   [ W*uz ]   [ W^{-T}*bz ]
     //
     // and return ux, uy, W*uz.
     //
     // If not F['singular']:
     //
     //     K*uy = A * S^{-1} * ( bx + GG'*W^{-1}*W^{-T}*bz ) - by
     //     S*ux = bx + GG'*W^{-1}*W^{-T}*bz - A'*uy
     //     W*uz = W^{-T} * ( GG*ux - bz ).
     //    
     // If F['singular']:
     //
     //     K*uy = A * S^{-1} * ( bx + GG'*W^{-1}*W^{-T}*bz + A'*by )
     //            - by
     //     S*ux = bx + GG'*W^{-1}*W^{-T}*bz + A'*by - A'*y.
     //     W*uz = W^{-T} * ( GG*ux - bz ).
     
     // z := W^{-1} * z = W^{-1} * bz
     scale(z, W, trans = 'T', inverse = 'I') 
     
     // If not F['singular']:
     //     x := L^{-1} * P * (x + GGs'*z)
     //        = L^{-1} * P * (x + GG'*W^{-1}*W^{-T}*bz)
     //
     // If F['singular']:
     //     x := L^{-1} * P * (x + GGs'*z + A'*y))
     //        = L^{-1} * P * (x + GG'*W^{-1}*W^{-T}*bz + A'*y)
     
     if mnl: base.gemv(F['Dfs'], z, x, trans = 'T', beta = 1.0)
     base.gemv(F['Gs'], z, x, offsetx = mnl, trans = 'T', 
     beta = 1.0)
     if F['singular']:
     base.gemv(A, y, x, trans = 'T', beta = 1.0)
     if type(F['S']) is matrix:
     blas.trsv(F['S'], x)
     else:
     cholmod.solve(F['Sf'], x, sys = 7)
     cholmod.solve(F['Sf'], x, sys = 4)
     
     // y := K^{-1} * (Asc*x - y)
     //    = K^{-1} * (A * S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz) - by)
     //      (if not F['singular'])
     //    = K^{-1} * (A * S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz + 
     //      A'*by) - by)  
     //      (if F['singular']).
     
     base.gemv(Asct, x, y, trans = 'T', beta = -1.0)
     if type(F['K']) is matrix:
     lapack.potrs(F['K'], y)
     else:
     cholmod.solve(Kf, y)
     
     // x := P' * L^{-T} * (x - Asc'*y)
     //    = S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz - A'*y) 
     //      (if not F['singular'])  
     //    = S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz + A'*by - A'*y) 
     //      (if F['singular'])
     
     base.gemv(Asct, y, x, alpha = -1.0, beta = 1.0)
     if type(F['S']) is matrix:
     blas.trsv(F['S'], x, trans='T')
     else:
     cholmod.solve(F['Sf'], x, sys = 5)
     cholmod.solve(F['Sf'], x, sys = 8)
     
     // W*z := GGs*x - z = W^{-T} * (GG*x - bz)
     if mnl:
     base.gemv(F['Dfs'], x, z, beta = -1.0)
     base.gemv(F['Gs'], x, z, beta = -1.0, offsety = mnl)
     
     return solve
     
     return factor
     
     
     / **
     Solution of KKT equations with zero 1,1 block, by eliminating the
     equality constraints via a QR factorization, and solving the
     reduced KKT system by another QR factorization.
     
     Computes the QR factorization
     
     A' = [Q1, Q2] * [R1; 0]
     
     and returns a function that (1) computes the QR factorization 
     
     W^{-T} * G * Q2 = Q3 * R3
     
     (with columns of W^{-T}*G in packed storage), and (2) returns a 
     function for solving 
     
     [ 0    A'   G'    ]   [ ux ]   [ bx ]
     [ A    0    0     ] * [ uy ] = [ by ].
     [ G    0   -W'*W  ]   [ uz ]   [ bz ]
     
     A is p x n and G is N x n where N =dims.l + dimsq + 
     sum( k**2 for k in dims['s'] ).
     
     * /
     void kkt_qr(G, dims, A) {
     
     p, n = A.size
     cdim =dims.l + dimsq + sum([ k**2 for k in dims['s'] ])
     cdim_pckd =dims.l + dimsq + sum([ k*(k+1)/2 for k in 
     dims['s'] ])
     
     // A' = [Q1, Q2] * [R1; 0]
     if type(A) is matrix:
     QA = +A.T
     else:
     QA = matrix(A.T)
     tauA = matrix(0.0, (p,1))
     lapack.geqrf(QA, tauA)
     
     Gs = matrix(0.0, (cdim, n))
     tauG = matrix(0.0, (n-p,1))
     u = matrix(0.0, (cdim_pckd, 1))
     vv = matrix(0.0, (n,1))
     w = matrix(0.0, (cdim_pckd, 1))
     
     def factor(W):
     
     // Gs = W^{-T}*G, in packed storage.
     Gs[:,:] = G
     scale(Gs, W, trans = 'T', inverse = 'I')
     pack2(Gs, dims)
     
     // Gs := [ Gs1, Gs2 ] 
     //     = Gs * [ Q1, Q2 ]
     lapack.ormqr(QA, tauA, Gs, side = 'R', m = cdim_pckd)
     
     // QR factorization Gs2 := [ Q3, Q4 ] * [ R3; 0 ] 
     lapack.geqrf(Gs, tauG, n = n-p, m = cdim_pckd, offsetA = 
     Gs.size[0]*p)
     
     def solve(x, y, z):
     
     // On entry, x, y, z contain bx, by, bz.  On exit, they 
     // contain the solution x, y, W*z of
     //
     //     [ 0         A'  G'*W^{-1} ]   [ x   ]   [bx       ]
     //     [ A         0   0         ] * [ y   ] = [by       ].
     //     [ W^{-T}*G  0   -I        ]   [ W*z ]   [W^{-T}*bz]
     //
     // The system is solved in five steps:
     //
     //       w := W^{-T}*bz - Gs1*R1^{-T}*by 
     //       u := R3^{-T}*Q2'*bx + Q3'*w
     //     W*z := Q3*u - w
     //       y := R1^{-1} * (Q1'*bx - Gs1'*(W*z))
     //       x := [ Q1, Q2 ] * [ R1^{-T}*by;  R3^{-1}*u ]
     
     // w := W^{-T} * bz in packed storage 
     scale(z, W, trans = 'T', inverse = 'I')
     pack(z, w, dims)
     
     // vv := [ Q1'*bx;  R3^{-T}*Q2'*bx ]
     blas.copy(x, vv)
     lapack.ormqr(QA, tauA, vv, trans='T') 
     lapack.trtrs(Gs, vv, uplo = 'U', trans = 'T', n = n-p, offsetA
     = Gs.size[0]*p, offsetB = p)
     
     // x[:p] := R1^{-T} * by 
     blas.copy(y, x)
     lapack.trtrs(QA, x, uplo = 'U', trans = 'T', n = p)
     
     // w := w - Gs1 * x[:p] 
     //    = W^{-T}*bz - Gs1*by 
     blas.gemv(Gs, x, w, alpha = -1.0, beta = 1.0, n = p, m = 
     cdim_pckd)
     
     // u := [ Q3'*w + v[p:];  0 ]
     //    = [ Q3'*w + R3^{-T}*Q2'*bx; 0 ]
     blas.copy(w, u)
     lapack.ormqr(Gs, tauG, u, trans = 'T', k = n-p, offsetA = 
     Gs.size[0]*p, m = cdim_pckd)
     blas.axpy(vv, u, offsetx = p, n = n-p)
     blas.scal(0.0, u, offset = n-p)
     
     // x[p:] := R3^{-1} * u[:n-p]  
     blas.copy(u, x, offsety = p, n = n-p)
     lapack.trtrs(Gs, x, uplo='U', n = n-p, offsetA = Gs.size[0]*p,
     offsetB = p)
     
     // x is now [ R1^{-T}*by;  R3^{-1}*u[:n-p] ]
     // x := [Q1 Q2]*x
     lapack.ormqr(QA, tauA, x) 
     
     // u := [Q3, Q4] * u - w 
     //    = Q3 * u[:n-p] - w
     lapack.ormqr(Gs, tauG, u, k = n-p, m = cdim_pckd, offsetA = 
     Gs.size[0]*p)
     blas.axpy(w, u, alpha = -1.0)  
     
     // y := R1^{-1} * ( v[:p] - Gs1'*u )
     //    = R1^{-1} * ( Q1'*bx - Gs1'*u )
     blas.copy(vv, y, n = p)
     blas.gemv(Gs, u, y, m = cdim_pckd, n = p, trans = 'T', alpha = 
     -1.0, beta = 1.0)
     lapack.trtrs(QA, y, uplo = 'U', n=p) 
     
     unpack(u, z, dims)
     
     return solve
     
     return factor
     
     }
     */

#define IMPL_CONEQP(Scalar) template<> \
void coneqp<Scalar>(const QPMatrix<Scalar> &P, KQP_VECTOR(Scalar) &q,\
    ConeQPReturn<Scalar> &result,\
    bool initVals,\
    Dimensions dims,\
    const QPMatrix<Scalar> *_G, KQP_VECTOR(Scalar)* _h, \
    KQP_MATRIX(Scalar) *_A, KQP_VECTOR(Scalar) *_b,\
    KKTPreSolver<Scalar>* kktpresolver, \
    ConeQPOptions<Scalar> options)    
    
    IMPL_CONEQP(double);
    IMPL_CONEQP(float);
} // ns
} // ns

