#include <Eigen/Cholesky> 

#include "coneprog.hpp"

#include "null_space.hpp"

DEFINE_LOGGER(logger, "kqp.kernel-evd");

namespace kqp {
    
    // --- QP solver
    
    template<typename Scalar>
    class KQP_KKTSolver : public cvxopt::KKTSolver<Scalar> {
        const Eigen::LLT<KQP_MATRIX(Scalar)> &cholK;
        const KQP_MATRIX(Scalar) &B;
        const KQP_MATRIX(Scalar) &BBT;
        // Number of pre-images
        int n;
        // Rank (number of basis vectors)
        int r;
        
        // Wdi
        KQP_VECTOR(Scalar) Wd;
        
        std::vector<Eigen::LLT<KQP_MATRIX(Scalar)> > L22, L33;
        std::vector<KQP_MATRIX(Scalar)> L32, L42, L43;
        Eigen::LLT<KQP_MATRIX(Scalar)> L44;
    public:
        KQP_KKTSolver(const Eigen::LLT<KQP_MATRIX(Scalar)> &cholK, const KQP_MATRIX(Scalar) &B, const KQP_MATRIX(Scalar) &BBT, const cvxopt::ScalingMatrix<Scalar> &w) 
        : cholK(cholK), B(B), BBT(BBT), n(B.cols()), r(w.d.diagonal().size() / n / 2) , Wd(w.d)
        {
            // The scaling matrix has dimension r * n * 2
            KQP_LOG_DEBUG(logger, "Preparing the KKT solver of dimension r=" << convert(r) << " and n=" << convert(n));
            
            // Gets U and V
            KQP_VECTOR(Scalar) U = w.d.diagonal().topRows(r*n);
            U.array() = U.array() * U.array();
            
            KQP_VECTOR(Scalar) V = w.d.diagonal().bottomRows(r*n);
            V.array() = V.array() * V.array();
            
            // Computes L22[i]
            L22.resize(r);
            for(int i = 0; i < r; i++) {
                KQP_MATRIX(Scalar) mX = BBT;
                mX.diagonal() += U.segment(i*n, n);
                L22[i].compute(mX);
            }
            
            // Computes L32[i]
            L32.resize(r);
            for(int i = 0; i < r; i++) {
                //  Solves L32 . L22' = - B . B^\top
                L32[i] = -BBT;
                L22[i].matrixU().template solveInPlace<Eigen::OnTheRight>(L32[i]);
            }
            
            // Computes L33
            L33.resize(r);
            for(int i = 0; i < r; i++) {
                // Cholesky decomposition of V + BBT - L32 . L32.T
                KQP_MATRIX(Scalar) mX = BBT - L32[i] * L32[i].adjoint();
                mX.diagonal() += V.segment(i*n, n);
                L33[i].compute(mX);
                
            }
            
            // Computes L42
            L42.resize(r);
            for(int i = 0; i < r; i++) {
                // Solves L42 L22' = Id
                L42[i].setIdentity(n,n);
                L22[i].matrixU().template solveInPlace<Eigen::OnTheRight>(L42[i]);
            }
            
            
            // Computes L43
            L43.resize(r);
            for(int i = 0; i < r; i++) {
                // Solves L43 L33' = Id - L42 L32'
                L43[i].noalias() = - L42[i] * L32[i].adjoint();
                L43[i].diagonal().array() += 1;
                L33[i].matrixU().template solveInPlace<Eigen::OnTheRight>(L43[i]);
            }
            
            // Computes L44: Cholesky of 
            KQP_MATRIX(Scalar) _L44(n,n);
            _L44.setConstant(0.);
            for(int i = 0; i < r; i++) {
                _L44 += L43[i] * L43[i].adjoint() + L42[i] * L42[i].adjoint();
            }
            L44.compute(_L44);
            
        }
        
        void reconstruct(KQP_MATRIX(Scalar) &L) const {
            L.resize(n*(r+1) + 2*n*r, n*(r+1) + 2*n*r);
            
            for(int i = 0; i < r; i++) {
                L.block(i*n, i*n, n, n) = cholK.matrixL();
                L.block((i+r)*n, (i+r)*n, n,n) = L22[i].matrixL();
                L.block((i+r)*n, i*n, n,n) = -B;
                L.block((i+2*r)*n, i*n, n,n) = B;
                
                L.block((i+2*r)*n, (i+r)*n, n,n) = L32[i];
                L.block((i+2*r)*n, (i+2*r)*n, n,n) = L33[i].matrixL();
                
                L.block((3*r)*n, (i+r)*n, n,n) = L42[i];
                L.block((3*r)*n, (i+2*r)*n, n,n) = L43[i];
            }
            L.block((3*r)*n, (3*r)*n, n,n) = L44.matrixL();
            
            
            Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(L.rows());
            D.diagonal().setConstant(1.);
            D.diagonal().segment(r*n,2*r*n).setConstant(-1);
            
            L = (L * D * L.adjoint()).eval();
        }
        
        
        virtual void solve(KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &y, KQP_VECTOR(Scalar) & z) const {
            // Prepares the access to sub-parts
            Eigen::VectorBlock<KQP_VECTOR(Scalar)> b = x.segment(r*n,n);
            
            // First phase
            for(int i = 0; i < r; i++) {
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> ai = x.segment(i*n, n);
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> ci = z.segment(i*n, n);
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> di = z.segment((i+r)*n, n);
                
                cholK.matrixL().solveInPlace(ai);
                
                // Solves L22 x = - B * ai - ci
                KQP_MATRIX(Scalar) Bai = B * ai;
                
                ci *= -1.;
                ci -= Bai;
                L22[i].matrixL().solveInPlace(ci);
                
                di *= -1.;
                di += Bai - L32[i] * ci;
                L33[i].matrixL().solveInPlace(di);
                
                b += L42[i] * ci + L43[i] * di;
            } 
            L44.matrixL().solveInPlace(b);
            
            //  Second phase
            L44.matrixU().solveInPlace(b);
            for(int i = 0; i < r; i++) {           
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> ai = x.segment(i*n, n);
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> ci = z.segment(i*n, n);
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> di = z.segment((i+r)*n, n);
                
                di -= L43[i].adjoint() * b;
                L33[i].matrixU().solveInPlace(di);
                
                
                ci -= L32[i].adjoint() * di + L42[i].adjoint() * b;
                L22[i].matrixU().solveInPlace(ci);
                
                ai += B.adjoint() * (ci - di);
                cholK.matrixU().solveInPlace(ai);
            }
            
            
            // Scale z
            z.array() *= Wd.diagonal().array();
        }
        
        
    };
    
    template<typename Scalar>
    KQP_KKTPreSolver<Scalar>::KQP_KKTPreSolver(const KQP_MATRIX(Scalar)& gramMatrix) :
    // Compute the Cholesky decomposition of the gram matrix
    lltOfK(Eigen::LLT<KQP_MATRIX(Scalar)>(gramMatrix))
    {       
        // Computes B in B A' = Id (L21 and L31)
        // i.e.  computes A B' = Id
        B.setIdentity(gramMatrix.rows(), gramMatrix.cols());
        lltOfK.matrixU().template solveInPlace<Eigen::OnTheRight>(B);
        
        // Computing B * B.T
        BBT.noalias() = B * B.adjoint();
    }
    
    template<typename Scalar>
    cvxopt::KKTSolver<Scalar> *KQP_KKTPreSolver<Scalar>::get(const cvxopt::ScalingMatrix<Scalar> &w) {
        KQP_LOG_DEBUG(logger, "Creating a new KKT solver");
        return new KQP_KKTSolver<Scalar>(lltOfK, B, BBT, w);
    }
    
    
    /**
     Class that knows how to multiply by
     
     \f$ \left(\begin{array}{ccc}
     K\\
     & \ddots\\
     &  & K
     \end{array}\right)
     \f$
     
     */
    template<typename Scalar>
    class KMult : public cvxopt::QPMatrix<Scalar> {
        Index n, r;
        const KQP_MATRIX(Scalar) &g;
    public:
        KMult(Index n, Index r, const KQP_MATRIX(Scalar) &g) : n(n), r(r), g(g) {}
        
        virtual void mult(const KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &y, bool trans = false) const {
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
    template<typename Scalar>
    class QPConstraints : public cvxopt::QPMatrix<Scalar> {
        Index n, r;
    public:
        QPConstraints(Index n, Index r) : n(n), r(r) {}
        
        virtual void mult(const KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &y, bool trans = false) const {
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
    template<typename Scalar>
    void solve_qp(int r, Scalar lambda, const KQP_MATRIX(Scalar) &gramMatrix, const KQP_MATRIX(Scalar) &alpha, kqp::cvxopt::ConeQPReturn<Scalar> &result) {
        Index n = gramMatrix.rows();
        KQP_VECTOR(Scalar) c(n*r + n);
        for(int i = 0; i < r; i++) {
            c.segment(i*n, n) = - 2 * gramMatrix * alpha.block(0, i, n, 1);
        }
        c.segment(r*n,n).setConstant(lambda);
        
        QPConstraints<Scalar> G(n, r);
        KQP_KKTPreSolver<Scalar> kkt_presolver(gramMatrix);
        cvxopt::ConeQPOptions<Scalar> options;
        
        cvxopt::coneqp<Scalar>(KMult<Scalar>(n,r, gramMatrix), c, result, 
                       false, cvxopt::Dimensions(),
                       &G, NULL, NULL, NULL,
                       &kkt_presolver,
                       options
                       );
        
    }
    
#define INSTANCE(Scalar) \
    template void solve_qp<Scalar>(int r, Scalar lambda, const KQP_MATRIX(Scalar) &gramMatrix, const KQP_MATRIX(Scalar) &alpha, kqp::cvxopt::ConeQPReturn<Scalar> &result);
    
    INSTANCE(float);
    INSTANCE(double);
    
}