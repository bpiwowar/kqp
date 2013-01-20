#include <kqp/coneprog.hpp>

#include <kqp/cleaning/qp_approach.hpp>

#include <Eigen/Cholesky> 


DEFINE_LOGGER(logger, "kqp.qp-approach");

namespace kqp {
    
    template<typename Derived>
    bool isnan(const Eigen::MatrixBase<Derived>& x)
    {
      return !(x.array() == x.array()).all();
    }
    
    // --- QP solver
    
    template<typename Scalar>
    class KQP_KKTSolver : public cvxopt::KKTSolver<Scalar> {
        const Eigen::LLT<KQP_MATRIX(Scalar)> &cholK;
        const KQP_MATRIX(Scalar) &B;
        const KQP_MATRIX(Scalar) &BBT;
        const KQP_VECTOR(Scalar) &nu;

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
        KQP_KKTSolver(const Eigen::LLT<KQP_MATRIX(Scalar)> &cholK, const KQP_MATRIX(Scalar) &B, const KQP_MATRIX(Scalar) &BBT, const cvxopt::ScalingMatrix<Scalar> &w, const KQP_VECTOR(Scalar) &nu) 
        : cholK(cholK), B(B), BBT(BBT), nu(nu), n(B.cols()), r(w.d.size() / n / 2) , Wd(w.d)
        {
            // The scaling matrix has dimension r * n * 2
            KQP_LOG_DEBUG(logger, "Preparing the KKT solver of dimension r=" << convert(r) << " and n=" << convert(n));
            
            // Gets U and V
            KQP_VECTOR(Scalar) U = w.d.topRows(r*n);
            U.array() = U.array() * U.array();
            
            KQP_VECTOR(Scalar) V = w.d.bottomRows(r*n);
            V.array() = V.array() * V.array();
            
            // Computes L22[i]
            L22.resize(r);
            for(int i = 0; i < r; i++) {
                KQP_MATRIX(Scalar) mX = nu[i] * nu[i] * BBT;
                mX.diagonal() += U.segment(i*n, n);
                L22[i].compute(mX);
                if (L22[i].info() != Eigen::ComputationInfo::Success) 
                    KQP_THROW_EXCEPTION_F(arithmetic_exception, "Error [%d] in computing L22[%d]", %L22[i].info() %i);                
            }


            // Computes L32[i]
            L32.resize(r);
            for(int i = 0; i < r; i++) {
                //  Solves L32 . L22' = - B . B^\top
                L32[i] = -nu[i] * nu[i] * BBT;
                L22[i].matrixU().template solveInPlace<Eigen::OnTheRight>(L32[i]);
                if (isnan(L32[i])) KQP_THROW_EXCEPTION_F(arithmetic_exception, "NaN in L32[%d]", %i);
            }
            
            // Computes L33
            L33.resize(r);
            for(int i = 0; i < r; i++) {
                // Cholesky decomposition of V + BBT - L32 . L32.T
                KQP_MATRIX(Scalar) mX = nu[i] * nu[i] * BBT - L32[i] * L32[i].adjoint();
                mX.diagonal() += V.segment(i*n, n);
                L33[i].compute(mX);
                if (L33[i].info() != Eigen::ComputationInfo::Success) 
                    KQP_THROW_EXCEPTION_F(arithmetic_exception, "Error [%d] in computing L33[%d]", %L33[i].info() %i);                
            }
            
            // Computes L42
            L42.resize(r);
            for(int i = 0; i < r; i++) {
                // Solves L42 L22' = Id
                L42[i].setIdentity(n,n);
                L22[i].matrixU().template solveInPlace<Eigen::OnTheRight>(L42[i]);
                if (isnan(L42[i])) KQP_THROW_EXCEPTION_F(arithmetic_exception, "NaN in L42[%d]", %i);                
            }
            
            
            // Computes L43
            L43.resize(r);
            for(int i = 0; i < r; i++) {
                // Solves L43 L33' = Id - L42 L32'
                L43[i].noalias() = - L42[i] * L32[i].adjoint();
                L43[i].diagonal().array() += 1;
                L33[i].matrixU().template solveInPlace<Eigen::OnTheRight>(L43[i]);
                if (isnan(L43[i])) KQP_THROW_EXCEPTION_F(arithmetic_exception, "NaN in L43[%d]", %i);                
            }
            
            // Computes L44: Cholesky of 
            KQP_MATRIX(Scalar) _L44(n,n);
            _L44.setConstant(0.);
            for(int i = 0; i < r; i++) {
                _L44 += L43[i] * L43[i].adjoint() + L42[i] * L42[i].adjoint();
            }
            L44.compute(_L44);
            if (L44.info() != Eigen::ComputationInfo::Success) 
                KQP_THROW_EXCEPTION_F(arithmetic_exception, "Error [%d] in computing L44", %L44.info());                
            
        }
        
        //! Reconstruct for debug purposes
        void reconstruct(KQP_MATRIX(Scalar) &L) const {
            L.resize(n*(r+1) + 2*n*r, n*(r+1) + 2*n*r);
            
            for(int i = 0; i < r; i++) {
                L.block(i*n, i*n, n, n) = cholK.matrixL();
                L.block((i+r)*n, (i+r)*n, n,n) = L22[i].matrixL();
                L.block((i+r)*n, i*n, n,n) = - nu[i] * B;
                L.block((i+2*r)*n, i*n, n,n) = nu[i] * B;
                
                L.block((i+2*r)*n, (i+r)*n, n,n) = L32[i];
                L.block((i+2*r)*n, (i+2*r)*n, n,n) = L33[i].matrixL();
                
                L.block((3*r)*n, (i+r)*n, n,n) = L42[i];
                L.block((3*r)*n, (i+2*r)*n, n,n) = L43[i];
            }
            L.block((3*r)*n, (3*r)*n, n,n) = L44.matrixL();
            
            
            Eigen::DiagonalMatrix<double, Dynamic> D(L.rows());
            D.diagonal().setConstant(1.);
            D.diagonal().segment(r*n,2*r*n).setConstant(-1);
            
            L = (L * D * L.adjoint()).eval();
        }
        
        
        virtual void solve(KQP_VECTOR(Scalar) &x, KQP_VECTOR(Scalar) &/*y*/, KQP_VECTOR(Scalar) & z) const {
            if (isnan(x)) KQP_THROW_EXCEPTION(arithmetic_exception, "NaN in x");
            if (isnan(z)) KQP_THROW_EXCEPTION(arithmetic_exception, "NaN in z");

            // Prepares the access to sub-parts
            Eigen::VectorBlock<KQP_VECTOR(Scalar)> b = x.segment(r*n,n);
            
            // First phase
            for(int i = 0; i < r; i++) {
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> ai = x.segment(i*n, n);
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> ci = z.segment(i*n, n);
                Eigen::VectorBlock<KQP_VECTOR(Scalar)> di = z.segment((i+r)*n, n);
                
                cholK.matrixL().solveInPlace(ai);
                
                // Solves L22 x = - B * ai - ci
                KQP_MATRIX(Scalar) Bai = nu[i] * B * ai;
                
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
                
                ai += B.adjoint() * nu[i] * (ci - di);
                cholK.matrixU().solveInPlace(ai);
            }
            
            
            // Scale z
            z.array() *= Wd.array();

            // Detect NaN
            
            
            if (isnan(x)) KQP_THROW_EXCEPTION(arithmetic_exception, "NaN in solved x");
            if (isnan(z)) KQP_THROW_EXCEPTION(arithmetic_exception, "NaN in solved z");
        }
        
        
        
    };
    
    //! Builds the pre-solver
    template<typename Scalar>
    KQP_KKTPreSolver<Scalar>::KQP_KKTPreSolver(const KQP_MATRIX(Scalar)& gramMatrix, const KQP_VECTOR(KQP_REAL_OF(Scalar)) &nu) :
    // Compute the Cholesky decomposition of the gram matrix
    nu(nu)
    {       
        Index n =gramMatrix.rows();

        // Computes B in B A' = G0 (L21 and L31)
        // i.e.  computes A B' = G0
        if (boost::is_complex<Scalar>::value) {
            // FIXME: optimise using 2 Cholesky decompositions (see TR)
            KQP_MATRIX(Real) dGram;
            dGram.bottomRightCorner(n,n) = dGram.topLeftCorner(n,n) = gramMatrix.real();
            dGram.bottomLeftCorner(n,n) = dGram.topRightCorner(n,n) = -gramMatrix.imag();
            lltOfK.compute(dGram);
            
            // Complex case, G0 = [Id, Id; Id, -Id]
            B.resize(2*n, 2*n);
            auto Idn = Eigen::Matrix<Real,Dynamic,Dynamic>::Identity(n,n);
            B.topLeftCorner(n,n) = Idn;
            B.topRightCorner(n,n) = Idn;
            B.bottomLeftCorner(n,n) = Idn;
            B.bottomRightCorner(n,n) = -Idn;            
        } else {
            lltOfK.compute(gramMatrix.real());
            B.setIdentity(gramMatrix.rows(), gramMatrix.cols());
        }
        
        lltOfK.matrixU().template solveInPlace<Eigen::OnTheRight>(B);
        
        // Computing B * B.T
        BBT.noalias() = B * B.adjoint();
    }
    
    template<typename Scalar>
    cvxopt::KKTSolver<typename KQP_KKTPreSolver<Scalar>::Real> *KQP_KKTPreSolver<Scalar>::get(const cvxopt::ScalingMatrix<typename KQP_KKTPreSolver<Scalar>::Real> &w) {
        KQP_LOG_DEBUG(logger, "Creating a new KKT solver");
        return new KQP_KKTSolver<Real>(lltOfK, B, BBT, w, nu);
    }
    
    
    /**
     Class that knows how to multiply by
     
     \f$ \left(\begin{array}{ccc}
     K'\\
     & \ddots\\
     &  & K'
     \end{array}\right)
     \f$
     
     where \f$K'\f$ is \f$K\f$ (real case) or \f$ 
     \left(\begin{array}{cc}
        Re(K)  & -Im(K) \\
        -Im(K) & Re(K)
     \end{array}\right)
     \f$
     
     @warning Assumes \f$K\f$ is Hermitian
     
     */
    template<typename Scalar>
    class KMult : public cvxopt::QPMatrix<KQP_REAL_OF(Scalar)> {
        Index n, r;
        const KQP_MATRIX(Scalar) &g;
    public:
        typedef KQP_REAL_OF(Scalar) Real;
        
        KMult(Index n, Index r, const KQP_MATRIX(Scalar) &g) : n(n), r(r), g(g) {}
        
        virtual void mult(const KQP_VECTOR(Real) &x, KQP_VECTOR(Real) &y, bool /*trans*/ = false) const {
            y.resize(x.rows());
            y.tail(n).setZero();
            
            if (boost::is_complex<Scalar>::value) {
                for(int i = 0; i < r; i++) {
                    const int i_re = 2 * i, i_im = 2 * i + 1;
                    y.segment(i_re*n, n).noalias() = g.real() * x.segment(i_re*n,n) - x.segment(i_im*n,n);
                    y.segment(i_im*n, n).noalias() = - g.imag() * x.segment(i_re*n,n) + g.real() * x.segment(i_im*n,n);
                }
            } else {
                for(int i = 0; i < r; i++) 
                    y.segment(i*n, n).noalias() = g.real() * x.segment(i*n,n);
            }
        }
        virtual Index rows() const { return g.rows(); }
        virtual Index cols() const { return g.cols(); }
        
    };
    
    /**
     * Multiplying with matrix G 
     *
     * Real case:
     * <pre>
     *  -G1          -Id
     *     ...       -Id
     *          -Gr  -Id
     *  G1           -Id
     *     ...       -Id
     *           Gr  -Id
     * </pre>
     * 
     * where Gr is \f$ nu_i * Id_n \f$ (real case) or
     * \f$ nu_i * [ Id_n, Id_n; Id_n -Id_n] \f$
     * of size 2nr x n (r+1)
     */
    template<typename Scalar>
    class QPConstraints : public cvxopt::QPMatrix<KQP_REAL_OF(Scalar)> {
    public:
        typedef KQP_REAL_OF(Scalar) Real;
    private:
        Index n, r;
        const KQP_VECTOR(Real) &nu;

    public:
        QPConstraints(Index n, Index r, const KQP_VECTOR(Real) &nu) : n(n), r(r), nu(nu) {}
        
        //! Get segments of size n (i.e. number of pre-images)
        inline auto get(const KQP_VECTOR(Real) &x, Index i) const -> decltype((x.segment(i*n, n))) { return x.segment(i*n,n); }
        inline auto get(KQP_VECTOR(Real) &x, Index i) const -> decltype((x.segment(i*n, n))) { return x.segment(i*n,n); }

        virtual void mult(const KQP_VECTOR(Real) &x, KQP_VECTOR(Real) &y, bool trans = false) const {
            // assumes y and x are two different things
            assert(&x != &y);
            
            if (boost::is_complex<Scalar>::value) {
                // Complex case
                if (!trans) {
                    // y = G x
                    y.resize(2*n*2*r);
                    
                    for(Index i = 0; i < 2 * r; i++) {
                        // Index of the "real part"
                        int di = i % 2 == 0 ? 0 : -1;
                        Index ip = i + di;

                        get(y, i+2*r) = nu[i] * (get(x,ip) + (nu[i] * (Real)di) * get(x,ip+1));
                        get(y, i) = - get(y, i+2*r) - get(x, 2*r);
                        get(y, i+2*r) -=  get(x, 2*r);
                    }
                } else {
                    // y = G' x
                    y.resize(n * (2*r+1));
                    get(y, 2*r).array().setConstant(0);
                    
                    for(int i = 0; i < 2*r; i++) {
                        // Index of the "real part"
                        int di = i % 2 == 0 ? 0 : -1;
                        Index ip = i + di;

                        get(y,i) = nu[i] * (- get(x,ip)  - di * get(x,ip+1) + get(x,ip+r) + di * get(x,ip+1+r) );
                        get(y,i) -= get(x,i) + get(x,i+r);
                    }
                    
                }
            } else {
                // Real case
                if (!trans) {
                    // y = G x
                    y.resize(2*n*r);
                    
                    for(int i = 0; i < r; i++) {
                        y.segment(i*n, n) = - nu[i] * x.segment(i*n, n) - x.segment(n*r, n);
                        y.segment((i+r)*n, n) = nu[i] * x.segment(i*n, n) - x.segment(n*r, n);
                    }
                } else {
                    // y = G' x
                    y.resize(n * (r+1));
                    y.segment(n * r, n).array().setConstant(0);
                    
                    for(int i = 0; i < r; i++) {
                        y.segment(i*n, n) =  nu[i] * (- x.segment(i*n, n) + x.segment((i+r)*n, n));
                        y.segment(n * r, n) -= x.segment(i*n, n) +  x.segment((i+r)*n, n);
                    }
                    
                }
            }
            
        }
        virtual Index rows() const { return 2*n*r; }
        virtual Index cols() const { return n * (r+1); }
        
    };
    
    
    template<typename Scalar>
    void solve_qp(int r, 
                  KQP_REAL_OF(Scalar) lambda, 
                  const KQP_MATRIX(Scalar) &gramMatrix, 
                  const KQP_MATRIX(Scalar) &alpha, 
                  const KQP_VECTOR(KQP_REAL_OF(Scalar)) &nu,
                  kqp::cvxopt::ConeQPReturn<KQP_REAL_OF(Scalar)> &result,
                  const cvxopt::ConeQPOptions<KQP_REAL_OF(Scalar)>& options) {
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        const bool isComplex = boost::is_complex<Scalar>::value; 
        
        KQP_LOG_DEBUG(logger, "Gram matrix:\n" << gramMatrix);
        KQP_LOG_DEBUG(logger,  "Alpha:\n" << alpha);
        KQP_LOG_DEBUG(logger,  "nu:\n" << nu);
        
        Index rp = isComplex ? 2 * r : r;
        
        Index n = gramMatrix.rows();
        KQP_VECTOR(Real) c(n*rp + n);
        
        for(int i = 0; i < rp; i++) 
            if (isComplex) {
                Index j = i / 2;
                if (i % 2 == 0) 
                    c.segment(i*n, n) = - (gramMatrix.real() * alpha.col(j).real()  + gramMatrix.imag() * alpha.col(j).imag());
                else 
                    c.segment(i*n, n) = - (- gramMatrix.imag() * alpha.col(j).real()  + gramMatrix.real() * alpha.col(j).imag());
            } else
                c.segment(i*n, n) = - gramMatrix.real() * alpha.real().block(0, i, n, 1);
        
        c.segment(rp*n,n).setConstant(lambda / Real(2));
        
        QPConstraints<Scalar> G(n, r, nu);
        KQP_KKTPreSolver<Scalar> kkt_presolver(gramMatrix, nu);
        
        KQP_LOG_DEBUG(logger,  "c:\n" << c.adjoint());

        cvxopt::coneqp<Real>(KMult<Scalar>(n,r, gramMatrix), c, result, 
                               false /* No initial value */, 
                               cvxopt::Dimensions(),
                               &G, NULL, NULL, NULL,
                               &kkt_presolver,
                               options
                               );
        
    }    
    
    
    
# define KQP_SCALAR_GEN(scalar) KQP_CLEANING__QP_APPROACH_H_GEN(, scalar) \
 template void solve_qp<scalar>(int r, KQP_REAL_OF(scalar) lambda, const KQP_MATRIX(scalar) &gramMatrix, \
    const KQP_MATRIX(scalar) &alpha, const KQP_VECTOR(KQP_REAL_OF(scalar)) &nu, kqp::cvxopt::ConeQPReturn<KQP_REAL_OF(scalar)> &result,\
    const cvxopt::ConeQPOptions<KQP_REAL_OF(scalar)>& options);

# include <kqp/for_all_scalar_gen.h.inc>
   
}