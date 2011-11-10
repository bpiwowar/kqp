#include <Eigen/Cholesky> 

#include "coneprog.h"

#include "kernel_evd.h"

DEFINE_LOGGER(logger, "kqp.kernel-evd");

namespace kqp {
    
    // --- QP solver
    
    class KQP_KKTSolver : public cvxopt::KKTSolver {
        Eigen::LLT<Eigen::MatrixXd> &cholK;
        Eigen::MatrixXd &B;
        Eigen::MatrixXd &BBT;
        // Number of pre-images
        int n;
        // Rank (number of basis vectors)
        int r;
        
        // Wdi
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> Wdi;
        
        std::vector<Eigen::LLT<Eigen::MatrixXd> > L22, L33;
        std::vector<Eigen::MatrixXd> L32, L42, L43;
        Eigen::LLT<Eigen::MatrixXd> L44;
    public:
        KQP_KKTSolver(Eigen::LLT<Eigen::MatrixXd> &cholK, Eigen::MatrixXd &B, Eigen::MatrixXd &BBT, const cvxopt::ScalingMatrix &w) 
        : cholK(cholK), B(B), BBT(BBT), n(B.cols()), r(w.d.diagonal().size() / n / 2) , Wdi(w.di)
        {
            // The scaling matrix has dimension r * n * 2
            KQP_LOG_DEBUG(logger, "Preparing the KKT solver of dimension r=" << convert(r) << " and n=" << convert(n));
            KQP_LOG_DEBUG(logger, "BBT is square of dimension " << convert(BBT.rows()));

            
            // Gets U and V
            Eigen::VectorXd U = w.d.diagonal().topRows(r*n);
            U.array() = U.array() * U.array();
            
            Eigen::VectorXd V = w.d.diagonal().bottomRows(r*n);
            V.array() = V.array() * V.array();
            
            // Computes L22[i]
            L22.resize(r);
            for(int i = 0; i < r; i++) {
                Eigen::MatrixXd mX = BBT;
                mX.diagonal() += U.segment(i*n, n);
                L22[i].compute(mX);
            }
            
            // Computes L32[i]
            L32.resize(r);
            for(int i = 0; i < r; i++) {
                //  Solves L32 . L22' = - B . B^\top
                L32[i] = -BBT;
                L22[i].matrixU().solveInPlace<Eigen::OnTheRight>(L32[i]);
            }
            
            // Computes L33
            L33.resize(r);
            for(int i = 0; i < r; i++) {
                // Cholesky decomposition of U + BBT - L32 . L32.T
                Eigen::MatrixXd mX = BBT - L32[i] * L32[i].adjoint();
                mX.diagonal() += V.segment(i*n, n);
                L33[i].compute(mX);
            }
            
            // Computes L42
            L42.resize(r);
            for(int i = 0; i < r; i++) {
                // Solves L42 L22' = Id
                L42[i].setIdentity(n,n);
                L22[i].matrixU().solveInPlace(L42[i]);
            }
            
            
            // Computes L43
            L43.resize(r);
            for(int i = 0; i < r; i++) {
                // Solves L43 L33' = Id - L42 L32'
                L43[i].noalias() = - L42[i] * L32[i].adjoint();
                L43[i].diagonal().array() += 1;
                L33[i].matrixU().solveInPlace<Eigen::OnTheRight>(L43[i]);
            }
            
            // Computes L44: Cholesky of 
            Eigen::MatrixXd _L44(n,n);
            for(int i = 0; i < r; i++) {
                _L44 += L43[i] * L43[i].adjoint() + L42[i] * L42[i].adjoint();
            }
            L44.compute(_L44);
            
        }
        
        
        
        virtual void solve(Eigen::VectorXd &x, Eigen::VectorXd &y, Eigen::VectorXd & z) const {
            // Prepares the access to sub-parts
            
            
            Eigen::VectorBlock<Eigen::VectorXd> b = x.segment(r*n,n);
            
            // First phase
            for(int i = 0; i < r; i++) {
                Eigen::VectorBlock<Eigen::VectorXd> ai = x.segment(i*n, n);
                Eigen::VectorBlock<Eigen::VectorXd> ci = z.segment(i*n, n);
                Eigen::VectorBlock<Eigen::VectorXd> di = z.segment((i+r)*n, n);
                
                cholK.solveInPlace(ai);
                
                Eigen::MatrixXd Bai = B * ai;
                
                ci -= Bai;
                L22[i].solveInPlace(ci);
                
                di = Bai - L32[i] * ci - di;
                L33[i].solveInPlace(di);
                
                b += L42[i] * ci + L43[i] * di;
            } 
            L44.solveInPlace(b);
            
            //  Second phase
            L44.matrixU().solveInPlace(b);
            for(int i = 0; i < r; i++) {           
                Eigen::VectorBlock<Eigen::VectorXd> ai = x.segment(i*n, n);
                Eigen::VectorBlock<Eigen::VectorXd> ci = z.segment(i*n, n);
                Eigen::VectorBlock<Eigen::VectorXd> di = z.segment((i+r)*n, n);
                
                di -= L43[i].adjoint() * b;
                L33[i].matrixU().solveInPlace(di);
                
                
                ci -= L32[i].adjoint() * di - L42[i] * b;
                L22[i].matrixU().solveInPlace(ci);
                
                ai += B.adjoint() * (ci - di);
                cholK.matrixU().solveInPlace(ai);
            }
            
            
            // Scale z
            z.array() *= Wdi.diagonal().array();
        }
        
        
    };
    
    KQP_KKTPreSolver::KQP_KKTPreSolver(const Eigen::MatrixXd& gramMatrix) :
    // Compute the Cholesky decomposition of the gram matrix
    lltOfK(Eigen::LLT<Eigen::MatrixXd>(gramMatrix))
    {
        
        // Computes B in B A' = Id (L21 and L31)
        // i.e.  computes A B' = Id
        B.setIdentity(gramMatrix.rows(), gramMatrix.cols());
        lltOfK.solveInPlace(B);
        B.adjointInPlace();
        
        // Computing B * B.T
        BBT = B * B.adjoint();
        
    }
    
    cvxopt::KKTSolver *KQP_KKTPreSolver::get(const cvxopt::ScalingMatrix &w) {
        KQP_LOG_DEBUG(logger, "Creating a new KKT solver");
        return new KQP_KKTSolver(lltOfK, B, BBT, w);
    }
    
    
    // ---
    
    template <typename scalar, class F>
    typename FeatureMatrix<scalar,F>::Matrix FeatureMatrix<scalar, F>::computeInnerProducts(const FeatureMatrix<scalar, F>& other) const {
        
        Eigen::MatrixXd m = Eigen::MatrixXd(size(), other.size());
        
        for (int i = 0; i < size(); i++)
            for (int j = i; j < other.size(); j++)
                m(j,i) = m(i, j) = computeInnerProduct(i, get(j));
        
        return m;
    }
    
    
    template <typename scalar, class F>
    typename FeatureMatrix<scalar,F>::Vector FeatureMatrix<scalar, F>::computeInnerProducts(const F& vector) const {
        typename FeatureMatrix::Vector innerProducts(size());
        for (int i = size(); --i >= 0;)
            innerProducts[i] = computeInnerProduct(i, vector);
        return innerProducts;
    }
    
    
    template <typename scalar, class F>
    typename FeatureMatrix<scalar,F>::Matrix  FeatureMatrix<scalar, F>::computeGramMatrix() const {
        // We loose space here, could be used otherwise???
        Eigen::MatrixXd m = Eigen::MatrixXd(size(), size()).selfadjointView<Eigen::Upper>();
        
        for (int i = size(); --i >= 0;)
            for (int j = i + 1; --j >= 0;) {
                double x = computeInnerProduct(i, get(j));
                m(i,j) = m(j,i) = x;
            }
        return m;
    }
    
    // --- Scalar matrix
    
    template <typename scalar> ScalarMatrix<scalar>::~ScalarMatrix() {}
    
    
    // --- Direct builder
    
    template <typename scalar, class F> void DirectBuilder<scalar,F>::add(double alpha, const F &v) {
        this->list.add(v);
    }
    
}