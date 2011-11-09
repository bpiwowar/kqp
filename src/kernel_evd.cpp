#include <Eigen/Cholesky> 

#include "coneprog.h"

#include "kernel_evd.h"

using namespace kqp;

double EPSILON = 1e-17;


// --- QP solver

class KQP_KKTSolver : public cvxopt::KKTSolver {
    Eigen::LLT<Eigen::MatrixXd> &cholK;
    Eigen::MatrixXd &B;
    Eigen::MatrixXd &BBT;
    // Number of pre-images
    int n;
    // Rank (number of basis vectors)
    int r;
    
    std::vector<Eigen::MatrixXd> L22;
public:
    KQP_KKTSolver(Eigen::LLT<Eigen::MatrixXd> &cholK, Eigen::MatrixXd &B, Eigen::MatrixXd &BBT, const cvxopt::ScalingMatrix &w) 
        : cholK(cholK), B(B), BBT(BBT), n(B.cols()), r(w.d.diagonal().size() / n / 2) 
    {
        // The scaling matrix has dimension r * n * 2
        
        
        // Gets U and V
        Eigen::VectorXd U = w.d.diagonal().topRows(r*n);
        U.array() = U.array() * U.array();
        
        Eigen::VectorXd V = w.d.diagonal().bottomRows(r*n);
        V.array() = V.array() * V.array();
        
        // Computes L22[i]
        L22.resize(r);
        for(int i = 0; i < r; i++) {
            L22[i] = BBT;
            L22[i].diagonal() += U;
//            L22[i].llt
        }
        
        //    
        // Computing L22
        //    L22 = []
        //    for i in range(r):
        //        BBTi = BBT + spmatrix(U[i*n:(i+1)*n] ,range(n),range(n))
        //        cholesky(BBTi)
        //        L22.append(BBTi)
        //
        //    print "# Computing L32"
        //    L32 = []
        //    for i in range(r):
        //        # Solves L32 . L22 = - B . B^\top
        //        C = -BBT
        //        blas.trsm(L22[i], C, side='R', transA='T')
        //        L32.append(C)
        //    
        //    print "# Computing L33"
        //    L33 = []
        //    for i in range(r):
        //        A = spmatrix(U[i*n:(i+1)*n] ,range(n),range(n))  + BBT - L32[i] * L32[i].T
        //        cholesky(A)
        //        L33.append(A)
        //        
        //    print "# Computing L42"
        //    L42 = []
        //    for i in range(r):
        //        A = +L22[i].T
        //        lapack.trtri(A, uplo='U')
        //        L42.append(A)
        //    
        //    print "# Computing L43"
        //    L43 = []
        //    for i in range(r):
        //        A =  id_n - L42[i] * L32[i].T
        //        blas.trsm(L33[i], A, side='R', transA='T')
        //        L43.append(A)
        //
        //    print "# Computing L44 and D4"
        //
        //    
        //    # The indices for the diagonal of a dense matrix
        //    L44 = matrix(0, (n,n))
        //    for i in range(r):
        //        L44 = L44 + L43[i] * L43[i].T + L42[i] * L42[i].T
        //   
        //    cholesky(L44)
        //           
    }
    
    virtual void solve(Eigen::VectorXd &x, Eigen::VectorXd &y, Eigen::VectorXd & z) const {
//        # Maps to our variables x,y,z and t
//        a = []
//        b = x[n*r:n*r + n]
//        c = []
//        d = []
//        for i in range(r):
//            a.append(x[i*n:(i+1)*n])
//            c.append(z[i*n:(i+1)*n])
//            d.append(z[(i+r)*n:(i+r+1)*n])
//
//        if DEBUG:
//            # Now solves using cvxopt
//            xp = +x
//            zp = +z
//            solve(xp,y,zp)
//
//        # First phase
//        for i in range(r):
//            blas.trsm(cholK, a[i])
//
//            Bai = B * a[i]
//            
//            c[i] = - Bai - c[i]
//            blas.trsm(L22[i], c[i])
//
//            d[i] =  Bai - L32[i] * c[i] - d[i]
//            blas.trsm(L33[i], d[i])
//
//            b = b + L42[i] * c[i] + L43[i] * d[i]
//            
//        blas.trsm(L44, b)
//
//        # Second phase
//        blas.trsm(L44, b, transA='T')
//        
//        for i in range(r):
//            d[i] = d[i] - L43[i].T * b
//            blas.trsm(L33[i], d[i], transA='T')
//
//            c[i] = c[i] - L32[i].T * d[i] - L42[i].T * b
//            blas.trsm(L22[i], c[i], transA='T')
//
//            a[i] = a[i] + B.T * (c[i] - d[i])
//            blas.trsm(cholK, a[i], transA='T')
//
//        # Store in vectors and scale
//
//        x[n*r:n*r + n] = b
//        for i in range(r):
//            x[i*n:(i+1)*n] = a[i]
//
//            # Normalise
//            for j in range(n):
//                c[i][j] = c[i][j] / U[i*n+j]
//                d[i][j] = d[i][j] / V[i*n+j]
//                
//            z[i*n:(i+1)*n] = c[i]
//            z[(i+r)*n:(i+r+1)*n] = d[i]
    }
    
    
};

class KQP_KKTPreSolver : public cvxopt::KKTPreSolver {
    Eigen::LLT<Eigen::MatrixXd> lltOfK;
    Eigen::MatrixXd B, BBT;
    
public:
    KQP_KKTPreSolver(const Eigen::MatrixXd& gramMatrix) :
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
    
    KQP_KKTSolver *get(const cvxopt::ScalingMatrix &w) {
        return new KQP_KKTSolver(lltOfK, B, BBT, w);
    };
};


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