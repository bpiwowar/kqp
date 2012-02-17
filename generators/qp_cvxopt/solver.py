#
# The KKT solver
#

from cvxopt import lapack, matrix, spmatrix, blas, mul

DEBUG = 0
# Clean-up the upper triangular
def makeLT(M):
    N = M.size
    for i in range(N[0]):
        for j in range(i+1, N[1]):
            M[i,j] = 0

def cholesky(A):
    """ Cholesky with clean-up """
    lapack.potrf(A)
    makeLT(A)


def solver(n,r,g):
    global DEBUG
    id_n = spmatrix(1., range(n), range(n))
    
    # Copy A and compute the Cholesky decomposition
    if DEBUG > 0: print "# Computing the cholesky decomposition of P"
    cholK = +g
    cholesky(cholK)

    if DEBUG > 0: print "# Computing the B in B A.T = Id (L21 and L31)"
    B = matrix(id_n, (n,n))
    blas.trsm(cholK, B, side='R', transA='T')

    if DEBUG > 0: print "# Computing B B^T"
    BBT = B * B.T
            
    def F(W):
        """
        Returns a function f(x, y, z) that solves the KKT conditions

        """
        U = +W['d'][0:n*r]
        U = U ** 2
        V = +W['d'][n*r:2*n*r] 
        V = V ** 2
        Wd = +W['d']

        if DEBUG > 0: print "# Computing L22"
        L22 = []
        for i in range(r):
            BBTi = BBT + spmatrix(U[i*n:(i+1)*n] ,range(n),range(n))
            cholesky(BBTi)
            L22.append(BBTi)

        if DEBUG > 0: print "# Computing L32"
        L32 = []
        for i in range(r):
            # Solves L32 . L22 = - B . B^\top
            C = -BBT
            blas.trsm(L22[i], C, side='R', transA='T')
            L32.append(C)

        if DEBUG > 0: print "# Computing L33"
        L33 = []
        for i in range(r):
            A = spmatrix(V[i*n:(i+1)*n] ,range(n),range(n))  + BBT - L32[i] * L32[i].T
            cholesky(A)
            L33.append(A)

        if DEBUG > 0: print "# Computing L42"
        L42 = []
        for i in range(r):
            A = +L22[i].T
            lapack.trtri(A, uplo='U')
            L42.append(A)

        if DEBUG > 0: print "# Computing L43"
        L43 = []
        for i in range(r):
            A =  id_n - L42[i] * L32[i].T
            blas.trsm(L33[i], A, side='R', transA='T')
            L43.append(A)

        if DEBUG > 0: print "# Computing L44 and D4"


        # The indices for the diagonal of a dense matrix
        L44 = matrix(0, (n,n))
        for i in range(r):
            L44 = L44 + L43[i] * L43[i].T + L42[i] * L42[i].T

        cholesky(L44)

        # WARNING: y, z and t have been permuted (LD33 and LD44piv)

        if DEBUG > 0: print "## PRE-COMPUTATION DONE ##"


        # Checking the decomposition
        if DEBUG > 1:
            if DEBUG > 2:
                mA = []
                mB = []
                mD = []
                for i in range(3*r+1):
                    m = []
                    for j in range(3*r+1): m.append(zero_n)
                    mA.append(m)


                for i in range(r):
                    mA[i][i] = cholK
                    mA[i+r][i+r] = L22[i]
                    mA[i][i+r] = -B
                    mA[i][i+2*r] = B

                    mA[i+r][i+2*r] = L32[i]
                    mA[i+2*r][i+2*r] = L33[i]

                    mA[i+r][3*r] = L42[i]
                    mA[i+2*r][3*r] = L43[i]

                    mD.append(id_n)

                mA[3*r][3*r] = L44
                for i in range(2*r): mD.append(-id_n)
                mD.append(id_n)

                printing.options['width'] = 30
                mA = sparse(mA)
                mD = spdiag(mD)

                print "LL^T =\n%s" % (mA * mD * mA.T),
                print "P =\n%s" % P,

                print g
                print "W = %s" % W["d"].T,
                print "U^2 = %s" % U.T,
                print "V^2 = %s" % V.T,

            print "### Pre-compute for check"
            solve = chol2(W,P)


        def f(x, y, z):
            """
            On entry bx, bz are stored in x, z.  On exit x, z contain the solution,
            with z scaled: z./di is returned instead of z.
            """

            # Maps to our variables x,y,z and t
            if DEBUG > 0:
                print "... Computing ..."
                print "bx = %sbz = %s" % (x.T, z.T),
            a = []
            b = x[n*r:n*r + n]
            c = []
            d = []
            for i in range(r):
                a.append(x[i*n:(i+1)*n])
                c.append(z[i*n:(i+1)*n])
                d.append(z[(i+r)*n:(i+r+1)*n])

            if DEBUG:
                # Now solves using cvxopt
                xp = +x
                zp = +z
                solve(xp,y,zp)

            # First phase
            for i in range(r):
                blas.trsm(cholK, a[i])

                Bai = B * a[i]

                c[i] = - Bai - c[i]
                blas.trsm(L22[i], c[i])

                d[i] =  Bai - L32[i] * c[i] - d[i]
                blas.trsm(L33[i], d[i])

                b = b + L42[i] * c[i] + L43[i] * d[i]

            blas.trsm(L44, b)

            # Second phase
            blas.trsm(L44, b, transA='T')

            for i in range(r):
                d[i] = d[i] - L43[i].T * b
                blas.trsm(L33[i], d[i], transA='T')

                c[i] = c[i] - L32[i].T * d[i] - L42[i].T * b
                blas.trsm(L22[i], c[i], transA='T')

                a[i] = a[i] + B.T * (c[i] - d[i])
                blas.trsm(cholK, a[i], transA='T')

            # Store in vectors and scale

            x[n*r:n*r + n] = b
            for i in range(r):
                x[i*n:(i+1)*n] = a[i]
                z[i*n:(i+1)*n] = c[i]
                z[(i+r)*n:(i+r+1)*n] = d[i]

            z[:] = mul( Wd, z)

            if DEBUG:
                print "x  = %s" % x.T,
                print "z  = %s" % z.T,
                print "Delta(x) = %s" % (x - xp).T,
                print "Delta(z) = %s" % (z - zp).T,
                delta= blas.nrm2(x-xp) + blas.nrm2(z-zp)
                if (delta > 1e-8):
                    print "--- DELTA TOO HIGH = %.3e ---" % delta


        return f
    return F
