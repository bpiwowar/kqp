#!/usr/bin/python

# Tests with cvxopt
#
# Preliminar to implementation in C++
#
from cvxopt import solvers, matrix, sparse, spmatrix, uniform, printing, mul, div, misc, lapack, blas


printing.options['dformat'] = '%.1f'
printing.options['width'] = 15


# n is the number of vectors used to build the basis
# r is the rank (number of basis vectors)
# [Note that n >= r]

choice = "random"

if choice == "random":
    # Bulds up a random example
    n = 30
    r = 10
    Lambda = 0.1

    # Construct an n * n positive definite matrix by computing a lower
    # triangular matrix with positive diagonal entries
    
    A = uniform(n, n)
    for i in range(n):
        A[i,i] = abs(A[i,i])+0.3
        for j in range(i+1,n): A[i,j] = 0

    g = A * A.T
    mA = +g
    
    a = uniform(n * r)
    
elif choice == "simple-1":
    n = 2
    r = 2
    Lambda = 1
    g = matrix([1,0,0,1], (2,2))
    a = matrix([1,0,0,.5], (n*r, 1))
else:
    print "Unknown choice [%s]" % choice
    sys.exit(1)
    
# --- Builds and solve


# n is the number of feature vectors
# r is the number of basis vectors
# a is the list of 
# Lambda is the regularisation coefficient
# Gram matrix g (size n x n)
# the coefficients a (size nr x 1)

# We have n*r + n variables

zero_n = spmatrix([],[],[],(n,n))
id_n = spmatrix(1., range(n), range(n))

# Construct P

print "Constructing P..."
l = []
for i in range(r+1):
    sl = []
    for j in range(r+1):
        if i < r and i == j: sl.append(g)
        else: sl.append(zero_n)
    l.append(sl)
P = sparse(l)

print "Constructing q..."
q = matrix(0., (n * r + n, 1))
for i in range(r):
    q[i*n:(i+1)*n,0] = - g * a[i*n:(i+1)*n,0]
q[n*r:n*r+n] = Lambda / 2.

print "Constructing G (%d x %d) and q" % (2 * n*r, n*r + n)
id_nr = spmatrix(1., range(n*r), range(n*r))
id_col = []
for i in range(r):
    id_col.append(-id_n)
id_col = sparse([id_col])
G = sparse([ [ -id_nr, id_nr ], [id_col, id_col ] ])
h = matrix(0., (2*n*r,1))


dims = {"l": h.size[0], "q": 0, "s": 0}

# --- Custom

chol2 = misc.kkt_chol2(G, dims, spmatrix([], [], [], (0, q.size[0])))

def Fchol2(W):
    """
    Uses the Cholesky factorisation, in order to see how the 
    optimisation works
    """
    
    solve = chol2(W,P)
	
#    di = W['di']
#    print W['di']
    
    def f(x, y, z):
        print "** SOLVING KKT **"
        return solve(x,y,z)

    return f


cholK = None

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
                             
def F(W):

    """
    Returns a function f(x, y, z) that solves the KKT conditions

    """
    global cholK

    if cholK is None:
        # Copy A and compute the Cholesky decomposition
        print "# Computing the cholesky decomposition of P"
        cholK = +g
        cholesky(cholK)
        
        print "# Computing the B in B A.T = Id (L21 and L31)"
        B = matrix(id_n, (n,n))
        lapack.potrs(cholK, B)

        print "# Computing B B^T"
        BBT = B * B.T


    U = W['di'][0:n*r] ** 2
    V = W['di'][n*2+1:2*n*r]  ** 2
  
    print "# Computing L22"
    L22 = []
    for i in range(r):
        BBTi = BBT + spmatrix(U[i*n:(i+1)*n] ,range(n),range(n))
        cholesky(BBTi)
        L22.append(BBTi)

    print "# Computing L32"
    L32 = []
    for i in range(r):
        # Solves L32 . L22 = - B . B^\top
        B = -BBT
        blas.trsm(L22[i], B, side='R', transA='T')
        L32.append(B)
    
    print "# Computing L33"
    L33 = []
    for i in range(r):
        A = V[i]  + BBT - L32[i] * L32[i].T 
        cholesky(A)
        L33.append(A)
        
    print "# Computing L42"
    L42 = []
    for i in range(r):
        A = +L22[i].T
        lapack.trtri(A)
        L42.append(A)
    
    print "# Computing L43"
    L43 = []
    for i in range(r):
        A =  L42[i] * L32[i].T + id_n
        blas.trsm(L33[i], A, side='R', transA='T')
        L43.append(A)

    print "# Computing L44 and D4"

    
    # The indices for the diagonal of a dense matrix
    L44 = matrix(0, (n,n))
    for i in range(r):
        L44 = L44 + L43[i] * L43[i].T + L42[i] * L42[i].T
   
    cholesky(L44)

    # WARNING: y and t have been permuted (LD33 and LD44piv)
    print "## PRE-COMPUTATION DONE ##"
    
    

    def f(x, y, z):
        """
        On entry bx, bz are stored in x, z.  On exit x, z contain the solution,
        with z scaled: z./di is returned instead of z.
        """

        # Maps to our variables x,y,z and t
        a = []
        b = -x[n*r:n*r + n]
        c = []
        d = []
        for i in range(r):
            a.append(x[i*n:(i+1)*n])
            c.append(z[i*n:(i+1)*n])
            d.append(z[(i+r)*n:(i+r+1)*n])
        
        # First phase
        for i in range(r):
            blas.trsm(cholK, a[i])

            Bai = B * a[i]
            
            c[i] = Bai - c[i]
            blas.trsm(L22[i], c[i])

            d[i] = Bai - d[i] - L32[i] * c[i]
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

        # Store in vectors and scale
        x[n*r:n*r + n] = b
        for i in range(r):
            x[i*n:(i+1)*n] = a[i]

            # Normalise
            for j in range(n):
                c[i][j] = c[i][j] / U[i*n+j]
                d[i][j] = d[i][j] / V[i*n+j]
                
            z[i*n:(i+1)*n] = c[i]
            z[(i+r)*n:(i+r+1)*n] = d[i]

        print "Done"
        
    return f

# --- Init values (x, s and y, z)

# initvals = 

# --- Solving 

print "   [[[Solving system with default...]]]"
sol=solvers.coneqp(P, q, G, h, kktsolver=Fchol2)
print sol['status']
if (n * r < 10): print sol['x']

print "\n\n   [[[Solving with optimised]]]"
sol = solvers.coneqp(P, q, G, h, kktsolver=F)
print sol['status']

print "Done"
