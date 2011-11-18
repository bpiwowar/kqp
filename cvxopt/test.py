#!/usr/bin/python

# Tests with cvxopt
#
# Preliminar to implementation in C++
#
from cvxopt import solvers, matrix, sparse, spmatrix, uniform, printing, mul, div, misc, lapack, blas, spdiag
from time import  time

printing.options['dformat'] = '%.2e'
printing.options['width'] = 15


# n is the number of vectors used to build the basis
# r is the rank (number of basis vectors)
# [Note that n >= r]

DEBUG = 0
choice = "simple-1"

if choice == "random":
    # Bulds up a random example
    n = 50
    r = 25
    np = 30
    Lambda = 1000

    # Construct an n * n positive definite matrix by computing a lower
    # triangular matrix with positive diagonal entries
    
    A = uniform(n, n)
    for i in range(n):
        A[i,i] = abs(A[i,i])+0.2
        for j in range(i+1,n): A[i,j] = 0

    print A[::n+1].T

    g = A * A.T
    mA = +g

    # Construct the vectors
    a = uniform(n * r)
    
elif choice == "simple-1":
    n = 2
    r = 2
    np = 1
    Lambda = 1
    g = matrix([1.,0.,0.,1.], (2,2))
    a = matrix([1,0,0,.5], (n*r, 1))
else:
    print "Unknown choice [%s]" % choice
    sys.exit(1)


# --- Setting lambda

# Computes the Deltas
errors = []
for j in xrange(n):
    delta = 0.
    maxa = 0
    for i in xrange(r):
        delta += a[i*n + j] ** 2
        maxa = max(maxa, abs(a[i*n + j]))
        
    errors.append({'delta': delta * g[j,j], 'maxa': maxa, 'index': j })

# Sort by delta
errors.sort(key=lambda error: error['delta'])
Lambda = sum(e['delta'] for e in errors[0:np]) / sum(e['maxa'] for e in errors[0:np])


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
    q[i*n:(i+1)*n,0] = - 2 * g * a[i*n:(i+1)*n,0]
q[n*r:n*r+n] = Lambda

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
        if DEBUG > 0: print "** SOLVING KKT **"
        if DEBUG > 1: print "bx = %sbz = %s" % (x.T, z.T),
        solve(x,y,z)
        if DEBUG > 1: print "x = %sz = %s" % (x.T, z.T),

    return f


cholK = None


from solver import solver

# --- Init values (x, s and y, z)


# --- Solving 

if (n * r < 10):
    print "=== Problem structure ==="
    print
    print "=== P"
    print P
    print
    print "=== q"
    print q.T
    print
    print "=== G"
    print G
    print
    print "=== h"
    print h.T
    print

    
print "=== Problem size: n=%d and r=%d ===\n\n" % (n, r)
print "* np = %d" % np
print "* lambda = %g" % Lambda

print "\n\n   [[[Solving with optimised]]]"
T1 = time()
sol = solvers.coneqp(P, q, G, h, kktsolver=solver(n,r,g))
print "Time taken = %s" % (time() - T1)
print sol['status']
if (n * r < 10): print "Solution = %s" % sol['x'].T

printing.options['width'] = n

nzeros=0
xi = sol['x'][n*r:n*(r+1)]
maxxi=max(xi)
print sum(xi[0:n])
for i in xrange(n):
    if xi[i]/maxxi < 1e-4: nzeros += 1

print "xi = %s" % sorted(xi.T/maxxi)
print "Sparsity = %d on %d" % (nzeros, n)

print "   [[[Solving system with default...]]]"
T1 = time()
sol=solvers.coneqp(P=P, q=q, G=G, h=h, kktsolver=Fchol2)
print "Time taken = %s" % (time() - T1)
print sol['status']
if (n * r < 10):
    print "Solution = %s" % sol['x'].T
    print (G * sol['x']).T


print "Done"
