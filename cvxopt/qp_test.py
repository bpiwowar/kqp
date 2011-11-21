from solver import solver
from cvxopt import matrix, uniform, setseed, spmatrix, sparse, solvers, printing

printing.options['width'] = 30
solvers.options['DEBUG'] = 0
DEBUG = 0

## print row per row
def print_cxx(name, m, transpose=False):
    print "%s << " % name,
    if transpose: (cols, rows) = m.size
    else: (rows, cols) = m.size
        
    for i in xrange(rows):
        for j in xrange(cols):
            if transpose: print "%.15e" % m[j,i],
            else: print "%.15e" % m[i,j],
            if i != rows-1 or j != cols - 1: print ", ",
    print ";"


def doit(name, n,r, g, a, Lambda):
    zero_n = spmatrix([],[],[],(n,n))
    id_n = spmatrix(1., range(n), range(n))

    print
    print
    print "// ------- Generated from kkt_test.py ---"
    print "int qp_test_%s() {" % name
    print
    print "// Problem"
    print "int n = %d;" % n
    print "int r = %d;" % r
    print "Eigen::MatrixXd g(n,n);"

    print "Eigen::MatrixXd a(r, n);"
    print "double lambda = %15e;" % Lambda

    print_cxx("a", a)
    print "a.adjointInPlace();"
    print_cxx("g", g)

    print
    print "// Solve"

    print "kqp::cvxopt::ConeQPReturn result;"
    print "solve_qp(n, r, lambda, g, a, result);"
    print

    # Construct P

    print "/*"
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
        if DEBUG > 0: print "a[%d] = %s" % (i, a[i*n:(i+1)*n,0].T),
        q[i*n:(i+1)*n,0] = - 2 * g * a[i*n:(i+1)*n,0]
    q[n*r:n*r+n] = Lambda
    if DEBUG > 1: print "q = %s" % q.T,

    print "Constructing G (%d x %d) and q" % (2 * n*r, n*r + n)
    id_nr = spmatrix(1., range(n*r), range(n*r))
    id_col = []
    for i in range(r):
        id_col.append(-id_n)
    id_col = sparse([id_col])
    G = sparse([ [ -id_nr, id_nr ], [id_col, id_col ] ])
    h = matrix(0., (2*n*r,1))

    dims = {"l": h.size[0], "q": 0, "s": 0}

    sol = solvers.coneqp(P, q, G, h, kktsolver=solver(n,r,g))

    print "*/"
    print
    print "// Solution"
    if sol["status"] != "optimal": raise "Solution is not optimal..."
    
    print "Eigen::VectorXd s_x(n*(r+1));"
    print_cxx("s_x", sol["x"])

    print """
            double error_x = (result.x - s_x).norm() / (double)s_x.rows();

            KQP_LOG_INFO(logger, "Average error (x): " << convert(error_x));
            KQP_LOG_ASSERT(logger, error_x < EPSILON, "Error for x is too high");
            return 0;
        }
"""


# --- Simple test

n = 2
r = 2
g = matrix([1,0, 0,1], (n,n), 'd')
a = matrix([1, 0, 0, 0.4], (n*r,1), 'd')
doit("simple", n, r, g, a, 1.)


# --- Simple test

setseed(1)
n = 8
r = 5
g = uniform(n,n)
g = g * g.T
a = uniform(n*r,1)
doit("random", n, r, g, a, 1.)
