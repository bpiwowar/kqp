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
            if transpose: print "%.50g" % m[j,i],
            else: print "%.50g" % m[i,j],
            if i != rows-1 or j != cols - 1: print ", ",
    print ";"


def doit(name, n,r, g, a, nu, Lambda):
    zero_n = spmatrix([],[],[],(n,n))
    id_n = spmatrix(1., range(n), range(n))

    print
    print
    print "// ------- Generated from qp_test.py ---"
    print "template<typename Scalar> int qp_test_%s() {" % name
    print
    print "typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;"
    print "typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;"
    print
    print "// Problem"
    print "int n = %d;" % n
    print "int r = %d;" % r
    print "Matrix g(n,n);"
    print "Matrix a(r, n);"
    print "Vector nu(n, 1);"
    print "Scalar lambda = %50g;" % Lambda

    print_cxx("a", a)
    print "a.adjointInPlace();"
    print_cxx("g", g)
    
    print_cxx("nu", nu)

    print
    print "// Solve"

    print "kqp::cvxopt::ConeQPReturn<Scalar> result;"
    print "solve_qp(r, lambda, g, a, nu, result, options);"
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
        q[i*n:(i+1)*n,0] = - g * a[i*n:(i+1)*n,0]
    q[n*r:n*r+n] = Lambda / 2.
    if DEBUG > 1: print "q = %s" % q.T,

    print "Constructing G (%d x %d) and q" % (2 * n*r, n*r + n)
	
    s = []
    for i in range(r): s += [nu[i]] * n
    s_nr = spmatrix(s, range(n*r), range(n*r))
    id_col = []
    for i in range(r):
        id_col.append(-id_n)
    id_col = sparse([id_col])
    G = sparse([ [ -s_nr, s_nr ], [id_col, id_col ] ])
    h = matrix(0., (2*n*r,1))

    dims = {"l": h.size[0], "q": 0, "s": 0}

    sol = solvers.coneqp(P, q, G, h) #, kktsolver=solver(n,r,g))

    print "*/"
    print
    print "// Solution"
    if sol["status"] != "optimal": raise "Solution is not optimal..."
    
    print "Eigen::VectorXd s_x(n*(r+1));"
    print_cxx("s_x", sol["x"])

    print """
            double error_x = (result.x - s_x).norm() / s_x.rows();
            // const double threshold = std::max(epsilon() * s_x.norm() / s_x.rows(), epsilon());
            const double threshold = 1e-10;
            KQP_LOG_INFO_F(logger, "Average error (x) = %g [threshold %g]", %error_x % threshold);
            KQP_LOG_ASSERT(logger, error_x < threshold, "Error for x is too high");
            return 0;
        }
"""


# --- Simple test

n = 2
r = 2
g = matrix([1,0, 0,1], (n,n), 'd')
a = matrix([1, 0, 0, 0.4], (n*r,1), 'd')
nu = matrix(1.,(n,1),'d')
doit("simple", n, r, g, a, nu, 1.)


# --- Random test

setseed(1)
n = 8
r = 5
g = uniform(n,n)
g = g * g.T
a = uniform(n*r,1)
nu = matrix(1.,(n,1),'d')
doit("random", n, r, g, a, nu, 1.)

# --- Simple test

n = 2
r = 2
g = matrix([1,0, 0,1], (n,n), 'd')
a = matrix([1, 0, 0, 0.4], (n*r,1), 'd')

nu = uniform(n,1,2,10)
doit("simple_nu", n, r, g, a, nu, 1.)


# --- Random test

n = 8
r = 5
g = uniform(n,n)
g = g * g.T
a = uniform(n*r,1)
nu = uniform(n,1,0.1,2)
doit("random_nu", n, r, g, a, nu, 5.)

