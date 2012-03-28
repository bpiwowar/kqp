from solver import solver
from cvxopt import matrix, uniform, setseed

def print_cxx(name, m):
    print "%s << " % name,
    N = m.size[0]*m.size[1]
    for i in xrange(N):
        print "%.15e" % m[i],
        if i != N-1: print ", ",
    print ";"

def doit(name, n,r, g,W, x,z):
    print
    print
    print "// ------- Generated from kkt_test.py ---"
    print "template <typename Scalar> int kkt_test_%s() {" % name
    print
    print "typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;"
    print "typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;"
    print
    print "// Problem"
    print "int n = %d;" % n
    print "int r = %d;" % r
    print "Matrix g(n,n);"

    print "cvxopt::ScalingMatrix<Scalar> w;"
    print "w.d.resize(2*r*n);"
    print "Vector x(n*(r+1)), y, z(2*n*r);"

    print_cxx("x", x)
    print_cxx("z", z)

    print_cxx("g", g)
    print_cxx("w.d", W['d'])

    print
    print "// Solve"
    print "KQP_KKTPreSolver<Scalar> kkt_presolver(g, Vector::Ones(n));"
    print "boost::shared_ptr<cvxopt::KKTSolver<Scalar> > kktSolver(kkt_presolver.get(w));"
    print "kktSolver->solve(x,y,z);"
    print
    
    F = solver(n,r,g)
    sF = F(W)
    sF(x, None, z)

    print
    print "// Solution"
    
    print "Eigen::VectorXd s_x(n*(r+1)), s_z(2*n*r);"
    print_cxx("s_x", x)
    print_cxx("s_z", z)

    print """
            Scalar error_x = (x - s_x).norm() / (Scalar)x.rows();
            Scalar error_z = (z - s_z).norm() / (Scalar)z.rows();

            KQP_LOG_INFO(logger, "Average error (x): " << convert(error_x));
            KQP_LOG_INFO(logger, "Average error (z): " << convert(error_z));
            KQP_LOG_ASSERT(logger, error_x < EPSILON, "Error for x is too high");
            KQP_LOG_ASSERT(logger, error_z < EPSILON, "Error for z is too high");
            return 0;
        }
"""

# --- Simple test

n = 2
r = 2
g = matrix([1,0, 0,1], (n,n), 'd')
x = matrix(1., (n*(r+1),1))
z = matrix(0., (2*n*r,1))
W = {'d': matrix(1., (2*n*r,1))}
doit("simple", n, r, g, W, x, z)

# --- Random test (diagonal g)

setseed(0)
n = 5
r = 10

g = matrix(0., (n,n), 'd')
g[::n+1] = 1

W = {'d': uniform(2*n*r, 1) }

x = uniform(n*(r+1),1)
z = uniform(2*n*r,1)

doit("diagonal_g", n, r, g, W, x, z)

# --- Constant diagonal

setseed(-10)
n = 5
r = 10
g = uniform(n,n)
g = g * g.T

W = {'d': matrix(1., (2*n*r,1))}

x = uniform(n*(r+1),1)
z = uniform(2*n*r,1)

doit("diagonal_d", n, r, g, W, x, z)


# --- Fully random

setseed(10)
n = 5
r = 10
g = uniform(n,n)
g = g * g.T

W = {'d': uniform(2*n*r, 1) }

x = uniform(n*(r+1),1)
z = uniform(2*n*r,1)

doit("random", n, r, g, W, x, z)
