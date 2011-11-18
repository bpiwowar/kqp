from solver import solver
from cvxopt import matrix, uniform, setseed

def print_cxx(name, m):
    print "%s << " % name,
    N = m.size[0]*m.size[1]
    for i in xrange(N):
        print "%.15e" % m[i],
        if i != N-1: print ", ",
    print ";"

def doit(n,r, g,W, x,z):
    print
    print "// -------"
    print "// Problem"
    print "int n = %d;" % n
    print "int r = %d;" % r
    print "Eigen::MatrixXd g(n,n);"

    print "cvxopt::ScalingMatrix w;"
    print "w.d.resize(2*r*n);"
    print "Eigen::VectorXd x(n*(r+1)), y, z(2*n*r);"

    print_cxx("x", x)
    print_cxx("z", z)

    print_cxx("g", g)
    print_cxx("w.d.diagonal()", W['d'])

    print
    print "// Solve"
    print "KQP_KKTPreSolver kkt_presolver(g);"
    print "boost::shared_ptr<cvxopt::KKTSolver> kktSolver(kkt_presolver.get(w));"
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


# --- Simple test

n = 2
r = 2
g = matrix([1,0, 0,1], (n,n), 'd')
x = matrix(1., (n*(r+1),1))
z = matrix(0., (2*n*r,1))
W = {'d': matrix(1., (2*n*r,1))}
doit(n, r, g, W, x, z)

# --- Random test

setseed(0)
n = 5
r = 10
g = uniform(2,2)
g = g * g.T
W = {'d': uniform(2*n*r, 1) }


x = uniform(n*(r+1),1)
z = uniform(2*n*r,1)

doit(n, r, g, W, x, z)
