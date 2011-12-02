#ifndef __KQP_TESTS_H__
#define __KQP_TESTS_H__


namespace kqp {
    //! Test the rank-one EVD update
    int evd_update_test(int argc, const char **argv);
    
    //! The kqp QP solver
    int kqp_qp_solver_test(int argc, const char **argv);
    
    //! The kernel EVD tests
    int kevd_tests(int argc, const char **argv);
}

#endif
