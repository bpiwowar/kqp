#ifndef __KQP_TESTS_H__
#define __KQP_TESTS_H__

#include <vector>

namespace kqp {
    //! Test the rank-one EVD update
    int evd_update_test(std::deque<std::string> &args);
    
    //! The kqp QP solver
    int kqp_qp_solver_test(std::deque<std::string> &args);
    
    //! The kernel EVD tests
    int do_kevd_tests(std::deque<std::string> &args);
}

#endif
