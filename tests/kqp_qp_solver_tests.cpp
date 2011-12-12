 #include <iostream>
#include <complex>

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include "kqp.hpp"
#include "null_space.hpp"
#include "coneprog.hpp"
#include "test.hpp"

DEFINE_LOGGER(logger, "kqp.test.kqp-qp-solver")


namespace kqp {
   
    
#include "generated/kkt_solver.cpp"
#include "generated/qp_solver.cpp"
    
    int kqp_qp_solver_test(std::vector<std::string> &args) {
        KQP_LOG_INFO(logger, "Starting qp solver tests");
        std::string name = args[0];
        
        if (name == "kkt-solver-simple") 
            return kkt_test_simple();

        if (name == "kkt-solver-diagonal-g")
            return kkt_test_diagonal_g();

        if (name == "kkt-solver-diagonal-d")
            return kkt_test_diagonal_d();
        
        if (name == "kkt-solver-random")
            return kkt_test_random();

        
        
        if (name == "simple") 
            return qp_test_simple();
        
        if (name == "random") 
            return qp_test_random();
        
        BOOST_THROW_EXCEPTION(illegal_argument_exception()
                              << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
        
        return 0;
    }
    
}