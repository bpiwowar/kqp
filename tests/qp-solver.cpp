#include <iostream>
#include <complex>
#include <deque>

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <kqp/kqp.hpp>
#include <kqp/coneprog.hpp>
#include <kqp/reduced_set/qp_approach.hpp>

DEFINE_LOGGER(logger, "kqp.test.kqp-qp-solver")


namespace kqp {
   
    
#include "generated/kkt_solver.cpp"
#include "generated/qp_solver.cpp"
    
    int kqp_qp_solver_test(std::deque<std::string> &args) {
        KQP_LOG_INFO(logger, "Starting qp solver tests");
        std::string name = args[0];
        
        if (name == "kkt-solver-simple") 
            return kkt_test_simple<double>();

        if (name == "kkt-solver-diagonal-g")
            return kkt_test_diagonal_g<double>();

        if (name == "kkt-solver-diagonal-d")
            return kkt_test_diagonal_d<double>();
        
        if (name == "kkt-solver-random")
            return kkt_test_random<double>();

        
        
        if (name == "simple") 
            return qp_test_simple<double>();
        
        if (name == "random") 
            return qp_test_random<double>();
        
        BOOST_THROW_EXCEPTION(illegal_argument_exception()
                              << errinfo_message((boost::format("Unknown evd_update_test [%s]") % name).str()));
        
        return 0;
    }
    
}