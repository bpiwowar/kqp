#include <Eigen/Dense>

#include "kqp.hpp"
#include "kernel-evd-tests.hpp"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd")

namespace kqp {
    namespace kevd_tests {
        double tolerance = 1e-10;   
        
        int direct_builder(const Dense_evd_test &);
        int accumulator(const Dense_evd_test &, bool);
        int incremental(const Dense_evd_test &);
        
    }
    
    using namespace kqp::kevd_tests;
    
    int do_kevd_tests(std::deque<std::string> &args) {
        std::string name = args[0];
        
        // Constant random seed
        Dense_evd_test test;
        test.nb_add = 10;
        test.n = 10;
        test.max_preimages = 1;
        test.max_lc = 1;
        
        if (name == "direct-builder") 
            return kevd_tests::direct_builder(test);
        
        if (name == "accumulator") 
            return kevd_tests::accumulator(test, false);
        
        if (name == "accumulator-no-lc")
            return kevd_tests::accumulator(test, true);
        
        if (name == "incremental") 
            return kevd_tests::incremental(test);

        
        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown evd_update_test [%s]", %name);
        
    }
    
    
}