#include <Eigen/Dense>

#include "kqp.hpp"
#include "kernel-evd-tests.hpp"

DEFINE_LOGGER(logger, "kqp.test.kernel_evd")

namespace kqp {
    namespace kevd_tests {
        double tolerance = 1e-10;   
        
        
    }
    
    using namespace kqp::kevd_tests;
    
    
    int do_kevd_tests(std::deque<std::string> &args) {
        if (args.size() != 2) 
            KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Expected two arguments, the kernel EVD builder and task, and got %d", %args.size());
        
        std::string name = args[0];        
        std::string task = args[1];
        
        // Constant random seed
        std::map<std::string, Dense_evd_test> tests;
        {
            // 1 rank-1 update
            Dense_evd_test test;
            test.nb_add = 1;
            test.n = 3;
            test.max_preimages = 1;
            test.max_lc = 1;
            tests["rank-1-once"] = test;
        }
        {
            // 10 rank-1 updates
            Dense_evd_test test;
            test.nb_add = 10;
            test.n = 10;
            test.max_preimages = 1;
            test.max_lc = 1;
            tests["rank-1"] = test;
        }
        {
            // 1 rank-n update
            Dense_evd_test test;
            test.nb_add = 1;
            test.n = 10;
            test.max_preimages = 3;
            test.max_lc = 3;
            test.min_lc = 2;
            tests["rank-n-once"] = test;
        }
        {
            // 2 rank-n update
            Dense_evd_test test;
            test.nb_add = 2;
            test.n = 10;
            test.max_preimages = 3;
            test.max_lc = 3;
            test.min_lc = 2;
            tests["rank-n-twice"] = test;
        }
        
        {
            // 15 rank-n update
            Dense_evd_test test;
            test.nb_add = 15;
            test.n = 10;
            test.max_preimages = 3;
            test.min_lc = 2;
            test.max_lc = 3;
            tests["rank-n"] = test;
        }
        
      
        Dense_evd_test test = tests[task];
        
        if (name == "direct") 
            return kevd_tests::Direct_builder().run(test);
        
        
        if (name == "accumulator") 
            return kevd_tests::Accumulator(true).run(test);
        
        if (name == "accumulator-no-lc")
            return kevd_tests::Accumulator(false).run(test);
        
        if (name == "incremental") 
            return 
            kevd_tests::Incremental().run(test);
        
        
        KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown evd_update_test [%s]", %name);
        
    }
    
    
}