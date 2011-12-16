#include <vector>
#include <iostream>
#include <boost/exception/diagnostic_information.hpp> 

#include "kqp.hpp"
#include "test.hpp"
#include <cstdlib>

using namespace kqp;

int main(int argc, const char **argv) {  
    

    std::deque<std::string> args;
    for(int i = 1; i < argc; i++) 
        args.push_back(argv[i]);
    
    try {
        long seed = 0;
        
        while (args.size() > 0) {
            if (args[0] == "--seed" && args.size() >= 2) {
                args.pop_front();
                seed = std::atol(args[0].c_str());
                args.pop_front();
            } else break;
            
        }
        if (args.size() < 1)
            KQP_THROW_EXCEPTION(illegal_argument_exception, "No task was given");
        
        std::string name = args[0];
        args.pop_front();       
        
        try {
            if (name == "evd-update") 
                return evd_update_test(args);
            else if (name == "kqp-qp-solver") 
                return kqp_qp_solver_test(args);            
            else if (name == "kernel-evd") 
                return do_kevd_tests(args);            
            
        } catch(const boost::exception &e) {
            std::cerr << boost::diagnostic_information(e) << std::endl;
            throw;
        }
        std::cerr << "No test with name [" << name << "]" << std::endl;
    } catch(exception &e) {

        std::cerr << e.what() << std::endl;
        throw;
    }    
    return 1;
}
