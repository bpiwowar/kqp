#include <iostream>
#include <boost/exception/diagnostic_information.hpp> 

#include "kqp.h"
#include "test.h"

using namespace kqp;

int main(int argc, const char **argv) {
   
    if (argc < 2) {
        std::cerr << "No test name given" << std::endl;
        return 1;   
    }
    
    std::string name = argv[1];
    const char **other_argv = &argv[2];
    
    try {
        if (name == "evd-update") 
            return evd_update_test(argc - 2, other_argv);
        else if (name == "kqp-qp-solver") {
            return kqp_qp_solver_test(argc - 2, other_argv);            
        }
    } catch(const boost::exception &e) {
        std::cerr << boost::diagnostic_information(e) << std::endl;
        throw;
    }
    std::cerr << "No test with name [" << name << "]" << std::endl;
    return 1;
}
