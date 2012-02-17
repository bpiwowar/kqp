/*
 This file is part of the Kernel Quantum Probability library (KQP).
 
 KQP is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 KQP is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with KQP.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <iostream>
#include <boost/exception/diagnostic_information.hpp> 

#include "kqp.hpp"
#include <deque>
#include <cstdlib>

using namespace kqp;


// Map from an ID to a test method
namespace {
    typedef int (*arg_function)(std::deque<std::string>&);
    typedef std::map<std::string, arg_function> TestMap;
    TestMap tests;
    
    struct Declare {
        Declare(std::string name, arg_function f) {
            tests[name] = f;
        }
    };
}

#define DEFINE_TEST(id,name) \
    namespace kqp { int name(std::deque<std::string> &args); } \
    namespace { Declare name ## _decl (id, &name); }

DEFINE_TEST("evd-update", evd_update_test);

DEFINE_TEST("kqp-qp-solver", kqp_qp_solver_test)
DEFINE_TEST("kernel-evd", do_kevd_tests)

DEFINE_TEST("reduced-set/unused", test_reduced_set_unused)
DEFINE_TEST("reduced-set/null-space", test_reduced_set_null_space)
DEFINE_TEST("reduced-set/qp", test_reduced_set_qp)

DEFINE_TEST("probabilities", do_probabilities_tests)

DEFINE_TEST("divergence/simple", divergence_simpleTest);
DEFINE_TEST("divergence/full", divergence_fullTest);
DEFINE_TEST("divergence/simple-epsilon", divergence_simpleEpsilonTest);
DEFINE_TEST("divergence/zero", divergence_zeroTest);

DEFINE_LOGGER(logger,  "kqp.test.main");

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
        
        KQP_LOG_INFO_F(logger, "Setting the seed to %d", %seed);
        std::srand(seed);
        
        try {
            const TestMap::const_iterator it = tests.find(name);
            if (it != tests.end())
               return it->second(args);

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
