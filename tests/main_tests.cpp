#include <iostream>
#include <boost/exception/diagnostic_information.hpp> 

#include "kqp.hpp"
#include "test.hpp"
#include <boost/program_options.hpp>
#include <cstdlib>

using namespace kqp;
namespace po = boost::program_options;

int main(int argc, const char **argv) {  
    po::variables_map vm;
    
    po::options_description general_options("General options");
    general_options.add_options()
    ("help", "produce help message")
    ("seed", po::value<unsigned int>()->default_value(0), "Random seed value")
    ("task", po::value<std::string>(), "Task name");
    
    
    po::options_description options;
    options.add(general_options);
    
    try {
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(general_options).allow_unregistered().run(); 
        po::store(parsed, vm);
        po::notify(vm);   
        
        if (vm.count("help")) 
        { 
            std::cout << general_options << std::endl; 
            return 0; 
        } 
        
        std::srand(vm["seed"].as<unsigned int>() );
        if (vm["task"].empty())
            KQP_THROW_EXCEPTION(illegal_argument_exception, "No task was given");
        std::string name = vm["task"].as<std::string>();
        
        std::vector<std::string> other_argv = po::collect_unrecognized(parsed.options, po::include_positional);
        
        
        try {
            if (name == "evd-update") 
                return evd_update_test(other_argv);
            else if (name == "kqp-qp-solver") 
                return kqp_qp_solver_test(other_argv);            
            else if (name == "kernel-evd") 
                return kevd_tests(other_argv);            
            
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
