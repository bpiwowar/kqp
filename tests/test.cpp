#include <iostream>
#include "fastRankOneUpdate.h"

int main(int argc, const char **argv) {
    std::cerr << "Everything went OK" << std::endl;
    
    if (argc < 2) {
        std::cerr << "No test name given" << std::endl;
        return 1;   
    }
    
    std::string name = argv[1];

    
    
    std::cerr << "No test with name [" << name << "]" << std::endl;
    return 1;
}
