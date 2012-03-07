
#include <kqp/kqp.hpp>

namespace kqp {
    
    // Machine zero
    double EPSILON = 1e-15;
        
#ifndef NOLOGGING
    // Main KQP logger
    log4cxx::LoggerPtr main_logger(log4cxx::Logger::getLogger("kqp"));

#endif
    
}


/** 
 
 @mainpage Kernel Quantum Probability project 
 @author B. Piwowarski
 @date May 2011
 
 This project aims at providing an API that allows to compute quantum densities (semi-definite positive hermitian operators) 
 or events (subspaces). It provides tools to compute quantum probabilities and update densities (conditionalisation), and 
 supports the use of kernels to implicitely define the space, thus allowing working in very high dimensional spaces. 
 
 This project is <a href="https://github.com/bpiwowar/kqp">hosted on Github</a>.
 */



/**
 @page Main classes
 @author B. Piwowarski
 @date October 2011
 
 There are three main modules:
 - \ref FeatureMatrix "Feature matrices" are used to represent a set of feature vector. In a finite vectorial space, this is typically a matrix. 
 - \ref KernelEVD "Operator builders" are the classes that compute a thin representations of (kernel) linear operators, based on a single type of feature matrix;
 - \ref Probabilities "Probabilities" can then be computed from built operators (they both define events and quantum probability densities).
 
 */



/**
 @defgroup FeatureMatrix Feature matrices
 
 Feature matrices are used to represent a set of feature vector. In a finite vectorial space, this is typically a matrix.
 
 */

/**
 @defgroup KernelEVD Building kernel linear operators
 
 */

/**
 @defgroup Probabilities Computing quantum probabilities
 
 
 */
