#ifndef NOLOGGING
#include "log4cxx/logger.h"
#include "log4cxx/basicconfigurator.h"
#include "log4cxx/consoleappender.h"
#include "log4cxx/patternlayout.h"
#include "log4cxx/propertyconfigurator.h"
#include "log4cxx/helpers/exception.h"
#endif
#include "kqp.h"

namespace kqp {

double EPSILON = 1e-15;


#ifndef NOLOGGING
const LoggerInit __LOGGER_INIT;

bool LoggerInit::check() {
    return true;
}

LoggerInit::LoggerInit() {
    log4cxx::BasicConfigurator::configure();
    log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("kqp"));
//    log4cxx::ConsoleAppenderPtr appender = new log4cxx::ConsoleAppender();
//    appender->setTarget("system.err");
//    appender->setLayout(new log4cxx::PatternLayout());
//    log4cxx::BasicConfigurator::configure(appender);
    KQP_LOG_DEBUG(logger, "Initialised the logging system");
}
#endif

}

/** 
 
 @mainpage Kernel Quantum Probability project 
 @author B. Piwowarski
 @date May 2011
 
 This project aims at providing an API that allows to compute quantum densities (semi-definite positive hermitian operators) 
 or events (subspaces). It provides tools to compute quantum probabilities and update densities (conditionalisation), and 
 supports the use of kernels to implicitely define the space, thus allowing working in very high dimensional spaces. 

 This project is <a href="http://sourceforge.net/projects/kqp/">hosted on SourceForge</a>.
*/



/**
 @page Main classes
 @author B. Piwowarski
 @date October 2011
 
 There are three main modules:
 - \ref FeatureMatrix "Feature matrices" are used to represent a set of feature vector. In a finite vectorial space, this is typically a matrix. 
 - \ref OperatorBuilder "Operator builders" are the classes that compute a thin representations of (kernel) linear operators, based on a single type of feature matrix;
 - \ref Probabilities "Probabilities" can then be computed from built operators (they both define events and quantum probability densities).
 
*/



/**
 @defgroup FeatureMatrix Feature matrices
 
 Feature matrices are used to represent a set of feature vector. In a finite vectorial space, this is typically a matrix.
 
*/

/**
 @defgroup OperatorBuilder Building kernel linear operators

*/

/**
 @defgroup Probabilities Computing quantum probabilities
 
 
*/
