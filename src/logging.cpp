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

#ifndef NOLOGGING

#include "log4cxx/logger.h"
#include "log4cxx/basicconfigurator.h"
#include "log4cxx/consoleappender.h"
#include "log4cxx/patternlayout.h"
#include "log4cxx/propertyconfigurator.h"
#include "log4cxx/helpers/exception.h"

#include <kqp/kqp.hpp>
#include <kqp/logging.hpp>

namespace kqp {
    LoggerConfig LOGGER_CONFIG;
    
    LoggerConfig::LoggerConfig() : initialized(false) {
    }
    
    
    void LoggerConfig::setLevel(const std::string &loggerId, const std::string &level) {
        // By default, use info
        log4cxx::Logger::getLogger(loggerId)->setLevel(log4cxx::Level::toLevel(level, log4cxx::Level::getInfo()));
    }
    
    
    void LoggerConfig::setDefaultLevel(const std::string &level) {
        log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::toLevel(level, log4cxx::Level::getInfo()));
    }
    
    void LoggerConfig::prepareLogger() {
        if (initialized) return;  
        
        log4cxx::PatternLayoutPtr layout = new log4cxx::PatternLayout("%5p [%r] (%F:%L) - %m%n");
        
        log4cxx::ConsoleAppenderPtr appender = new log4cxx::ConsoleAppender(layout, log4cxx::ConsoleAppender::getSystemErr());
        appender->setName("kqp-appender");
        log4cxx::BasicConfigurator::configure(appender);
        
        initialized = true;
        
        KQP_LOG_DEBUG(main_logger, "Initialised the logging system");
    }
    
    


    void prepareLogger() {
        LOGGER_CONFIG.prepareLogger();
    }
}
#endif
