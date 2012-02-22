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

#ifndef __KQP_LOGGING_H__
#define __KQP_LOGGING_H__

#ifndef NOLOGGING

#include <map>

namespace kqp {
    class LoggerConfig {
    public:
        //! Default constructor
        LoggerConfig();
                
        //! Set the level for one logger
        void setLevel(const std::string &loggerId, const std::string &level);

        //! Set the root logger default level
        void setDefaultLevel(const std::string &level);

        //! Prepare logger
        void prepareLogger();
        
    private:
        bool initialized;
    };

    //! The global logger configurator
    extern LoggerConfig LOGGER_CONFIG;
}

#endif
#endif