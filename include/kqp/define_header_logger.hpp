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


// Fixme: __COUNTER__ might not be compatible with some compilers
// Works with GCC, clang, MSVC

#include <boost/preprocessor/cat.hpp>

#include <kqp/kqp.hpp>


# ifndef KQP_HLOGGER

#   define KQP_HLOGGER_N 1

#   // Defines helper macros

#   define KQP_HLOG_DEBUG(MESSAGE) KQP_LOG_DEBUG(KQP_HLOGGER, MESSAGE)
#   define KQP_HLOG_INFO(MESSAGE)  KQP_LOG_INFO(KQP_HLOGGER, MESSAGE)
#   define KQP_HLOG_WARN(MESSAGE)  KQP_LOG_WARN(KQP_HLOGGER, MESSAGE)
#   define KQP_HLOG_ERROR(MESSAGE)  KQP_LOG_ERROR(KQP_HLOGGER, MESSAGE)

#   define KQP_HLOG_DEBUG_F(FORMAT, OBJS) KQP_LOG_DEBUG_F(KQP_HLOGGER, FORMAT, OBJS)
#   define KQP_HLOG_INFO_F(FORMAT, OBJS)  KQP_LOG_INFO_F(KQP_HLOGGER, FORMAT, OBJS)
#   define KQP_HLOG_WARN_F(FORMAT, OBJS)  KQP_LOG_WARN_F(KQP_HLOGGER, FORMAT, OBJS)
#   define KQP_HLOG_ERROR_F(FORMAT, OBJS)  KQP_LOG_ERROR_F(KQP_HLOGGER, FORMAT, OBJS)

# else 
#   undef KQP_HLOGGER
#   undef DEFINE_KQP_HLOGGER
#   if (KQP_HLOGGER_N == 1) 
#     undef  KQP_HLOGGER_N
#     define KQP_HLOGGER_N 2
#   elseif (KQP_HLOGGER_N == 2) 
#     undef  KQP_HLOGGER_N
#     define KQP_HLOGGER_N 3
#   elseif (KQP_HLOGGER_N == 3) 
#     undef  KQP_HLOGGER_N
#     define KQP_HLOGGER_N 4
#   else
#     error
#   endif
# endif

# // New KQP Header logger
# define KQP_HLOGGER BOOST_PP_CAT(KQP_HEADER_LOGGER_, KQP_HLOGGER_N)

# // New definition
# define DEFINE_KQP_HLOGGER(NAME) DEFINE_LOGGER(KQP_HLOGGER, NAME)

