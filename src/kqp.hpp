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
#ifndef __KQP_H__
#define __KQP_H__

#include <boost/exception/errinfo_at_line.hpp>
#include <boost/exception/info.hpp>
#include <boost/exception/exception.hpp>
#include <boost/format.hpp>
#include <string>
#include <complex>
#include "cxxabi.h"

#include "Eigen/Core"

#ifndef NOLOGGING
#include "log4cxx/logger.h"
#endif


namespace kqp {
    
    typedef Eigen::DenseIndex Index;
    
    /** Check if the value is a NaN */
    inline bool isNaN(double x) {
        return !(x == x);
    }
    
    inline bool isNaN(float x) {
        return !(x == x);
    }
    
    template<typename scalar> inline bool isNaN(const std::complex<scalar> &x) {
        return !(std::real(x) == std::real(x)) || !(std::imag(x) == std::imag(x));
    }
    
    
    /** Anything below is considered zero in approximations */
    extern double EPSILON;
    
    
    typedef boost::error_info<struct errinfo_file_name_,std::string> errinfo_message;
    
    /**Base class for exceptions */
    class exception : public virtual std::exception, public  virtual boost::exception  {};   
    
    /** Illegal argument */
    class illegal_argument_exception : public virtual exception {};
    
    /** Arithmetic exception */
    class arithmetic_exception : public virtual exception {};
    
    /** Out of bound exception */
    class out_of_bound_exception : public virtual exception {};
    
    /** Illegal operation */
    class illegal_operation_exception : public virtual exception {};
    
    /** Not implemented */
    class not_implemented_exception : public virtual exception {};

    /** Assertion */
    class assertion_exception : public virtual exception {};

    
    inline std::string demangle(const std::type_info &x) {
        //     __cxa_demangle(const char* __mangled_name, char* __output_buffer, size_t* __length, int* __status);
        static size_t size = 500;
        static char *buffer = (char*)malloc(size * sizeof(char));
        return abi::__cxa_demangle(x.name(), buffer, &size, 0);
    }
    
    /** Convert anything to a string. Might be specialized */
    template <class T> std::string convert(const T &x) {
        std::ostringstream strout;
        strout << x;
        return strout.str();
    }
    
    /** Convert anything to a string. Might be specialized */
    template<class T> T convert(const std::string &s) {
        T x;
        std::istringstream str_in(s);
        if (!str_in) BOOST_THROW_EXCEPTION(errinfo_message("Bad input"));
        else if (! (str_in >> x )) BOOST_THROW_EXCEPTION(errinfo_message("Bad input"));
        return x;
    }
    

#define KQP_FOR_ALL_SCALAR_TYPES(prefix, suffix) \
    prefix double suffix; \
    prefix float suffix; \
    prefix std::complex<double> suffix; \
    prefix std::complex<float> suffix;

    //! Returns an empty matrix (useful as a constant)
    template<typename Scalar>
    struct EMPTY {
        static const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix();
    };
    
    template<typename Scalar>
    bool is_empty(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &m) { 
        return m.rows() == 0 && m.cols() == 0;
    }
    
    template<typename Scalar>
    bool is_empty(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &m) { 
        return m.rows() == 0;
    }

    
    KQP_FOR_ALL_SCALAR_TYPES(extern template struct kqp::EMPTY<, >);

    // --- USEFUL MACROS ---
    

#define KQP_MATRIX(Scalar) Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> 
#define KQP_VECTOR(Scalar) Eigen::Matrix<Scalar, Eigen::Dynamic, 1> 
    
    
    
    //! Demangle a pointer
#define KQP_DEMANGLEP(x) (x ? kqp::demangle(typeid(x)) : kqp::demangle(typeid(*x)))
    //! Demangle a reference
#define KQP_DEMANGLE(x) kqp::demangle(typeid(x))
    
    //! Hidden macro for STRING_IT(x)
#define KQP_XSTRING_IT(x) #x
    
    //! String-ize the parameter
#define KQP_STRING_IT(x) KQP_XSTRING_IT(x)
    
    
    // ---- LOGGING MACROS ---
    
#ifdef NOLOGGING
    
#define DEFINE_LOGGER(logger, loggerName)
#define KQP_LOG_DEBUG(name,message)
#define KQP_LOG_DEBUG_S(name,message)
#define KQP_LOG_INFO(name,message)
#define KQP_LOG_WARN(name,message)
#define KQP_LOG_ERROR(name,message)
#define KQP_LOG_ASSERT(name,condition,message)

//! Throw an exception with a message
#define KQP_THROW_EXCEPTION(type, message) BOOST_THROW_EXCEPTION(type() << errinfo_message(message))

#else
    
    class LoggerInit {
    public:
         
        LoggerInit();
        static bool check();
    };
    extern const LoggerInit __LOGGER_INIT;
    extern log4cxx::LoggerPtr main_logger;
    
    // We define the logger
    
#define DEFINE_LOGGER(loggerId, loggerName) \
namespace { \
bool _KQP_LOG_CHECKER_ = kqp::LoggerInit::check(); \
log4cxx::LoggerPtr loggerId(log4cxx::Logger::getLogger(loggerName));  \
}
    
    // * Note * Use the if (false) construct to compile code; the code optimizer
    // is able to remove the corresponding code, so it does change
    // the speed 
    // * Note 2 * When NDEBUG is defined, we fully skip DEBUG messages
#ifndef NDEBUG
    

    /** Debug */
#define KQP_LOG_DEBUG(name,message) LOG4CXX_DEBUG(name, message)
    /** Debug and display the class name */
#define KQP_LOG_DEBUG_S(name,message) LOG4CXX_DEBUG(name, "[" << KQP_DEMANGLE(*this) << "/" << this << "] " << message)
    /** Assertion with message */
#define KQP_LOG_ASSERT(name,condition,message) { if (!(condition)) { LOG4CXX_ERROR(name, "Assert failed [" << KQP_STRING_IT(condition) << "] " << message); assert(false); } }

//! Throw an exception with a message (when NDEBUG is not defined, log a message and abort for backtrace access)
#define KQP_THROW_EXCEPTION(type, message) { KQP_LOG_ERROR(main_logger, "[Exception " << KQP_DEMANGLE(type()) << "] " << message);  abort(); }

#else // No DEBUG
    
//! Throw an exception with a message
#define KQP_THROW_EXCEPTION(type, message) BOOST_THROW_EXCEPTION(type() << errinfo_message(message))

    /** Debug */
#define KQP_LOG_DEBUG(name,message) { if (false) LOG4CXX_DEBUG(name, message) }
    /** Debug and display the class name */
#define KQP_LOG_DEBUG_S(name,message) { if (false) LOG4CXX_DEBUG(name, "[" << KQP_DEMANGLE(*this) << "/" << this << "/" << "] " << message); }
    /** Assertion with message */
#define KQP_LOG_ASSERT(name,condition,message) { if (false && !(condition)) { LOG4CXX_ERROR(name, "Assert failed [" << KQP_STRING_IT(condition) << "] " << message); assert(false); } }
    
#endif // ELSE
    
    
#define KQP_LOG_INFO(name,message) { LOG4CXX_INFO(name, message); }
#define KQP_LOG_WARN(name,message) { LOG4CXX_WARN(name, message); }
#define KQP_LOG_ERROR(name,message) { LOG4CXX_ERROR(name, message); }


#endif // ndef(NOLOGGING)
    
// With message versions
#define KQP_THROW_EXCEPTION_F(type, message, arguments) KQP_THROW_EXCEPTION(type, (boost::format(message) arguments).str())
#define KQP_LOG_DEBUG_F(name,message,args) KQP_LOG_DEBUG(name, boost::format(message) args)
#define KQP_LOG_INFO_F(name,message,args) KQP_LOG_INFO(name, boost::format(message) args)
#define KQP_LOG_WARN_F(name,message,args) KQP_LOG_WARN(name, boost::format(message) args)
#define KQP_LOG_ERROR_F(name,message,args) KQP_LOG_ERROR(name, boost::format(message) args)
  
} // NS kqp

#endif