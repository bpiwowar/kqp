
#ifndef __KQP_H__
#define __KQP_H__

#include <boost/exception/errinfo_at_line.hpp>

#include <boost/exception/info.hpp>
#include <boost/exception/exception.hpp>
#include <string>
#include <complex>

namespace kqp {
    
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

}

#endif