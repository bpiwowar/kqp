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

#ifndef __KQP_EXCEPTIONS_H__
#define __KQP_EXCEPTIONS_H__

#include <boost/throw_exception.hpp>
#include <boost/exception/errinfo_at_line.hpp>
#include <boost/exception/info.hpp>
#include <boost/exception/exception.hpp>
#include <boost/format.hpp>

// Stack information
#include <execinfo.h>
#include <dlfcn.h>

namespace kqp {
    /** Message information */    
    typedef boost::error_info<struct errinfo_message_,std::string> errinfo_message;

    /** Stack information */
    template <typename T> struct trace_info_struct_ {
        int count;
        T **pointers;
        T *_pointers[100];
        trace_info_struct_() {
         pointers = _pointers + 1;
         count = backtrace( pointers, 100 ) - 1;       
        }
    };

    typedef trace_info_struct_<void> trace_info_struct;
    typedef boost::error_info<struct tag_stack, trace_info_struct> stack_info;
    inline stack_info trace_info() { return stack_info(trace_info_struct()); }

    template<typename T>
    inline std::ostream  & operator<<( std::ostream  & x,  const trace_info_struct_<T>& trace ) {
        char **stack_syms(backtrace_symbols( trace.pointers, trace.count ));
        Dl_info info;
        x << "[" << trace.count << "]\n";
        for ( int i = 0 ; i < trace.count ; ++i )
        {
            dladdr(trace.pointers[i], &info);
            x << stack_syms[i] 
                << "\t" << (boost::format("%x") %  info.dli_fbase)
                << (boost::format(" start+%x ") % ((char*)trace.pointers[i]-(char*)info.dli_fbase))
                << info.dli_sname << "\n";
        }
        std::free( stack_syms );
        return x;
    }

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

    /** Illegal argument */
    class bad_cast_exception : public virtual exception {};

    #undef BOOST_THROW_EXCEPTION
    #define BOOST_THROW_EXCEPTION(x) kqp::throw_exception_(x,BOOST_CURRENT_FUNCTION,__FILE__,__LINE__)

    // Throw an exception
    template <class E>
    BOOST_ATTRIBUTE_NORETURN
    void
    throw_exception_( E const & x, char const * current_function, char const * file, int line )
    {
        using namespace boost;
        using namespace boost::exception_detail;
        throw_exception(
            set_info(
                set_info(
                    set_info(
                        set_info(
                            enable_error_info(x),
                            kqp::trace_info()),
                        throw_function(current_function)),
                    throw_file(file)),
                throw_line(line)));
    }

}

#endif