# - Try to find LibXml2
# Once done this will define
#  LIBLOG4CXX_FOUND - System has LibXml2
#  LIBLOG4CXX_INCLUDE_DIRS - The LibXml2 include directories
#  LIBLOG4CXX_LIBRARIES - The libraries needed to use LibXml2
#  LIBLOG4CXX_DEFINITIONS - Compiler switches required for using LibXml2

find_package(PkgConfig)
pkg_check_modules(PC_LIBLOG4CXX QUIET libxml-2.0)
set(LIBLOG4CXX_DEFINITIONS ${PC_LIBLOG4CXX_CFLAGS_OTHER})

find_path(LIBLOG4CXX_INCLUDE_DIR log4cxx/log4cxx.h
          HINTS ${PC_LIBLOG4CXX_INCLUDEDIR} ${PC_LIBLOG4CXX_INCLUDE_DIRS}
          PATH_SUFFIXES liblog4cxx )

find_library(LIBLOG4CXX_LIBRARY NAMES log4cxx liblog4cxx
             HINTS ${PC_LIBLOG4CXX_LIBDIR} ${PC_LIBLOG4CXX_LIBRARY_DIRS} )

set(LIBLOG4CXX_LIBRARIES ${LIBLOG4CXX_LIBRARY} )
set(LIBLOG4CXX_INCLUDE_DIRS ${LIBLOG4CXX_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBLOG4CXX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Liblog4cxx  DEFAULT_MSG
                                  LIBLOG4CXX_LIBRARY LIBLOG4CXX_INCLUDE_DIR)

mark_as_advanced(LIBLOG4CXX_INCLUDE_DIR LIBLOG4CXX_LIBRARY )
