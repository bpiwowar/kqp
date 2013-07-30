# This file is part of the Kernel Quantum Probability library (KQP).
# 
# KQP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#  
# KQP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#  
# You should have received a copy of the GNU General Public License
# along with KQP.  If not, see <http://www.gnu.org/licenses/>.
 

# --- Basic information about the project
 
cmake_minimum_required (VERSION 2.6)
INCLUDE (CheckCXXSourceRuns)

PROJECT (kqp) 
SET (libkqp_VERSION "1.0.0-SNAPSHOT")
SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# --- Main options

OPTION(JAVA_KQP                 "Build Java wrapper"            ON)
OPTION(PYTHON_KQP               "Build Python wrapper"          ON)
OPTION(OPEN_MP                  "Use Open MP"                   ON)

# --- Get version number from libkqp_VERSION

STRING (REGEX MATCHALL "[0-9]" libkqp_VERSION_PARTS "${libkqp_VERSION}")

LIST (GET libkqp_VERSION_PARTS 0 libkqp_VERSION_MAJOR)
LIST (GET libkqp_VERSION_PARTS 1 libkqp_VERSION_MINOR)
LIST (GET libkqp_VERSION_PARTS 2 libkqp_VERSION_PATCH)

SET (libkqp_SOVERSION "${libkqp_VERSION_MAJOR}.${libkqp_VERSION_MINOR}")

# --- Output some information

if (NOT(CMAKE_BUILD_TYPE))
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE(STATUS "Creating project KQP ${libkqp_VERSION} with build type ${CMAKE_BUILD_TYPE}")

# --- Compilation flags


# - Gnu CC
IF(CMAKE_COMPILER_IS_GNUCXX)
  MESSAGE(STATUS "GCC version ${CMAKE_C_COMPILER_VERSION}")
    
  # Get the major and minor version
  IF(CMAKE_C_COMPILER_VERSION VERSION_GREATER "4.6.999999")
	SET(CXX_STD "c++11")
  ELSE()
	SET(CXX_STD "c++0x")
  ENDIF()
  
  SET (EXTRA_C_FLAGS "${EXTRA_C_FLAGS} --std=${CXX_STD} -Wno-deprecated -Wall -W -Wextra -fPIC -Werror ")
  SET (EXTRA_C_FLAGS_RELEASE "-O3 ${EXTRA_C_FLAGS}")
  SET (EXTRA_C_FLAGS_DEBUG "-O0 -g ${EXTRA_C_FLAGS}")

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# - Xcode
IF (CMAKE_GENERATOR MATCHES Xcode)
	SET_PROPERTY(GLOBAL PROPERTY XCODE_BUILD_VARIANTS "Debug Release")
	SET(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++0x")
	SET(CMAKE_XCODE_GCC_OPTIMIZATION_LEVEL "0")
    # SET(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
ENDIF(CMAKE_GENERATOR MATCHES Xcode)



# --- Add our own modules

INCLUDE_DIRECTORIES(${LIBLOG4CXX_INCLUDE_DIRS})

# --- Find Boost

FIND_PACKAGE(Boost COMPONENTS serialization)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# -- Find log4cxx

FIND_PACKAGE(log4cxx)
INCLUDE_DIRECTORIES(${LIBLOG4CXX_INCLUDE_DIRS})

# --- Check for openmp

IF(OPEN_MP MATCHES "ON")
  MESSAGE(STATUS "Testing OPEN MP support")
  SET(OLD_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  try_compile(COMPILES_OPENMP "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/openmp/dummy.cpp" 
    CMAKE_FLAGS "-fopenmp" OUTPUT_VARIABLE comp)
  IF(NOT(COMPILES_OPENMP))
    MESSAGE(STATUS "No OpenMP support detected")
    # MESSAGE(STATUS "${comp}")
	  SET(CMAKE_CXX_FLAGS "${OLD_CXX_FLAGS}")
    SET(OPEN_MP "OFF")
  ELSE()
    MESSAGE(STATUS "OpenMP enabled")
  ENDIF()
ENDIF()

IF(OPEN_MP MATCHES "ON")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	MESSAGE(STATUS "Using OpenMP")
ELSEIF(OPEN_MP MATCHES "OFF")
	MESSAGE(STATUS "Not using OpenMP")
ELSE()
    
ENDIF()

# --- Build kqp library


file(GLOB kqp_kqp_H  "include/kqp/*.hpp")
file(GLOB kqp_fmatrix_H "include/kqp/feature_matrix/*.hpp"  "include/kqp/feature_matrix.hpp")
file(GLOB kqp_kernel_evd_H "include/kqp/kernel_evd/*.hpp" "include/kqp/kernel_evd.hpp")
file(GLOB kqp_cleaning_H "include/kqp/cleaning/*.hpp" "include/kqp/cleaning.hpp")

SET(kqp_H "")
 
FOREACH(name kqp fmatrix cleaning kernel_evd)
	LIST(APPEND kqp_SRC "${kqp_${name}_H}")
ENDFOREACH(name)

SOURCE_GROUP("Spaces" FILES ${kqp_fmatrix_H})
SOURCE_GROUP("Kernel EVD" FILES ${kqp_kernel_evd_H})
SOURCE_GROUP("Cleaning" FILES ${kqp_cleaning_H})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

SET(LIBKQP_LIBRARIES ${LIBLOG4CXX_LIBRARY} ${Boost_LIBRARIES})

# --- Doxygen

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile ${WORKING_DIRECTORY} ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

# --- Tests

ENABLE_TESTING()

ADD_CUSTOM_TARGET(run-tests COMMAND ${CMAKE_CTEST_COMMAND})


# Tests
add_subdirectory(tests)

# Benchmark
add_subdirectory(benchmark)

#  Wrappers
add_subdirectory(swig)

# Examples
add_subdirectory(examples)

# --- Installation

INSTALL(DIRECTORY include/kqp  DESTINATION include)
INSTALL(DIRECTORY include/Eigen  DESTINATION include/kqp)

