# This file is part of the Kernel Quantum Probability library (KQP).
# 
# KQP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
 
# KQP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
 
# You should have received a copy of the GNU General Public License
# along with KQP.  If not, see <http://www.gnu.org/licenses/>.
 
 
cmake_minimum_required (VERSION 2.6)

# --- Main options

# Project name and version
PROJECT (kqp) 
SET (libkqp_VERSION "0.0.1")

# Build by default a release version of the library
SET(CMAKE_BUILD_TYPE "Release")

# --- Version number

STRING (REGEX MATCHALL "[0-9]" libkqp_VERSION_PARTS "${libkqp_VERSION}")

LIST (GET libkqp_VERSION_PARTS 0 libkqp_VERSION_MAJOR)
LIST (GET libkqp_VERSION_PARTS 1 libkqp_VERSION_MINOR)
LIST (GET libkqp_VERSION_PARTS 2 libkqp_VERSION_PATCH)

SET (libkqp_SOVERSION "${libkqp_VERSION_MAJOR}.${libkqp_VERSION_MINOR}")


# --- Compilation flags

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET (EXTRA_C_FLAGS "--std=c++0x -Wno-deprecated -Wall -W -Wextra -fPIC")
  SET (EXTRA_C_FLAGS_RELEASE "-O3")
  SET (EXTRA_C_FLAGS_DEBUG "-O3 -g")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

IF (CMAKE_GENERATOR MATCHES Xcode)
	set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++0x")
ENDIF(CMAKE_GENERATOR MATCHES Xcode)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")


# --- Options

OPTION(JAVA_KQP "Build kqp JNI library" ON)
OPTION(OPEN_MP "Use Open MP (if available)" OFF)

# --- Add our own modules

SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
INCLUDE_DIRECTORIES(${LIBLOG4CXX_INCLUDE_DIRS})

# --- Find Boost

FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# -- Find log4cxx

FIND_PACKAGE(log4cxx)
INCLUDE_DIRECTORIES(${LIBLOG4CXX_INCLUDE_DIRS})


# --- Check for openmp

IF(OPEN_MP MATCHES "ON")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	MESSAGE(STATUS "Using OpenMP")
ENDIF(OPEN_MP MATCHES "ON")

# --- Build kqp library


file(GLOB kqp_kqp_H  "include/kqp/*.hpp")
file(GLOB kqp_fmatrix_H "include/kqp/feature_matrix/*.hpp"  "include/kqp/feature_matrix.hpp")
file(GLOB kqp_kernel_evd_H "include/kqp/kernel_evd/*.hpp" "include/kqp/kernel_evd.hpp")
file(GLOB kqp_reduced_set_H "include/kqp/reduced_set/*.hpp" "include/kqp/reduced_set.hpp")

file(GLOB kqp_kqp_CPP  "src/*.cpp")
file(GLOB kqp_fmatrix_CPP "src/feature_matrix/*.cpp"  "src/feature_matrix.cpp")
file(GLOB kqp_kernel_evd_CPP "src/kernel_evd/*.cpp" "src/kernel_evd.cpp")
file(GLOB kqp_reduced_set_CPP "src/reduced_set/*.cpp" "src/reduced_set.cpp")

FOREACH(name kqp fmatrix kernel_evd reduced_set)
	SET(kqp_${name}_SRC ${kqp_${name}_H} ${kqp_${name}_CPP})
ENDFOREACH(name)

SOURCE_GROUP("Feature matrix" FILES ${kqp_fmatrix_SRC})
SOURCE_GROUP("Kernel EVD" FILES ${kqp_kernel_evd_SRC})
SOURCE_GROUP("Reduced Set" FILES ${kqp_reduced_set_SRC})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

ADD_LIBRARY (kqp ${kqp_kqp_SRC} ${kqp_fmatrix_SRC} ${kqp_kernel_evd_SRC} ${kqp_reduced_set_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen")

TARGET_LINK_LIBRARIES(kqp ${LIBLOG4CXX_LIBRARY})

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
add_subdirectory(tests)


# --- Subprojects: examples, benchmark 

# Examples
add_subdirectory(examples)

# Benchmark
add_subdirectory(benchmark)

#  Wrappers
add_subdirectory(swig)
