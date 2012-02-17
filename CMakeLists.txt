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
project (kqp) 

# --- Version number

SET (kqp_VERSION "0.0.1")

STRING (REGEX MATCHALL "[0-9]" libkqp_VERSION_PARTS "${libkqp_VERSION}")

LIST (GET libkqp_VERSION_PARTS 0 libkqp_VERSION_MAJOR)
LIST (GET libkqp_VERSION_PARTS 1 libkqp_VERSION_MINOR)
LIST (GET libkqp_VERSION_PARTS 2 libkqp_VERSION_PATCH)

SET (libkqp_SOVERSION "${libkqp_VERSION_MAJOR}.${libkqp_VERSION_MINOR}")


# --- Compilation flags

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET (EXTRA_C_FLAGS "-Wno-deprecated -Wall -W -Wextra -fPIC")
  SET (EXTRA_C_FLAGS_RELEASE "-O3")
  SET (EXTRA_C_FLAGS_DEBUG "-O3 -g")
ENDIF()

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

file(GLOB kqp_fmatrix_SRC "src/feature_matrix/*.?pp"  "src/feature_matrix.*")
file(GLOB kqp_SRC "src/*.hpp"  "src/*.cpp")
file(GLOB kqp_operator_builder_SRC "src/kernel_evd/*.?pp" "src/kernel_evd.*")
file(GLOB kqp_reduced_set_SRC "src/reduced_set/*.?pp" "src/reduced_set.*")

SOURCE_GROUP("Feature matrix" FILES ${kqp_fmatrix_SRC})
SOURCE_GROUP("Kernel EVD" FILES ${kqp_operator_builder_SRC})
SOURCE_GROUP("Reduced Set" FILES ${kqp_reduced_set_SRC})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})

ADD_LIBRARY (kqp ${kqp_SRC} ${kqp_fmatrix_SRC} ${kqp_operator_builder_ScRC} ${kqp_reduced_set_SRC} "src/Eigen")

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

# --- Benchmark

add_subdirectory(benchmark)

# --- Tests

ENABLE_TESTING()
add_subdirectory(tests)


# --- JNI Java library

add_subdirectory(jni)
