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

# --- Build kqp

file(GLOB kqp_fmatrix_SRC "src/feature_matrix/*.?pp"  "src/feature_matrix.*")
file(GLOB kqp_SRC "src/*.hpp"  "src/*.cpp")
file(GLOB kqp_operator_builder_SRC "src/kernel_evd/*.?pp" "src/kernel_evd.*")

SOURCE_GROUP("Feature matrix" FILES ${kqp_fmatrix_SRC})
SOURCE_GROUP("Kernel EVD" FILES ${kqp_operator_builder_SRC})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})

add_library (kqp ${kqp_SRC} ${kqp_fmatrix_SRC} ${kqp_operator_builder_SRC} "src/Eigen")

TARGET_LINK_LIBRARIES(kqp ${LIBLOG4CXX_LIBRARY})

# --- Tests

ENABLE_TESTING()
add_subdirectory(tests)


# --- JNI Java library

add_subdirectory(jni)
