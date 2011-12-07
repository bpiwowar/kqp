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

# --- Add our own modules

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
INCLUDE_DIRECTORIES(${LIBLOG4CXX_INCLUDE_DIRS})

# --- Find Boost

FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# -- Find log4cxx

FIND_PACKAGE(log4cxx)
INCLUDE_DIRECTORIES(${LIBLOG4CXX_INCLUDE_DIRS})


# --- Build kqp

file(GLOB kqp_fmatrix_SRC "src/feature_matrix/*.hpp" "src/feature_matrix.hpp")
file(GLOB kqp_SRC "src/*.hpp"  "src/*.cpp")
file(GLOB kqp_operator_builder_SRC "src/kernel_evd/*.hpp" "src/kernel_evd.hpp")

SOURCE_GROUP(FeatureMatrix FILES ${kqp_fmatrix_SRC})
SOURCE_GROUP(OperatorBuilder FILES ${kqp_operator_builder_SRC})

set(PUBLIC_HEADER src/kernel_evd.h src/probabilities.h)

add_library (kqp ${kqp_SRC} ${kqp_fmatrix_SRC} ${kqp_operator_builder_SRC})

TARGET_LINK_LIBRARIES(kqp ${LIBLOG4CXX_LIBRARY})

# --- Tests

ENABLE_TESTING()
 add_subdirectory(tests)


# --- JNI Java library

if (JAVA_KQP) 
  FIND_PACKAGE(JNI REQUIRED)
  FIND_PACKAGE(SWIG REQUIRED)
  INCLUDE(${SWIG_USE_FILE})

  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory src)
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory sf/net/kqp)
  
  include_directories(${JNI_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})

  SET(CMAKE_SWIG_FLAGS -c++ -package sf.net.kqp)  
  set(CMAKE_SWIG_OUTDIR sf/net/kqp)

  SWIG_ADD_MODULE(kqp_jni java src/kqp.i)
#  SWIG_LINK_LIBRARIES(kqp_jni ${PYTHON_LIBRARIES})
endif()
