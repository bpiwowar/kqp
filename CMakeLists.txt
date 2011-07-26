cmake_minimum_required (VERSION 2.6)
project (kqp) 

OPTION(JAVA_KQP "Build kqp JNI library" ON)

# --- Main Library
FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

set(PUBLIC_HEADER src/kernel_evd.h src/probabilities.h)
add_library (kqp src/fastRankOneUpdate.cpp src/kernel_evd.cpp src/kqp.cpp src/probabilities.cpp) 


# --- Galahad

# add_subdirectory(galahad)

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
