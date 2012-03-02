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

  SET_SOURCE_FILES_PROPERTIES(kqp.i PROPERTIES CPLUSPLUS ON)
  SWIG_ADD_MODULE(kqp_jni java kqp.i)
  TARGET_LINK_LIBRARIES(kqp_jni kqp)
endif()
