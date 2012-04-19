if (JAVA_KQP OR JAVA_PYTHON) 

  FIND_PACKAGE(SWIG REQUIRED)
  
  INCLUDE(${SWIG_USE_FILE})
  
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
  SET_SOURCE_FILES_PROPERTIES(kqp.i PROPERTIES CPLUSPLUS ON)
  
  if (JAVA_KQP)
      FIND_PACKAGE(JNI REQUIRED)
      INCLUDE_DIRECTORIES(${JNI_INCLUDE_DIRS})
      execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory src)
      execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory sf/net/kqp)
      SET(CMAKE_SWIG_FLAGS -c++ -package sf.net.kqp)  
      set(CMAKE_SWIG_OUTDIR sf/net/kqp)
      SWIG_ADD_MODULE(kqp_jni java kqp.i)
      SWIG_LINK_LIBRARIES(kqp_jni kqp)
  endif()
  
  if (JAVA_PYTHON)
      INCLUDE_DIRECTORIES()
      FIND_PACKAGE(PythonLibs REQUIRED)
      INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
      SET(CMAKE_SWIG_FLAGS "")
      set(CMAKE_SWIG_OUTDIR ".")
      SWIG_ADD_MODULE(kqp_python python kqp.i)
      SWIG_LINK_LIBRARIES(kqp_python kqp ${PYTHON_LIBRARIES}) 
  endif()
  
endif()
