if (JAVA_KQP OR JAVA_PYTHON) 

  FIND_PACKAGE(SWIG REQUIRED)
  
  INCLUDE(${SWIG_USE_FILE})
  
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
  SET_SOURCE_FILES_PROPERTIES(kqp.i PROPERTIES CPLUSPLUS ON)
  
  if (JAVA_KQP)
      FIND_PACKAGE(JNI REQUIRED)
      INCLUDE_DIRECTORIES(${JNI_INCLUDE_DIRS})
      SET(JAVA_KQP_PACKAGE net.bpiwowar.kqp)
      STRING(REGEX REPLACE "\\." / JAVA_KQP_PACKAGE_PATH "${JAVA_KQP_PACKAGE}")
      
      SET(JAVA_KQP_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/src/main/java/${JAVA_KQP_PACKAGE_PATH})

      # FIXME: NAR_AOL should be set automatically
      MESSAGE(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
      
      if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
          SET(NAR_AOL "x86_64-MacOSX-g++")          
      else()
          MESSAGE(FATAL_ERROR " [JNI] Could not detect the NAR architecture")
      endif()
      
      SET(JAVA_KQP_NARDIR ${CMAKE_CURRENT_BINARY_DIR}/target/nar/${CMAKE_PROJECT_NAME}-${libkqp_VERSION}-${NAR_AOL}-jni/lib/)
      execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${JAVA_KQP_OUTDIR}")
      execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${JAVA_KQP_NARDIR}")
      SET(CMAKE_SWIG_FLAGS -c++ -package ${JAVA_KQP_PACKAGE})  
      set(CMAKE_SWIG_OUTDIR "${JAVA_KQP_OUTDIR}")
      
      SWIG_ADD_MODULE(kqp_jni java kqp.i)
      SWIG_LINK_LIBRARIES(kqp_jni kqp)
      
      # Build archives using NAR
      
      CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/pom.xml.in ${CMAKE_CURRENT_BINARY_DIR}/pom.xml)
      
      GET_TARGET_PROPERTY(KQP_JNI_LOCATION kqp_jni LOCATION)
      
      ADD_CUSTOM_TARGET(kqp_jni_maven 
          COMMAND ${CMAKE_COMMAND} -E copy ${KQP_JNI_LOCATION} "${JAVA_KQP_NARDIR}"
          COMMAND mvn -q -f ${CMAKE_CURRENT_BINARY_DIR}/pom.xml package install
          DEPENDS kqp_jni)
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
