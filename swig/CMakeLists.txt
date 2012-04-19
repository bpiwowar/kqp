if (JAVA_KQP OR JAVA_PYTHON) 

  FIND_PACKAGE(SWIG REQUIRED)
  SET(KQP_SWIG_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
  
  INCLUDE(${SWIG_USE_FILE})
  
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${KQP_SWIG_GENERATED})
  SET_SOURCE_FILES_PROPERTIES(kqp.i PROPERTIES CPLUSPLUS ON)
  
  # --- Process files to substitude each type
  FILE(GLOB KQP_SWIG_EACH_FMATRIX for_each_fmatrix/*)
  FILE(GLOB KQP_SWIG_EACH_SCALAR for_each_scalar/*)
  
  FUNCTION(SPLIT2 IN A B)
      STRING (REGEX MATCHALL "[^/]+" SPLIT2_PARTS "${IN}")
      LIST (GET SPLIT2_PARTS 0 "${A}")
      LIST (GET SPLIT2_PARTS 1 "${B}")
  ENDFUNCTION()
  
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${KQP_SWIG_GENERATED})
  
  
  # Loop over all types
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/kqp_all.i" "// Generated file\n\n")
  FOREACH(SWIG_SCALAR "double/Double")
      STRING (REGEX MATCHALL "[^/]+" SWIG_PARTS "${SWIG_SCALAR}")
      LIST (GET SWIG_PARTS 0 RTYPE)
      LIST (GET SWIG_PARTS 1 RNAME)
      FOREACH(SWIG_COMPLEX "${RTYPE}/${RNAME}" "std::complex<${RTYPE}>/Complex${RNAME}")
          STRING (REGEX MATCHALL "[^/]+" SWIG_PARTS "${SWIG_COMPLEX}")
          LIST (GET SWIG_PARTS 0 STYPE)
          LIST (GET SWIG_PARTS 1 SNAME)
          MESSAGE(STATUS "${STYPE} ${SNAME} / ${RTYPE} ${RNAME}")
          
          FOREACH(KQP_SWIG_SCALAR ${KQP_SWIG_EACH_SCALAR})
             get_filename_component(KQP_SWIG_FILENAME "${KQP_SWIG_SCALAR}" NAME)
              MESSAGE(STATUS "Processing ${KQP_SWIG_SCALAR} into ${KQP_SWIG_GENERATED}/${SNAME}-${KQP_SWIG_FILENAME}")
              file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/generated/kqp_all.i" "%include \"${SNAME}-${KQP_SWIG_FILENAME}\"\n")
              configure_file(${KQP_SWIG_SCALAR} "${KQP_SWIG_GENERATED}/${SNAME}-${KQP_SWIG_FILENAME}")
          ENDFOREACH()
          
          FOREACH(SWIG_FMATRIX "kqp::DenseMatrix< ${STYPE} >/Dense${SNAME}")
              STRING (REGEX MATCHALL "[^/]+" SWIG_PARTS "${SWIG_FMATRIX}")
              LIST (GET SWIG_PARTS 0 FTYPE)
              LIST (GET SWIG_PARTS 1 FNAME)
              FOREACH(KQP_SWIG_ONE_FMATRIX ${KQP_SWIG_EACH_FMATRIX})
                 get_filename_component(KQP_SWIG_FILENAME "${KQP_SWIG_ONE_FMATRIX}" NAME)
                  MESSAGE(STATUS "Processing ${KQP_SWIG_ONE_FMATRIX} into ${KQP_SWIG_GENERATED}/${FNAME}-${KQP_SWIG_FILENAME}")
                  file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/generated/kqp_all.i" "%include \"${FNAME}-${KQP_SWIG_FILENAME}\"\n")
                  configure_file(${KQP_SWIG_ONE_FMATRIX} "${KQP_SWIG_GENERATED}/${FNAME}-${KQP_SWIG_FILENAME}")
              ENDFOREACH()
          ENDFOREACH()
      ENDFOREACH()
  ENDFOREACH()
  
  
  
  
  if (JAVA_KQP)
      FIND_PACKAGE(JNI REQUIRED)
      INCLUDE_DIRECTORIES(${JNI_INCLUDE_DIRS})
      SET(JAVA_KQP_PACKAGE net.bpiwowar.kqp)
      STRING(REGEX REPLACE "\\." / JAVA_KQP_PACKAGE_PATH "${JAVA_KQP_PACKAGE}")
      
      SET(JAVA_KQP_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/src/main/java/${JAVA_KQP_PACKAGE_PATH})

      # FIXME: NAR_AOL should be set automatically
      MESSAGE(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
      
      if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
          SET(NAR_AOL "x86_64-MacOSX")          
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
