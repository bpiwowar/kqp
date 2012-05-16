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

if (JAVA_KQP OR JAVA_PYTHON) 

  # --- Set important variables
  FIND_PACKAGE(SWIG REQUIRED)
  SET(KQP_SWIG_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
  
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -Wno-error=unused-but-set-variable -Wno-error=maybe-uninitialized")
  
  INCLUDE(${SWIG_USE_FILE})
  
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${KQP_SWIG_GENERATED})
  SET_SOURCE_FILES_PROPERTIES(kqp.i PROPERTIES CPLUSPLUS ON)
  
  FILE(GLOB KQP_SWIG_DEPS */*.i *.i)
  
  # --- Command that generates the kqp_all.i (that depends on all the *.i files within the swig directory)
  SET(PRE_SWIG_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/CMakePreSWIG.txt")
  add_custom_command(
    OUTPUT ${KQP_SWIG_GENERATED}/kqp_all.i
    COMMAND ${CMAKE_COMMAND} -D KQP_SWIG_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -D KQP_SWIG_GENERATED="${KQP_SWIG_GENERATED}" -P "${PRE_SWIG_SCRIPT}" SOURCES ${KQP_SWIG_DEPS} 
    DEPENDS ${KQP_SWIG_DEPS} ${PRE_SWIG_SCRIPT})
  

  # --- Java JNI module  
  if (JAVA_KQP)
      FIND_PACKAGE(JNI REQUIRED)
      INCLUDE_DIRECTORIES(${JNI_INCLUDE_DIRS})
      SET(JAVA_KQP_PACKAGE net.bpiwowar.kqp)
      STRING(REGEX REPLACE "\\." / JAVA_KQP_PACKAGE_PATH "${JAVA_KQP_PACKAGE}")
      
      SET(JAVA_KQP_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/src/main/java/${JAVA_KQP_PACKAGE_PATH})
      SET(JAVA_KQP_TESTDIR ${CMAKE_CURRENT_BINARY_DIR}/test/main/java/${JAVA_KQP_PACKAGE_PATH})

      # Detect the architecture
      if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
        # 32 bits
        SET(NAR_AOL "x86")
      elseif(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        SET(NAR_AOL "x86_64")
      else()
        MESSAGE(FATAL_ERROR " [JNI] Could not detect the compiler target architecture")
      endif()
      
      # Detection of the sytem
      if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
          SET(NAR_AOL "${NAR_AOL}-MacOSX")          
      elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
          SET(NAR_AOL "${NAR_AOL}-Linux")
      elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
        SET(NAR_AOL "${NAR_AOL}-Windows")
      else()
          MESSAGE(FATAL_ERROR " [JNI] Could not detect the operating system")
      endif()
      
      MESSAGE(STATUS "Detected NAR-AOL: ${NAR_AOL} [${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_SIZEOF_VOID_P}/${CMAKE_SYSTEM_NAME}]")
      
      SET(JAVA_KQP_NARDIR ${CMAKE_CURRENT_BINARY_DIR}/target/nar/${CMAKE_PROJECT_NAME}-${libkqp_VERSION}-${NAR_AOL}-jni/lib/)
      file(MAKE_DIRECTORY "${JAVA_KQP_OUTDIR}")
      file(MAKE_DIRECTORY "${JAVA_KQP_NARDIR}")
      file(MAKE_DIRECTORY "${JAVA_KQP_TESTDIR}")
      
      SET(CMAKE_SWIG_FLAGS -c++ -package ${JAVA_KQP_PACKAGE})  
      set(CMAKE_SWIG_OUTDIR "${JAVA_KQP_OUTDIR}")
      
      SET(SWIG_MODULE_kqp_jni_EXTRA_DEPS ${KQP_SWIG_GENERATED}/kqp_all.i)
      SWIG_ADD_MODULE(kqp_jni java kqp.i)
      SWIG_LINK_LIBRARIES(kqp_jni kqp)
      
      
      # --- Build archives using NAR (target kqp_jni_maven)
      
      find_program(MAVEN_PATH mvn DOC "Path to maven executable" PATH ${MAVEN_PATH})
      
      if(${MAVEN_PATH} EQUAL "")
        MESSAGE(WARNING "Did not find maven executable (use MAVEN_PATH to set it manually)")
      else()
        MESSAGE(STATUS "Found maven: ${MAVEN_PATH}")
      
        CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/pom.xml.in ${CMAKE_CURRENT_BINARY_DIR}/pom.xml)
        FILE(GLOB JAVA_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/../examples/*.java)
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${JAVA_EXAMPLES} "${JAVA_KQP_TESTDIR}")
        GET_TARGET_PROPERTY(KQP_JNI_LOCATION kqp_jni LOCATION)
      
        ADD_CUSTOM_TARGET(kqp_jni_maven 
            COMMAND ${CMAKE_COMMAND} -E copy ${KQP_JNI_LOCATION} "${JAVA_KQP_NARDIR}"
            COMMAND ${MAVEN_PATH} -f ${CMAKE_CURRENT_BINARY_DIR}/pom.xml package install
            DEPENDS kqp_jni)

      endif()
      
          
          
  endif()
  
  # --- Python support
  
  if (JAVA_PYTHON)
      INCLUDE_DIRECTORIES()
      FIND_PACKAGE(PythonLibs REQUIRED)
      INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
      SET(CMAKE_SWIG_FLAGS "")
      set(CMAKE_SWIG_OUTDIR ".")
      SWIG_ADD_MODULE(kqp_python python kqp.i)
      SWIG_LINK_LIBRARIES(kqp_python kqp ${PYTHON_LIBRARIES}) 
      SET(SWIG_MODULE_kqp_python_EXTRA_DEPS ${KQP_SWIG_DEPS} pre_swig)
  endif()
  
endif()
