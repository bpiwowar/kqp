

FILE(GLOB kqp_benchmark_SRC  "*.hpp"   "*.cpp")

INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

ADD_EXECUTABLE(kqp_benchmark ${kqp_benchmark_SRC} ${kqp_benchmark_kevd_SRC})

TARGET_LINK_LIBRARIES(kqp_benchmark kqp)

add_custom_target(run-benchmarks COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS kqp_benchmark)

FUNCTION(do_benchmark id taskname)
   ADD_TEST(${id} ${EXECUTABLE_OUTPUT_PATH}/kqp_benchmark "${taskname}" ${ARGN})
ENDFUNCTION(do_benchmark taskname)
