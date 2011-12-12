

FILE(GLOB kqp_test_SRC   "*.hpp"   "*.cpp")

INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

ADD_EXECUTABLE(kqp_test ${kqp_test_SRC})

TARGET_LINK_LIBRARIES(kqp_test kqp)
TARGET_LINK_LIBRARIES(kqp_test boost_program_options)

add_custom_target(run-tests COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS kqp_test)

FUNCTION(do_test id taskname)
   ADD_TEST(${id} ${EXECUTABLE_OUTPUT_PATH}/kqp_test "--task" ${taskname} ${ARGN})
ENDFUNCTION(do_test taskname)

do_test(evd-update evd-update simple)
do_test(evd-update-c evd-update complex)

do_test(qp-kkt-simple "kqp-qp-solver" "kkt-solver-simple")
do_test(qp-kkt-diagonal-g "kqp-qp-solver" "kkt-solver-diagonal-g")
do_test(qp-kkt-diagonal-d "kqp-qp-solver" "kkt-solver-diagonal-d")
do_test(qp-kkt-random "kqp-qp-solver" "kkt-solver-random")

do_test(kqp-qp-simple "kqp-qp-solver" "simple")
do_test(kqp-qp-random "kqp-qp-solver" "random")

do_test(kernel-evd-direct kernel-evd direct-builder)
do_test(kernel-evd-accumulator kernel-evd accumulator)