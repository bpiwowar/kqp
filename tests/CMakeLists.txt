
INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)

file(GLOB kqp_test_SRC   "*.h"   "*.cpp")
ADD_EXECUTABLE(kqp_test ${kqp_test_SRC})

TARGET_LINK_LIBRARIES(kqp_test kqp)

add_test(evd-update ${EXECUTABLE_OUTPUT_PATH}/kqp_test "evd-update" "simple")
add_test(evd-update-c ${EXECUTABLE_OUTPUT_PATH}/kqp_test "evd-update" "complex")

add_test(evd-update-c ${EXECUTABLE_OUTPUT_PATH}/kqp_test "kqp-qp-solver" "kkt-solver-simple")
add_test(evd-update-c ${EXECUTABLE_OUTPUT_PATH}/kqp_test "kqp-qp-solver" "kkt-solver-diagonal-g")
add_test(evd-update-c ${EXECUTABLE_OUTPUT_PATH}/kqp_test "kqp-qp-solver" "kkt-solver-diagonal-d")
add_test(evd-update-c ${EXECUTABLE_OUTPUT_PATH}/kqp_test "kqp-qp-solver" "kkt-solver-random")
