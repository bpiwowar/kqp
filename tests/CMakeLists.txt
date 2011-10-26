
INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)

file(GLOB kqp_test_SRC   "*.h"   "*.cpp")
ADD_EXECUTABLE(kqp_test ${kqp_test_SRC})

TARGET_LINK_LIBRARIES(kqp_test kqp)

add_test(test1 ${EXECUTABLE_OUTPUT_PATH}/kqp_test "hello_world")
