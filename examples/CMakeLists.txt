execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory examples/examples)


INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

ADD_EXECUTABLE(examples/example_1 example_1.cpp)
TARGET_LINK_LIBRARIES(examples/example_1 kqp)

ADD_TEST(examples/example_1 examples/example_1)
