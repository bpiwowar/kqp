INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

ADD_EXECUTABLE(example/example_1 example_1.cpp)
TARGET_LINK_LIBRARIES(example/example_1 kqp)

