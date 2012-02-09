

FILE(GLOB kqp_test_SRC   "*.hpp"   "*.cpp")

INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

ADD_EXECUTABLE(kqp_test ${kqp_test_SRC} ${kqp_test_kevd_SRC})

TARGET_LINK_LIBRARIES(kqp_test kqp)

add_custom_target(run-tests COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS kqp_test)

FUNCTION(do_test id taskname)
   ADD_TEST(${id} ${EXECUTABLE_OUTPUT_PATH}/kqp_test "${taskname}" ${ARGN})
ENDFUNCTION(do_test taskname)

# --- EVD update tests

do_test(evd-update evd-update simple)
do_test(evd-update-c evd-update complex)

# --- QP solver

do_test(qp-kkt-simple "kqp-qp-solver" "kkt-solver-simple")
do_test(qp-kkt-diagonal-g "kqp-qp-solver" "kkt-solver-diagonal-g")
do_test(qp-kkt-diagonal-d "kqp-qp-solver" "kkt-solver-diagonal-d")
do_test(qp-kkt-random "kqp-qp-solver" "kkt-solver-random")

do_test(kqp-qp-simple "kqp-qp-solver" "simple")
do_test(kqp-qp-random "kqp-qp-solver" "random")

# --- Reduced set methods

do_test(reduced-set_unused "reduced-set/unused")
do_test(reduced-set_null-space "reduced-set/null-space")
do_test(reduced-set_qp "reduced-set/qp")

# --- Exact Kernel EVD

FUNCTION(do_kevd_test id builder)
    FOREACH(task "rank-1-once" "rank-1" "rank-n-once" "rank-n-twice" "rank-n")
        do_test("${id}-${task}" "kernel-evd"  "${builder}" "${task}" ${ARGN})
    ENDFOREACH(task)
ENDFUNCTION(do_kevd_test id taskname)

do_kevd_test(kernel-evd-direct direct)
do_kevd_test(kernel-evd-accumulator accumulator)
do_kevd_test(kernel-evd-accumulator-no-lc  accumulator-no-lc)
do_kevd_test(kernel-evd-incremental incremental)


# --- Approximate Kernel EVD


# --- Quantum Probabilities
