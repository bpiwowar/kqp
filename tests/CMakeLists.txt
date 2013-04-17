# Orthonormalization
FILE(GLOB kqp.test.evd-utils test_evd-utils.cpp ../src/subset.cpp)
ADD_EXECUTABLE(test_evd-utils ${kqp.test.evd-utils})
TARGET_LINK_LIBRARIES(test_evd-utils ${LIBKQP_LIBRARIES})

# Alt matrix
FILE(GLOB kqp.test.alt-matrix alt_matrix_tests.cpp)
ADD_EXECUTABLE(test_alt-matrix ${kqp.test.alt-matrix})
TARGET_LINK_LIBRARIES(test_alt-matrix ${LIBKQP_LIBRARIES})

# Feature matrix
FILE(GLOB kqp.test.fmatrix feature_matrix.cpp ../src/subset.cpp)
ADD_EXECUTABLE(test_fmatrix ${kqp.test.fmatrix})
TARGET_LINK_LIBRARIES(test_fmatrix ${LIBKQP_LIBRARIES})

# Rank-1 EVD update
FILE(GLOB kqp.test.evd-update evd-update.cpp)
ADD_EXECUTABLE(test_evd-update ${kqp.test.evd-update})
TARGET_LINK_LIBRARIES(test_evd-update ${LIBKQP_LIBRARIES})

# QP solver
FILE(GLOB kqp.test.qp-solver qp-solver.cpp )
ADD_EXECUTABLE(test_qp-solver ${kqp.test.qp-solver})
TARGET_LINK_LIBRARIES(test_qp-solver ${LIBKQP_LIBRARIES})

# Reduced set
FILE(GLOB kqp.test.reduced-set reduced-set_*.cpp)
ADD_EXECUTABLE(test_reduced-set ${kqp.test.reduced-set})
TARGET_LINK_LIBRARIES(test_reduced-set ${LIBKQP_LIBRARIES})

# Kernel EVD
FILE(GLOB kqp.test.kernel-evd kernel-evd*.[ch]pp)
ADD_EXECUTABLE(test_kernel-evd ${kqp.test.kernel-evd})
TARGET_LINK_LIBRARIES(test_kernel-evd ${LIBKQP_LIBRARIES})

# Divergence
FILE(GLOB kqp.test.divergence divergence_test.[ch]pp)
ADD_EXECUTABLE(test_divergence ${kqp.test.divergence})
TARGET_LINK_LIBRARIES(test_divergence ${LIBKQP_LIBRARIES})

# Projections
FILE(GLOB kqp.test.projection projection_test.[ch]pp)
ADD_EXECUTABLE(test_projection ${kqp.test.projection})
TARGET_LINK_LIBRARIES(test_projection ${LIBKQP_LIBRARIES})

# Probabilities
 FILE(GLOB kqp.test.probabilities probability_test.[ch]pp)
 ADD_EXECUTABLE(test_probabilities ${kqp.test.probabilities})
 TARGET_LINK_LIBRARIES(test_probabilities ${LIBKQP_LIBRARIES})
 


# All tests

ADD_DEPENDENCIES(run-tests DEPENDS test_alt-matrix test_evd-utils test_fmatrix test_evd-update test_projection test_probabilities test_qp-solver test_reduced-set test_kernel-evd test_divergence)

# --- Useful functions

FUNCTION(do_test id taskname)
   ADD_TEST(${id} "${EXECUTABLE_OUTPUT_PATH}/test_${taskname}"  ${ARGN})
   set_tests_properties(${id} PROPERTIES TIMEOUT 5)
ENDFUNCTION(do_test taskname)

FUNCTION(run_test task subtask)
   ADD_TEST("${task}/${subtask}" "${EXECUTABLE_OUTPUT_PATH}/test_${task}" ${subtask})
   set_tests_properties(${id} PROPERTIES TIMEOUT 5)
ENDFUNCTION(run_test taskname)

# --- Alt matrix
do_test(alt-matrix alt-matrix)

# --- EVD utils

do_test(evd-utils/orthonormalize/real-positive evd-utils orthonormalization/real-positive)
do_test(evd-utils/orthonormalize/real evd-utils orthonormalization/real)
do_test(evd-utils/orthonormalize/complex evd-utils orthonormalization/complex)

# --- Feature spaces
FOREACH(t dense sparse sparse-dense)
do_test(fmatrix_${t} fmatrix ${t})
ENDFOREACH()

# --- EVD update tests

do_test(evd-update evd-update simple)
do_test(evd-update-c evd-update complex)

# --- QP solver
do_test(qp-kkt-simple qp-solver "kkt-solver-simple")
do_test(qp-kkt-diagonal-g qp-solver "kkt-solver-diagonal-g")
do_test(qp-kkt-diagonal-d qp-solver "kkt-solver-diagonal-d")
do_test(qp-kkt-random qp-solver "kkt-solver-random")

do_test(kqp-qp-simple qp-solver "simple")
do_test(kqp-qp-random qp-solver "random")

do_test(kqp-qp-simple-nu qp-solver "simple/nu")
do_test(kqp-qp-random-nu qp-solver "random/nu")

# --- Reduced set methods

run_test(reduced-set unused)
run_test(reduced-set null-space)
run_test(reduced-set qp/exact)
run_test(reduced-set qp/approximate)

# --- Exact Kernel EVD

FUNCTION(do_kevd_test id builder)
    FOREACH(task "rank-1-once" "rank-1" "rank-n-once" "rank-n-twice" "rank-n")
        do_test("${id}/${task}" "kernel-evd"  "${builder}" "${task}" ${ARGN})
    ENDFOREACH(task)
ENDFUNCTION(do_kevd_test id taskname)

do_kevd_test(kernel-evd/direct direct)
do_kevd_test(kernel-evd/accumulator accumulator)
do_kevd_test(kernel-evd/accumulator-no-lc  accumulator-no-lc)
do_kevd_test(kernel-evd/incremental incremental)
do_kevd_test(kernel-evd/divide-and-conquer divide-and-conquer)

# --- Approximate Kernel EVD


# --- Quantum Probabilities

do_test(projection_simple projection simple)
do_test(projection_normal projection normal)

do_test(probabilities/orth probabilities orthonormalization)

# --- Quantum Probabilities (divergence)

do_test(divergence_simpleTest divergence simple)
do_test(divergence_fullTest divergence full)
do_test(divergence_simpleEpsilonTest divergence simpleEpsilon)
do_test(divergence_zeroTest divergence zero)

