
// Decomposition

%include "kqp/decomposition.hpp"
%template(Decomposition@SNAME@) kqp::Decomposition< @STYPE@ >;

// Decomposition cleaner

%shared_ptr(kqp::Cleaner< @STYPE@ >);
%shared_ptr(kqp::StandardCleaner< @STYPE@ >);

%template(Cleaner@SNAME@) kqp::Cleaner< @STYPE@ >;
%template(StandardCleaner@SNAME@) kqp::StandardCleaner< @STYPE@ >;

// ---- Kernel EVD

%shared_ptr(kqp::KernelEVD< @STYPE@ >);
%shared_ptr(kqp::DenseDirectBuilder< @STYPE@ >);
%shared_ptr(kqp::AccumulatorKernelEVD< @STYPE@, true >)
%shared_ptr(kqp::DivideAndConquerBuilder< @STYPE@ >)
%shared_ptr(kqp::IncrementalKernelEVD< @STYPE@ >)

%include "kqp/kernel_evd.hpp"
%template(KEVD@SNAME@) kqp::KernelEVD< @STYPE@ >;

#ifndef KEVDDIRECT@SNAME@
#define KEVDDIRECT@SNAME@
%include "kqp/kernel_evd/dense_direct.hpp"
%template(KEVDDirect@SNAME@) kqp::DenseDirectBuilder< @STYPE@ >;
#endif

%include "kqp/kernel_evd/accumulator.hpp"
%template(KEVDAccumulator@SNAME@) kqp::AccumulatorKernelEVD< @STYPE@, true >;

%include "kqp/kernel_evd/incremental.hpp"
%template(KEVDIncremental@SNAME@) kqp::IncrementalKernelEVD< @STYPE@ >;

%include "kqp/kernel_evd/divide_and_conquer.hpp"
%template(KEVDDivideAndConquer@SNAME@) kqp::DivideAndConquerBuilder< @STYPE@ >;
