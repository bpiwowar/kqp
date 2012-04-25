
// Decomposition

%include "kqp/decomposition.hpp"
%template(Decomposition@FNAME@) kqp::Decomposition< @FTYPE@ >;

// Decomposition cleaner

%shared_ptr(kqp::Cleaner< @FTYPE@ >);
%shared_ptr(kqp::StandardCleaner< @FTYPE@ >);

%template(Cleaner@FNAME@) kqp::Cleaner< @FTYPE@ >;
%template(StandardCleaner@FNAME@) kqp::StandardCleaner< @FTYPE@ >;

// ---- Kernel EVD

%shared_ptr(kqp::KernelEVD< @FTYPE@ >);
%shared_ptr(kqp::DenseDirectBuilder< @STYPE@ >);
%shared_ptr(kqp::AccumulatorKernelEVD< @FTYPE@, true >)
%shared_ptr(kqp::DivideAndConquerBuilder< @FTYPE@ >)
%shared_ptr(kqp::IncrementalKernelEVD< @FTYPE@ >)

%include "kqp/kernel_evd.hpp"
%template(KEVD@FNAME@) kqp::KernelEVD< @FTYPE@ >;

#ifndef KEVDDIRECT@SNAME@
#define KEVDDIRECT@SNAME@
%include "kqp/kernel_evd/dense_direct.hpp"
%template(KEVDDirect@SNAME@) kqp::DenseDirectBuilder< @STYPE@ >;
#endif

%include "kqp/kernel_evd/accumulator.hpp"
%template(KEVDAccumulator@FNAME@) kqp::AccumulatorKernelEVD< @FTYPE@, true >;

%include "kqp/kernel_evd/incremental.hpp"
%template(KEVDIncremental@FNAME@) kqp::IncrementalKernelEVD< @FTYPE@ >;

%include "kqp/kernel_evd/divide_and_conquer.hpp"
%template(KEVDDivideAndConquer@FNAME@) kqp::DivideAndConquerBuilder< @FTYPE@ >;
