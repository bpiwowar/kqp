
// --- Decomposition

%include "kqp/decomposition.hpp"
%template(Decomposition@SNAME@) kqp::Decomposition< @STYPE@ >;

// --- Decomposition cleaner

%include "kqp/cleanup.hpp"
%shared_ptr(kqp::Cleaner< @STYPE@ >);
%shared_ptr(kqp::CleanerList< @STYPE@ >);
%shared_ptr(kqp::CleanerRank< @STYPE@ >);

%template(Cleaner@SNAME@) kqp::Cleaner< @STYPE@ >;
%template(CleanerList@SNAME@) kqp::CleanerList< @STYPE@ >;
%template(CleanerRank@SNAME@) kqp::CleanerRank< @STYPE@ >;

// ---- Kernel EVD

%shared_ptr(kqp::KernelEVD< @STYPE@ >);
%shared_ptr(kqp::DenseDirectBuilder< @STYPE@ >);
%shared_ptr(kqp::AccumulatorKernelEVD< @STYPE@, true >)
%shared_ptr(kqp::AccumulatorKernelEVD< @STYPE@, false >)
%shared_ptr(kqp::DivideAndConquerBuilder< @STYPE@ >)
%shared_ptr(kqp::IncrementalKernelEVD< @STYPE@ >)

%include "kqp/kernel_evd.hpp"
%template(KEVD@SNAME@) kqp::KernelEVD< @STYPE@ >;

%include "kqp/kernel_evd/dense_direct.hpp"
%template(KEVDDirect@SNAME@) kqp::DenseDirectBuilder< @STYPE@ >;

%include "kqp/kernel_evd/accumulator.hpp"
%template(KEVDLCAccumulator@SNAME@) kqp::AccumulatorKernelEVD< @STYPE@, true >;

%include "kqp/kernel_evd/accumulator.hpp"
%template(KEVDAccumulator@SNAME@) kqp::AccumulatorKernelEVD< @STYPE@, false >;

%include "kqp/kernel_evd/incremental.hpp"
%template(KEVDIncremental@SNAME@) kqp::IncrementalKernelEVD< @STYPE@ >;

%include "kqp/kernel_evd/divide_and_conquer.hpp"
%template(KEVDDivideAndConquer@SNAME@) kqp::DivideAndConquerBuilder< @STYPE@ >;
