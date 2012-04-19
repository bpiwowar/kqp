// --- Decomposition & cleaning

%shared_ptr(kqp::Selector< @FTYPE@ > );
%ignore kqp::Selector< @FTYPE@ >::selection;

%shared_ptr(kqp::RankSelector< @FTYPE@,true > );
%ignore kqp::RankSelector< @FTYPE@, true >::selection;

%template( Cleaner@FNAME@) kqp::Cleaner< @FTYPE@ >;
%shared_ptr(kqp::Cleaner< @FTYPE@ >);

%template( StandardCleaner@FNAME@) kqp::StandardCleaner< @FTYPE@ >;
%shared_ptr(kqp::StandardCleaner< @FTYPE@ > );

%template(Decomposition@FNAME@) kqp::Decomposition< @FTYPE@ >;

// ---- Kernel EVD

%include "kqp/kernel_evd.hpp"
%template(KEVD@FNAME@) kqp::KernelEVD< @FTYPE@ >;
%shared_ptr(kqp::KernelEVD< @FTYPE@ > );

%include "kqp/kernel_evd/dense_direct.hpp"
%template(KEVDDirect@FNAME@) kqp::DenseDirectBuilder< @STYPE@ >;

%include "kqp/kernel_evd/accumulator.hpp"
%template(KEVDAccumulator@FNAME@) kqp::AccumulatorKernelEVD< @FTYPE@, true >;

%include "kqp/kernel_evd/incremental.hpp"
%template(KEVDIncremental@FNAME@) kqp::IncrementalKernelEVD< @FTYPE@ >;

%include "kqp/kernel_evd/divide_and_conquer.hpp"
%template(KEVDDivideAndConquer@FNAME@) kqp::DivideAndConquerBuilder< @FTYPE@ >;
