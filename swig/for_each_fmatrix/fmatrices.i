
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

%include "kqp/kernel_evd/dense_direct.hpp"
%template(KEVDDirect@SNAME@) kqp::DenseDirectBuilder< @STYPE@ >;

%include "kqp/kernel_evd/accumulator.hpp"
%template(KEVDAccumulator@FNAME@) kqp::AccumulatorKernelEVD< @FTYPE@, true >;

%include "kqp/kernel_evd/incremental.hpp"
%template(KEVDIncremental@FNAME@) kqp::IncrementalKernelEVD< @FTYPE@ >;

%include "kqp/kernel_evd/divide_and_conquer.hpp"
%template(KEVDDivideAndConquer@FNAME@) kqp::DivideAndConquerBuilder< @FTYPE@ >;



%shared_ptr(IntValue)

%inline %{
#include <boost/shared_ptr.hpp>

struct IntValue {
  int value;
  IntValue(int v) : value(v) {}
};

static int extractValue(const IntValue &t) {
  return t.value;
}

static int extractValueSmart(boost::shared_ptr<IntValue> t) {
  return t->value;
}
%}