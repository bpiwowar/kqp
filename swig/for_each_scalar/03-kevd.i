
// ---- Rank selection


%include "kqp/rank_selector.hpp"
%shared_ptr(kqp::RankSelector< @STYPE@, true >);
%shared_ptr(kqp::RankSelector< @STYPE@, false >);
%shared_ptr(kqp::RatioSelector< @STYPE@ >);


%template(EigenList@SNAME@) kqp::EigenList< @STYPE@ >;

%ignore kqp::Selector< @STYPE@ >::selection;
shared_template(Selector@SNAME@, kqp::Selector< @STYPE@ >);

shared_template(Aggregator@SNAME@, kqp::Aggregator< @STYPE@ >);
shared_template(AggregatorMax@SNAME@, kqp::Max< @STYPE@ >);
shared_template(AggregatorMean@SNAME@, kqp::Mean< @STYPE@ >);

%ignore kqp::RatioSelector@SNAME@::selection;
%template(RatioSelector@SNAME@) kqp::RatioSelector< @STYPE@ >;

%ignore kqp::RankSelector< @STYPE@, true >::selection;
%template(RankSelectorAbs@SNAME@) kqp::RankSelector< @STYPE@,true >;

%ignore kqp::RankSelector< @STYPE@, false >::selection;
%template(RankSelector@SNAME@) kqp::RankSelector< @STYPE@,false >;

%ignore kqp::ChainSelector< @STYPE@ >::selection;
shared_template(ChainSelector@SNAME@, kqp::ChainSelector< @STYPE@ >);

// --- Decomposition

%include "kqp/decomposition.hpp"
shared_template(Decomposition@SNAME@, kqp::Decomposition< @STYPE@ >)
%extend kqp::Decomposition< @STYPE@ > {
    std::string save() {
        std::ostringstream oss;
        boost::archive::binary_oarchive ar(oss);
        ar & *self;
        return oss.str();
    }
    
    void load(const std::string &data) {
        std::istringstream iss(data);
        boost::archive::binary_iarchive ar(iss);
        ar & *self;        
    }
}

// --- Decomposition cleaner


%define DefineCleaner(NAME, TYPE)
DefineTemplateClass(NAME, TYPE, kqp::CleanerBase)
%enddef

%include "kqp/cleanup.hpp"
DefineCleaner(Cleaner@SNAME@, kqp::Cleaner< @STYPE@ >);
DefineCleaner(CleanerList@SNAME@, kqp::CleanerList< @STYPE@ >);
DefineCleaner(CleanerRank@SNAME@, kqp::CleanerRank< @STYPE@ >);

%include "kqp/cleaning/qp_approach.hpp"
DefineCleaner(CleanerQP@SNAME@, kqp::CleanerQP< @STYPE@ >);

%include "kqp/cleaning/unused.hpp"
DefineCleaner(CleanerUnused@SNAME@, kqp::CleanerUnused< @STYPE@ >);

%include "kqp/cleaning/null_space.hpp"
DefineCleaner(CleanerNullSpace@SNAME@, kqp::CleanerNullSpace< @STYPE@ >);

%include "kqp/cleaning/collapse.hpp"
DefineCleaner(CleanerCollapse@SNAME@, kqp::CleanerCollapse< @STYPE@ >);

// ---- Kernel EVD

%define DefineKernelEVD(NAME, TYPE)
DefineTemplateClass(NAME, %kqparg(TYPE), kqp::KernelEVDBase)
%enddef

%shared_ptr(kqp::KernelEVD< @STYPE@ >);
%shared_ptr(kqp::DenseDirectBuilder< @STYPE@ >);
%shared_ptr(kqp::AccumulatorKernelEVD< @STYPE@, true >)
%shared_ptr(kqp::AccumulatorKernelEVD< @STYPE@, false >)
%shared_ptr(kqp::DivideAndConquerBuilder< @STYPE@ >)
%shared_ptr(kqp::IncrementalKernelEVD< @STYPE@ >)

%include "kqp/kernel_evd.hpp"
DefineKernelEVD(KEVD@SNAME@, kqp::KernelEVD< @STYPE@ >);

%include "kqp/kernel_evd/dense_direct.hpp"
DefineKernelEVD(KEVDDirect@SNAME@, kqp::DenseDirectBuilder< @STYPE@ >);

%include "kqp/kernel_evd/accumulator.hpp"
DefineKernelEVD(KEVDLCAccumulator@SNAME@, %kqparg(kqp::AccumulatorKernelEVD< @STYPE@, true >));

%include "kqp/kernel_evd/accumulator.hpp"
DefineKernelEVD(KEVDAccumulator@SNAME@, %kqparg(kqp::AccumulatorKernelEVD< @STYPE@, false >));

%include "kqp/kernel_evd/incremental.hpp"
DefineKernelEVD(KEVDIncremental@SNAME@, kqp::IncrementalKernelEVD< @STYPE@ >);

%include "kqp/kernel_evd/divide_and_conquer.hpp"
DefineKernelEVD(KEVDDivideAndConquer@SNAME@, kqp::DivideAndConquerBuilder< @STYPE@ >);


// Save & Load decompositions

