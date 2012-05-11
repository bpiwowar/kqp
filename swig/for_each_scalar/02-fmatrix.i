
%define shared_template(NAME, TYPE)
%shared_ptr(TYPE)
%template(NAME) TYPE;
%enddef

// --- Features matrices

%define FMatrixCommonDefs(NAME, TYPE)
%shared_ptr(TYPE)
%ignore TYPE::subset;
%template(NAME) TYPE;
%enddef

%shared_ptr(kqp::SpaceBase< @STYPE@ >)
%shared_ptr(kqp::DenseSpace< @STYPE@ >)
%shared_ptr(kqp::SparseDenseSpace< @STYPE@ >)
%shared_ptr( kqp::SparseSpace< @STYPE@ >)

%shared_ptr(kqp::FeatureMatrixBase< @STYPE@ >)

namespace kqp {
template<> struct ScalarDefinitions< @STYPE@ > {
   typedef kqp::AltMatrix< typename kqp::AltVector< @RTYPE@ >::VectorType,  typename kqp::AltVector< @RTYPE@ >::ConstantVectorType > RealAltVector;
   typedef kqp::AltMatrix< typename kqp::AltDense< @STYPE@ >::DenseType, typename kqp::AltDense< @STYPE@ >::IdentityType >  ScalarAltMatrix;
};
};
%template() kqp::ScalarDefinitions< @STYPE@ >;

%include <kqp/feature_matrix.hpp>
FMatrixCommonDefs(FeatureMatrixBase@SNAME@, kqp::FeatureMatrixBase< @STYPE@ >)
%ignore kqp::FeatureMatrix< @STYPE@ >::subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) const;
%template(FeatureMatrix@SNAME@) kqp::FeatureMatrix< @STYPE@ >;
%template(Space@SNAME@) kqp::Space< @STYPE@ >;
%template(SpaceBase@SNAME@) kqp::SpaceBase< @STYPE@ >;


// Dense
%include <kqp/feature_matrix/dense.hpp>
%template(DenseSpace@SNAME@) kqp::DenseSpace< @STYPE@ >;

FMatrixCommonDefs(Dense@SNAME@, kqp::Dense< @STYPE@ >)
%extend kqp::Dense< @STYPE@ > {
  Index dataSize() const {
    return sizeof(@STYPE@) * self->size() * self->dimension();
  };
};

// Sparse dense
%include <kqp/feature_matrix/sparse_dense.hpp>
%template(SparseDenseSpace@SNAME@) kqp::SparseDenseSpace< @STYPE@ >;
FMatrixCommonDefs(SparseDense@SNAME@, kqp::SparseDense< @STYPE@ >)

// Sparse dense
%include <kqp/feature_matrix/sparse.hpp>
%template(SparseSpace@SNAME@) kqp::SparseSpace< @STYPE@ >;
FMatrixCommonDefs(Sparse@SNAME@, kqp::Sparse< @STYPE@ >)

// ---- Kernel spaces

%include <kqp/feature_matrix/unary_kernel.hpp>
%shared_ptr(kqp::UnaryKernelSpace< @STYPE@ >);
%shared_ptr(kqp::GaussianSpace< @STYPE@ >);
%shared_ptr(kqp::PolynomialSpace< @STYPE@ >);
%template(UnaryKernelSpace@SNAME@) kqp::UnaryKernelSpace< @STYPE@ >;
%template(GaussianSpace@SNAME@) kqp::GaussianSpace< @STYPE@ >;
%template(PolynomialSpace@SNAME@) kqp::PolynomialSpace< @STYPE@ >;


// ---- Rank selection

%include "kqp/rank_selector.hpp"
%shared_ptr(kqp::Selector< @STYPE@ >);
%shared_ptr(kqp::RankSelector< @STYPE@, true >);
%shared_ptr(kqp::RankSelector< @STYPE@, false >);
%shared_ptr(kqp::MinimumSelector< @STYPE@ >);


%ignore kqp::Selector< @STYPE@ >::selection;
%template(Selector@SNAME@) kqp::Selector< @STYPE@ >;

%ignore MinimumSelector@SNAME@::selection;
%template(MinimumSelector@SNAME@) kqp::MinimumSelector< @STYPE@ >;

%ignore kqp::RankSelector< @STYPE@, true >::selection;
%template(RankSelectorAbs@SNAME@) kqp::RankSelector< @STYPE@,true >;

%ignore kqp::RankSelector< @STYPE@, false >::selection;
%template(RankSelector@SNAME@) kqp::RankSelector< @STYPE@,false >;

// --- Cleanup
%include "kqp/cleanup.hpp"
%shared_template(Cleaner@SNAME@, kqp::Cleaner< @STYPE@ >);
%shared_template(CleanerRank@SNAME@, kqp::CleanerRank< @STYPE@ >);
%shared_template(CleanerList@SNAME@, kqp::CleanerList< @STYPE@ >);
