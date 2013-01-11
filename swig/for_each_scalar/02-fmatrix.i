
%define shared_template(NAME, TYPE)
%shared_ptr(TYPE)
%template(NAME) TYPE;
%enddef

// --- Features matrices


%define AbstractSpaceCommonDefs(NAME, TYPE)
%shared_ptr(TYPE)
%template(NAME) TYPE;
%enddef

%define SpaceCommonDefs(NAME, TYPE)
AbstractSpaceCommonDefs(NAME, TYPE)
%extend TYPE {
    static const TYPE *cast(const kqp::SpaceBase< @STYPE@ > *base) {
        return dynamic_cast<const TYPE *>(base);
    }
}
%{
  kqp::SpaceFactory::Register< @STYPE@, TYPE > REGISTER_ ## NAME; 
%}

%enddef

%define FMatrixCommonDefs(NAME, TYPE)
%shared_ptr(TYPE)
%ignore TYPE::subset;
%template(NAME) TYPE;
%extend TYPE {
    static const TYPE *cast(const kqp::FeatureMatrixBase< @STYPE@ > *base) {
        return dynamic_cast<const TYPE * >(base);
    }
}
%enddef

%shared_ptr(kqp::FeatureMatrixBase< @STYPE@ >)

namespace kqp {
template<> struct ScalarDefinitions< @STYPE@ > {
   typedef kqp::AltMatrix< typename kqp::AltVector< @RTYPE@ >::VectorType,  typename kqp::AltVector< @RTYPE@ >::ConstantVectorType > RealAltVector;
   typedef kqp::AltMatrix< typename kqp::AltDense< @STYPE@ >::DenseType, typename kqp::AltDense< @STYPE@ >::IdentityType >  ScalarAltMatrix;
};
};
%template() kqp::ScalarDefinitions< @STYPE@ >;

%include <kqp/feature_matrix.hpp>
%include <kqp/space_factory.hpp>
FMatrixCommonDefs(FeatureMatrix@SNAME@, kqp::FeatureMatrixBase< @STYPE@ >)

%shared_ptr(kqp::SpaceBase< @STYPE@ >)
%template(Space@SNAME@) kqp::SpaceBase< @STYPE@ >;


// Dense
%include <kqp/feature_matrix/dense.hpp>
SpaceCommonDefs(DenseSpace@SNAME@,kqp::DenseSpace< @STYPE@ >);
FMatrixCommonDefs(Dense@SNAME@, kqp::Dense< @STYPE@ >)
%extend kqp::Dense< @STYPE@ > {
  Index dataSize() const {
    return sizeof(@STYPE@) * self->size() * self->dimension();
  };
};

// Sparse dense
%include <kqp/feature_matrix/sparse_dense.hpp>
SpaceCommonDefs(SparseDenseSpace@SNAME@, kqp::SparseDenseSpace< @STYPE@ >)
FMatrixCommonDefs(SparseDense@SNAME@, kqp::SparseDense< @STYPE@ >)

// Sparse 
%include <kqp/feature_matrix/sparse.hpp>
SpaceCommonDefs(SparseSpace@SNAME@, kqp::SparseSpace< @STYPE@ >)
FMatrixCommonDefs(Sparse@SNAME@, kqp::Sparse< @STYPE@ >)

// ---- Kernel spaces

%include <kqp/feature_matrix/unary_kernel.hpp>

AbstractSpaceCommonDefs(UnaryKernelSpace@SNAME@, kqp::UnaryKernelSpace< @STYPE@ >);

SpaceCommonDefs(GaussianSpace@SNAME@, kqp::GaussianSpace< @STYPE@ >);

SpaceCommonDefs(PolynomialSpace@SNAME@, kqp::PolynomialSpace< @STYPE@ >);

%include <kqp/feature_matrix/tensor.hpp>

%shared_ptr(kqp::TensorMatrix< @STYPE@ >);
%template(TensorMatrix@SNAME@) kqp::TensorMatrix< @STYPE@ >;

SpaceCommonDefs(TensorSpace@SNAME@, kqp::TensorSpace< @STYPE@ >);

// ---- Rank selection


%include "kqp/rank_selector.hpp"
%shared_ptr(kqp::Selector< @STYPE@ >);
%shared_ptr(kqp::RankSelector< @STYPE@, true >);
%shared_ptr(kqp::RankSelector< @STYPE@, false >);
%shared_ptr(kqp::MinimumSelector< @STYPE@ >);
%template(EigenList@SNAME@) kqp::EigenList< @STYPE@ >;


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
