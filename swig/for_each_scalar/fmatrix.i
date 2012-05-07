

// --- Features matrices

%define FMatrixCommonDefs(NAME, TYPE)
%shared_ptr(TYPE)
%ignore TYPE::subset;
%template(NAME) TYPE;
%enddef

%shared_ptr(kqp::FeatureSpaceBase< @STYPE@ >)
%shared_ptr(kqp::DenseFeatureSpace< @STYPE@ >)
%shared_ptr(kqp::SparseDenseFeatureSpace< @STYPE@ >)
%shared_ptr( kqp::SparseFeatureSpace< @STYPE@ >)

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
%template(FeatureSpace@SNAME@) kqp::FeatureSpace< @STYPE@ >;
%template(FeatureSpaceBase@SNAME@) kqp::FeatureSpaceBase< @STYPE@ >;


// Dense
%include <kqp/feature_matrix/dense.hpp>
%template(DenseSpace@SNAME@) kqp::DenseFeatureSpace< @STYPE@ >;

FMatrixCommonDefs(Dense@SNAME@, kqp::DenseMatrix< @STYPE@ >)
%extend kqp::DenseMatrix< @STYPE@ > {
  Index dataSize() const {
    return sizeof(@STYPE@) * self->size() * self->dimension();
  };
};

// Sparse dense
%include <kqp/feature_matrix/sparse_dense.hpp>
%template(SparseDenseSpace@SNAME@) kqp::SparseDenseFeatureSpace< @STYPE@ >;
FMatrixCommonDefs(SparseDense@SNAME@, kqp::SparseDenseMatrix< @STYPE@ >)

// Sparse dense
%include <kqp/feature_matrix/sparse.hpp>
%template(SparseSpace@SNAME@) kqp::SparseFeatureSpace< @STYPE@ >;
FMatrixCommonDefs(Sparse@SNAME@, kqp::SparseMatrix< @STYPE@ >)


// ---- Kernel spaces

%include <kqp/feature_matrix/unary_kernel.hpp>
%shared_ptr(kqp::UnaryKernelSpace< @STYPE@ >);
%shared_ptr(kqp::GaussianKernelSpace< @STYPE@ >);
%shared_ptr(kqp::PolynomialKernelSpace< @STYPE@ >);
%template(UnaryKernelSpace@SNAME@) kqp::UnaryKernelSpace< @STYPE@ >;
%template(GaussianSpace@SNAME@) kqp::GaussianKernelSpace< @STYPE@ >;
%template(PolynomialSpace@SNAME@) kqp::PolynomialKernelSpace< @STYPE@ >;


// ---- Decompositions & cleaning

%include "kqp/rank_selector.hpp"
%include "kqp/cleanup.hpp"
%shared_ptr(kqp::Selector< @STYPE@ >);
%shared_ptr(kqp::RankSelector< @STYPE@, true >);
%shared_ptr(kqp::RankSelector< @STYPE@, false >);


%ignore kqp::Selector< @STYPE@ >::selection;
%template(Selector@SNAME@) kqp::Selector< @STYPE@ >;

%ignore kqp::RankSelector< @STYPE@, true >::selection;
%template(RankSelectorAbs@SNAME@) kqp::RankSelector< @STYPE@,true >;

%ignore kqp::RankSelector< @STYPE@, false >::selection;
%template(RankSelector@SNAME@) kqp::RankSelector< @STYPE@,false >;

