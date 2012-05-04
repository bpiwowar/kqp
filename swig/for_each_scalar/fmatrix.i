

// --- Features matrices


%shared_ptr(FeatureMatrixBase@SNAME@)

%include <kqp/feature_matrix.hpp>
%template(FeatureMatrixBase@SNAME@) kqp::FeatureMatrixBase< @STYPE@ >;
%template(FeatureMatrix@SNAME@) kqp::FeatureMatrix< @STYPE@ >;
%template(FeatureSpace@SNAME@) kqp::FeatureSpace< @STYPE@ >;
%template(FeatureSpaceBase@SNAME@) kqp::FeatureSpaceBase< @STYPE@ >;


// Dense
%shared_ptr(Dense@SNAME@)
%shared_ptr(DenseSpace@SNAME@)
%include <kqp/feature_matrix/dense.hpp>
%template(DenseSpace@SNAME@) kqp::DenseFeatureSpace< @STYPE@ >;
%template(Dense@SNAME@) kqp::DenseMatrix< @STYPE@ >;
%extend kqp::DenseMatrix< @STYPE@ > {
  Index dataSize() const {
    return sizeof(@STYPE@) * self->size() * self->dimension();
  };
};

// Sparse dense
%include <kqp/feature_matrix/sparse_dense.hpp>
%template(SparseDense@SNAME@) kqp::SparseDenseMatrix< @STYPE@ >;
%template(SparseDenseSpace@SNAME@) kqp::SparseDenseFeatureSpace< @STYPE@ >;

// Sparse dense
%include <kqp/feature_matrix/sparse.hpp>
%template(Sparse@SNAME@) kqp::SparseMatrix< @STYPE@ >;
%template(SparseSpace@SNAME@) kqp::SparseFeatureSpace< @STYPE@ >;

// ---- Decompositions & cleaning

%include "kqp/rank_selector.hpp"
%include "kqp/cleanup.hpp"
%shared_ptr(Selector@SNAME@);
%shared_ptr(RankSelectorAbs@SNAME@);
%shared_ptr(RankSelector@SNAME@);

%ignore kqp::Selector< @STYPE@ >::selection;
%template(Selector@SNAME@) kqp::Selector< @STYPE@ >;

%template(RankSelectorAbs@SNAME@) kqp::RankSelector< @STYPE@,true >;
%ignore kqp::RankSelector< @STYPE@, true >::selection;

