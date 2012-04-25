
// --- Traits for feature matrices

namespace kqp {
    template<> struct ftraits< kqp::DenseMatrix<@STYPE@ > > {
        typedef @STYPE@ Scalar; 
        typedef @RTYPE@ Real;
        typedef SCALAR_ALTMATRIX_@SNAME@ ScalarAltMatrix;
        typedef Eigen::Matrix<@RTYPE@ ,Eigen::Dynamic,Eigen::Dynamic> ScalarMatrix;
        
        typedef Eigen::Matrix<@RTYPE@ ,Eigen::Dynamic,1> RealVector;
        typedef REAL_ALTVECTOR_@RNAME@ RealAltVector;
    };
    
}


%template() kqp::ftraits< kqp::DenseMatrix< @STYPE@ > >;

// --- Features matrices

%include <kqp/feature_matrix.hpp>


// Dense
%template() kqp::FeatureMatrix< kqp::DenseMatrix< @STYPE@ > >;
%include <kqp/feature_matrix/dense.hpp>
%template(Dense@SNAME@) kqp::DenseMatrix< @STYPE@ >;
%extend kqp::DenseMatrix< @STYPE@ > {
  Index dataSize() const {
    return sizeof(@STYPE@) * self->size() * self->dimension();
  };
};

// Sparse dense
%template() kqp::FeatureMatrix< kqp::SparseDenseMatrix< @STYPE@ > >;
%include <kqp/feature_matrix/sparse_dense.hpp>
%template(SparseDense@SNAME@) kqp::SparseDenseMatrix< @STYPE@ >;

// ---- Decompositions & cleaning

%include "kqp/rank_selector.hpp"
%include "kqp/cleanup.hpp"

%ignore kqp::Selector< @FTYPE@ >::selection;
%template(Selector@SNAME@) kqp::Selector< @STYPE@ >;
%shared_ptr(RankSelectorAbs@SNAME@);

%template(RankSelectorAbs@SNAME@) kqp::RankSelector< @STYPE@,true >;
%ignore kqp::RankSelector< @STYPE@, true >::selection;
%shared_ptr(RankSelectorAbs@SNAME@);

