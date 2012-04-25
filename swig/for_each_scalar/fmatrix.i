
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
    
    template<> struct ftraits< kqp::SparseDenseMatrix<@STYPE@ > > {
        typedef @STYPE@ Scalar; 
        typedef @RTYPE@ Real;
        typedef SCALAR_ALTMATRIX_@SNAME@ ScalarAltMatrix;
        typedef Eigen::Matrix<@RTYPE@ ,Eigen::Dynamic,Eigen::Dynamic> ScalarMatrix;
        
        typedef Eigen::Matrix<@RTYPE@ ,Eigen::Dynamic,1> RealVector;
        typedef REAL_ALTVECTOR_@RNAME@ RealAltVector;
    };

    template<> struct ftraits< kqp::SparseMatrix<@STYPE@ > > {
        typedef @STYPE@ Scalar; 
        typedef @RTYPE@ Real;
        typedef SCALAR_ALTMATRIX_@SNAME@ ScalarAltMatrix;
        typedef Eigen::Matrix<@RTYPE@ ,Eigen::Dynamic,Eigen::Dynamic> ScalarMatrix;
        
        typedef Eigen::Matrix<@RTYPE@ ,Eigen::Dynamic,1> RealVector;
        typedef REAL_ALTVECTOR_@RNAME@ RealAltVector;
    };
    
}



// --- Features matrices

%include <kqp/feature_matrix.hpp>


// Dense
%template() kqp::FeatureMatrix< kqp::DenseMatrix< @STYPE@ > >;
%template() kqp::ftraits< kqp::DenseMatrix< @STYPE@ > >;
%include <kqp/feature_matrix/dense.hpp>
%template(Dense@SNAME@) kqp::DenseMatrix< @STYPE@ >;
%extend kqp::DenseMatrix< @STYPE@ > {
  Index dataSize() const {
    return sizeof(@STYPE@) * self->size() * self->dimension();
  };
};

// Sparse dense
%template() kqp::FeatureMatrix< kqp::SparseDenseMatrix< @STYPE@ > >;
%template() kqp::ftraits< kqp::SparseDenseMatrix< @STYPE@ > >;
%include <kqp/feature_matrix/sparse_dense.hpp>
%template(SparseDense@SNAME@) kqp::SparseDenseMatrix< @STYPE@ >;

// Sparse dense
%template() kqp::FeatureMatrix< kqp::SparseMatrix< @STYPE@ > >;
%template() kqp::ftraits< kqp::SparseMatrix< @STYPE@ > >;
%include <kqp/feature_matrix/sparse.hpp>
%template(Sparse@SNAME@) kqp::SparseMatrix< @STYPE@ >;

// ---- Decompositions & cleaning

%include "kqp/rank_selector.hpp"
%include "kqp/cleanup.hpp"
%shared_ptr(Selector@SNAME@);
%shared_ptr(RankSelectorAbs@SNAME@);
%shared_ptr(RankSelector@SNAME@);

%ignore kqp::Selector< @FTYPE@ >::selection;
%template(Selector@SNAME@) kqp::Selector< @STYPE@ >;

%template(RankSelectorAbs@SNAME@) kqp::RankSelector< @STYPE@,true >;
%ignore kqp::RankSelector< @STYPE@, true >::selection;

