// --- Alt vectors

%include "alt_matrix_swig.hpp"

%template(AltMatrix@SNAME@) kqp::swig::AltMatrix< @STYPE@ >;
%template(AltVector@SNAME@) kqp::swig::AltVector< @STYPE@ >;


%include <kqp/feature_matrix.hpp>
%template() kqp::FeatureMatrix< kqp::DenseMatrix< @STYPE@ > >;


namespace kqp {
template<> struct ftraits< DenseMatrix<@STYPE@ > > {
    typedef @STYPE@ Scalar; typedef @RTYPE@ Real;
    typedef kqp::swig::AltMatrix<@STYPE@ > ScalarAltMatrix;
    typedef kqp::swig::AltVector<@RTYPE@ > RealAltVector;
    typedef Eigen::Matrix<@RTYPE@ ,Eigen::Dynamic,Eigen::Dynamic> ScalarMatrix;
};
}


%include <kqp/feature_matrix/dense.hpp>

%template() kqp::ftraits< kqp::DenseMatrix< @STYPE@ > >;
%template(Dense@SNAME@) kqp::DenseMatrix< @STYPE@ >;

// ---- Decompositions & cleaning


%include "kqp/rank_selector.hpp"
%include "kqp/decomposition.hpp"
%include "kqp/cleanup.hpp"

%template(Selector@SNAME@) kqp::Selector< @STYPE@ >;
%template(RankSelectorAbs@SNAME@) kqp::RankSelector< @STYPE@,true >;

