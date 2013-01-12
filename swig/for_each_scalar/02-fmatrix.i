
%define shared_template(NAME, TYPE)
%shared_ptr(TYPE)
%template(NAME) TYPE;
%enddef

// --- Features matrices


%define AbstractSpaceCommonDefs(NAME, TYPE)
%shared_ptr(TYPE)
%template(NAME) TYPE;
%extend TYPE {
    static boost::shared_ptr< TYPE > cast(const boost::shared_ptr<kqp::AbstractSpace> &base) {
        return boost::dynamic_pointer_cast< TYPE >(base);
    }
    static bool isInstance(const boost::shared_ptr<kqp::AbstractSpace> &base) {
        return dynamic_cast<const TYPE *>(base.get()) != 0;
    }
}
%enddef

%define SpaceCommonDefs(NAME, TYPE)
AbstractSpaceCommonDefs(NAME, TYPE)
%{
  kqp::SpaceFactory::Register< @STYPE@, TYPE > REGISTER_ ## NAME; 
%}
%enddef

%define FMatrixCommonDefs(NAME, TYPE)
%shared_ptr(TYPE)
%ignore TYPE::subset;
%template(NAME) TYPE;
%extend TYPE {
    static boost::shared_ptr< TYPE > cast(const boost::shared_ptr<kqp::FeatureMatrixBase< @STYPE@ > > &base) {
        return boost::dynamic_pointer_cast< TYPE >(base);
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
AbstractSpaceCommonDefs(Space@SNAME@, kqp::SpaceBase< @STYPE@ >)
%template(KernelValues@SNAME@) kqp::KernelValues< @STYPE@ >;
%template(KernelValuesList@SNAME@) std::vector< kqp::KernelValues< @STYPE@ > >;

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


%include <kqp/feature_matrix/kernel_sum.hpp>

%template(FeatureMatrixList@SNAME@) std::vector< boost::shared_ptr< kqp::FeatureMatrixBase< @STYPE@ > > >;
FMatrixCommonDefs(KernelSumMatrix@SNAME@, kqp::KernelSumMatrix< @STYPE@ >);
SpaceCommonDefs(KernelSumSpace@SNAME@, kqp::KernelSumSpace< @STYPE@ >);

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
