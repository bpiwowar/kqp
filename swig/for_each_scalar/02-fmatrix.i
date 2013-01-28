
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
        return kqp::our_dynamic_cast< TYPE >(base);
    }
    static bool isInstance(const boost::shared_ptr<kqp::AbstractSpace> &base) {
        return kqp::our_dynamic_cast<const TYPE *>(base.get()) != 0;
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
        return kqp::our_dynamic_cast< TYPE >(base);
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

// --- Kernel values

%template(KernelValues@SNAME@) kqp::KernelValues< @STYPE@ >;
%template(KernelValuesList@SNAME@) std::vector< kqp::KernelValues< @STYPE@ > >;
%extend std::vector< kqp::KernelValues< @STYPE@ > > {
  void set(size_t i, @STYPE@ inner, @STYPE@ innerX, @STYPE@ innerY) {
    auto &x = (*self)[i];
    x._inner = inner;
    x._innerX = innerX;
    x._innerY = innerY;
  }

  void add(@STYPE@ inner, @STYPE@ innerX, @STYPE@ innerY) {
    self->push_back(kqp::KernelValues< @STYPE@ >(inner, innerX, innerY));
  }

  @STYPE@ inner(size_t i = 0) {
    return (*self)[i].inner();
  }
  @STYPE@ innerX(size_t i = 0) {
    return (*self)[i].innerX();
  }
  @STYPE@ innerY(size_t i = 0) {
    return (*self)[i].innerY();
  }
}

%template(KernelValuesListList@SNAME@) std::vector< std::vector< kqp::KernelValues< @STYPE@ > > >;
%extend std::vector< std::vector< kqp::KernelValues< @STYPE@ > > > {
    /** Adds a entry of size n and returns the index of the added value */
    size_t add(size_t n) {
        self->push_back(std::vector< kqp::KernelValues< @STYPE@ > >(n));
        return self->size() - 1;
    }
    
    void addAll(const std::vector< std::vector< kqp::KernelValues< @STYPE@ > > > &other) {
        self->insert(self->end(), other.begin(), other.end());
    }
    
    /** Add a new vector with copies */
    size_t add(size_t n, @STYPE@ inner, @STYPE@ innerX, @STYPE@ innerY) {
        self->push_back(std::vector< kqp::KernelValues< @STYPE@ > >(n, kqp::KernelValues< @STYPE@ >(inner, innerX, innerY)));
        return self->size() - 1;
    }
    
    /** Swap two entries */
    void swap(size_t i, size_t j) {
        (*self)[i].swap((*self)[j]);
    }
    
    void update(size_t i, const kqp::SpaceBase< @STYPE@ > &space) {
        space.update((*self)[i]);
    }

    void update(const kqp::SpaceBase< @STYPE@ > &space) {
        for(auto &values: *self)
            space.update(values);
    }

    void updatePartials(size_t i, const kqp::SpaceBase< @STYPE@ > &space, @RTYPE@ alpha, std::vector< @RTYPE@ > &partials, int mode) {
        space.updatePartials(alpha, partials, (*self)[i], mode);
    }

    void set(size_t i, size_t j, @STYPE@ inner, @STYPE@ innerX, @STYPE@ innerY) {
      auto &x = (*self)[i][j];
      x._inner = inner;
      x._innerX = innerX;
      x._innerY = innerY;
    }

    // Direct access to values
    @STYPE@ inner(size_t i, size_t j = 0) {
      return (*self)[i][j].inner();
    }
    @STYPE@ innerX(size_t i, size_t j = 0) {
      return (*self)[i][j].innerX();
    }
    @STYPE@ innerY(size_t i, size_t j = 0) {
      return (*self)[i][j].innerY();
    }
}

// --- Feature matrix

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

