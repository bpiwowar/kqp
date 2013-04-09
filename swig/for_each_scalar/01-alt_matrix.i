
#define SCALAR_ALTMATRIX_@SNAME@ kqp::AltMatrix< typename kqp::AltDense<@STYPE@>::DenseType, typename kqp::AltDense<@STYPE@>::IdentityType >

#define SCALARMATRIX Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic>

%template(@SNAME@List) std::vector< @STYPE@ >;

// Either a dense matrix or the identity
%template(AltMatrix@SNAME@) SCALAR_ALTMATRIX_@SNAME@;
%extend SCALAR_ALTMATRIX_@SNAME@ {
    static SCALAR_ALTMATRIX_@SNAME@ createIdentity(Index rows, Index cols) {
        return Eigen::Identity< @STYPE@ >(rows, cols);
    }
    static SCALAR_ALTMATRIX_@SNAME@ adopt(Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic > &other) {
        return SCALAR_ALTMATRIX_@SNAME@(std::move(other));
    }
    static SCALAR_ALTMATRIX_@SNAME@ copy(const Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic > &other) {
        return SCALAR_ALTMATRIX_@SNAME@(other);
    }
    
    kqp::_AltMatrix::AltMatrixType getType() {
        return self->isT1() ? kqp::_AltMatrix::DENSE_MATRIX : kqp::_AltMatrix::IDENTITY_MATRIX;
    }

    @STYPE@ get(Index i, Index j) const { return $self->operator()(i, j);} 
    
    Index rows() const { return $self->rows(); }
    Index cols() const { return $self->cols(); }
    
    SCALARMATRIX getDense() { return $self->t1(); }
    
    // Multiply with another matrix
    SCALARMATRIX multBy(const SCALARMATRIX &other) {
         return SCALARMATRIX((*$self) * other); 
    }
}

#ifndef REAL_ALTVECTOR_@RNAME@
#define REAL_ALTVECTOR_@RNAME@ kqp::AltMatrix< typename kqp::AltVector<@RTYPE@>::VectorType,  typename kqp::AltVector<@RTYPE@>::ConstantVectorType >

// Either a vector or a constant vector
%template(AltVector@RNAME@) REAL_ALTVECTOR_@RNAME@;
%extend REAL_ALTVECTOR_@RNAME@ {
    static REAL_ALTVECTOR_@RNAME@ createConstant(Index size, @RTYPE@ x) {
        return typename kqp::AltVector< @RTYPE@ >::ConstantVectorType(size, 1, x);
    }
    static REAL_ALTVECTOR_@RNAME@ adopt(Eigen::Matrix< @RTYPE@, Eigen::Dynamic, 1 > &other) {
        return REAL_ALTVECTOR_@RNAME@(std::move(other));
    }
    static REAL_ALTVECTOR_@RNAME@ copy(const Eigen::Matrix< @RTYPE@, Eigen::Dynamic, 1 > &other) {
        return REAL_ALTVECTOR_@RNAME@(other);
    }
    
    Index size() const { return $self->size(); }
    Index rows() const { return $self->rows(); }
    Index cols() const { return $self->cols(); }
    
    kqp::_AltVector::AltVectorType getType() {
        return self->isT1() ? kqp::_AltVector::DENSE_VECTOR : kqp::_AltVector::CONSTANT_VECTOR;
    }
    
    @STYPE@ get(Index i) { return $self->operator()(i, 0); }
    SCALARMATRIX getDense() { return $self->t1(); }
    @STYPE@ getConstant() { return $self->getStorage2().m_value; }
    
    SCALARMATRIX asDiagPostMultBy(const SCALARMATRIX &other) {
         return SCALARMATRIX(other * $self->asDiagonal()); 
    }    
}

#endif

#undef SCALARMATRIX