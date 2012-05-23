#define SCALAR_ALTMATRIX_@SNAME@ kqp::AltMatrix< typename kqp::AltDense<@STYPE@>::DenseType, typename kqp::AltDense<@STYPE@>::IdentityType >
%template(AltMatrix@SNAME@) SCALAR_ALTMATRIX_@SNAME@;
%extend SCALAR_ALTMATRIX_@SNAME@ {
    static SCALAR_ALTMATRIX_@SNAME@ createIdentity(Index rows, Index cols) {
        return Eigen::Matrix<@STYPE@,Eigen::Dynamic,Eigen::Dynamic>::Identity(rows, cols);
    }
    static SCALAR_ALTMATRIX_@SNAME@ adopt(Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic > &other) {
        return SCALAR_ALTMATRIX_@SNAME@(std::move(other));
    }
    static SCALAR_ALTMATRIX_@SNAME@ copy(const Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic > &other) {
        return SCALAR_ALTMATRIX_@SNAME@(other);
    }
    
    kqp::_AltMatrix::AltMatrixType getType() {
        return self->isT1() ? kqp::_AltMatrix::DENSE : kqp::_AltMatrix::IDENTITY;
    }
    
    Index rows() const { return $self->rows(); }
    Index cols() const { return $self->cols(); }
    
}

#ifndef REAL_ALTVECTOR_@RNAME@
#define REAL_ALTVECTOR_@RNAME@ kqp::AltMatrix< typename kqp::AltVector<@RTYPE@>::VectorType,  typename kqp::AltVector<@RTYPE@>::ConstantVectorType >

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
        return self->isT1() ? kqp::_AltVector::DENSE : kqp::_AltVector::CONSTANT;
    }
    
}
#endif