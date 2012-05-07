// Defines arrays in the native language
/*%include "carrays.i"*/
/*%array_class(@STYPE@, Array@SNAME@);*/

namespace Eigen {
template<> struct NumTraits< @STYPE@ > {
  typedef @RTYPE@ Real;
};
}

%template() Eigen::NumTraits< @STYPE@ >;

// Matrix (Eigen)

%template(EigenMatrix@SNAME@) Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic >;
%extend Eigen::Matrix<@STYPE@, Eigen::Dynamic, Eigen::Dynamic> {
    Index row() const { return self->rows(); }; 
    Index cols() const { return self->cols(); };
    void randomize() {
         *self = Eigen::Matrix<@STYPE@, Eigen::Dynamic, Eigen::Dynamic>::Random(self->rows(), self->cols()); 
    } 
    
    @STYPE@ operator()(Index i, Index j) const { 
        return (*self)(i,j);
    }
    
    void set(Index i, Index j, @STYPE@ value) {  
        (*self)(i,j) = value; 
    } 
}

// Vector (Eigen)
%template(EigenVector@SNAME@) Eigen::Matrix< @STYPE@, Eigen::Dynamic, 1 >;
%extend Eigen::Matrix<@STYPE@, Eigen::Dynamic, 1> {
    Index row() const { return self->rows(); }; 
    Index cols() const { return self->cols(); };
    void randomize() {
         *self = Eigen::Matrix<@STYPE@, Eigen::Dynamic, Eigen::Dynamic>::Random(self->rows(), self->cols()); 
    } 
    
    @STYPE@ operator()(Index i, Index j) const { 
        return (*self)(i,j);
    }
    
    
    void set(Index i, Index j, @STYPE@ value) {  
        (*self)(i,j) = value; 
    } 
}

%define EIGENSPARSE(NAME, MODE)
%template(Eigen ## NAME ## SparseMatrix@SNAME@) Eigen::SparseMatrix< @STYPE@, MODE >;
%extend Eigen::SparseMatrix< @STYPE@, Eigen::ColMajor > {
    Index row() const { return self->rows(); }; 
    Index cols() const { return self->cols(); };
    
    void reserve(const std::vector<Index> &reserveSizes) {
        $self->reserve(reserveSizes);
    }
    void reserve(Index size) {
        $self->reserve(size);
    }
    
    void insert(Index i, Index j, @STYPE@ value) {  
        self->insert(i,j) = value; 
    }
}
%enddef

EIGENSPARSE(Row, Eigen::RowMajor)
EIGENSPARSE(Col, Eigen::ColMajor)

