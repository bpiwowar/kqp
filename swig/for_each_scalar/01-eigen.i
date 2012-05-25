// Defines arrays in the native language
/*%include "carrays.i"*/
/*%array_class(@STYPE@, Array@SNAME@);*/

%define DEFINE_COMMON_EIGEN_MATRIX_OPERATIONS()
    // Aggregations
    @RTYPE@ squaredNorm() const { return $self->squaredNorm(); }
    @RTYPE@ norm() const { return $self->squaredNorm(); }
    @STYPE@ sum() const { return $self->sum(); }
    
    // Component wise nullary operations
    void cwiseSqrt () { $self->cwiseSqrt(); }
    
    // Component wise unary operations
    void cwiseMultBy(const @STYPE@& scalar) { *$self *= scalar; }
    void cwiseDivBy(const @STYPE@& scalar) { *$self /= scalar; }
%enddef


%template(@SNAME@Ref) kqp::PrimitiveRef< @STYPE@ >;

namespace Eigen {
template<> struct NumTraits< @STYPE@ > {
  typedef @RTYPE@ Real;
};
}

%template() Eigen::NumTraits< @STYPE@ >;

// Matrix (Eigen)

%template(EigenMatrix@SNAME@) Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic >;
%extend Eigen::Matrix<@STYPE@, Eigen::Dynamic, Eigen::Dynamic> {
    Index rows() const { return self->rows(); }; 
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
    
    DEFINE_COMMON_EIGEN_MATRIX_OPERATIONS();
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
    
    DEFINE_COMMON_EIGEN_MATRIX_OPERATIONS();
}

// --- Eigen Sparse Matrices ---


// Eigen sparse matrix
%define EIGENSPARSE(NAME, MODE)


%{
 typedef Eigen::SparseMatrix< @STYPE@, MODE >::InnerIterator Eigen ## NAME ## SparseMatrixIterator@SNAME@;   
%}
#define INNERITERATOR Eigen ## NAME ## SparseMatrixIterator@SNAME@
class INNERITERATOR {
public:
    inline const @STYPE@& value() const;

    inline Index index() const;
    inline Index outer() const;
    
private:
    INNERITERATOR();
};
%extend INNERITERATOR {
    kqp::PrimitiveRef< @STYPE@ > valueRef() { 
        return $self->valueRef();
    }

    void next() { ++*$self; }
    bool eof() { return !(*$self); }    
}



%template(Eigen ## NAME ## SparseMatrix@SNAME@) Eigen::SparseMatrix< @STYPE@, MODE >;
%extend Eigen::SparseMatrix< @STYPE@, MODE > {

    kqp::PrimitiveRef< @STYPE@ > coeffRef(Index i, Index j) { 
        return self->coeffRef(i,j);
    }

    Index rows() const { return self->rows(); }; 
    Index cols() const { return self->cols(); };

    INNERITERATOR iterator(Index outerIndex) {
        return Eigen::SparseMatrix< @STYPE@, MODE>::InnerIterator(*$self,outerIndex);
    }
    
    DEFINE_COMMON_EIGEN_MATRIX_OPERATIONS();
    
    
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

