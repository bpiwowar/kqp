// Defines arrays in the native language
/*%include "carrays.i"*/
/*%array_class(@STYPE@, Array@SNAME@);*/


%template(EigenMatrix@SNAME@) Eigen::Matrix< @STYPE@, Eigen::Dynamic, Eigen::Dynamic >;

%extend Eigen::Matrix<@STYPE@, Eigen::Dynamic, Eigen::Dynamic> {
    Index row() const { return self->rows(); }; 
    Index cols() const { return self->cols(); };
    void randomize() {
         *self = Eigen::Matrix<@STYPE@, Eigen::Dynamic, Eigen::Dynamic>::Random(self->rows(), self->cols()); 
    } 
    
    @STYPE@ operator()(Index i, Index j) const { 
        std::cerr << "Getting " << i << "," << j << std::endl; return (*self)(i,j); 
    }
    
    void set(Index i, Index j, @STYPE@ value) {  
        std::cerr << "Setting " << i << "," << j << " to " << value << std::endl;
        (*self)(i,j) = value; 
    } 
}

%template(EigenVector@SNAME@) Eigen::Matrix< @STYPE@, Eigen::Dynamic, 1 >;
%extend Eigen::Matrix<@STYPE@, Eigen::Dynamic, 1> {
    Index row() const { return self->rows(); }; 
    Index cols() const { return self->cols(); };
    void randomize() {
         *self = Eigen::Matrix<@STYPE@, Eigen::Dynamic, Eigen::Dynamic>::Random(self->rows(), self->cols()); 
    } 
    
    @STYPE@ operator()(Index i, Index j) const { 
        std::cerr << "Getting " << i << "," << j << std::endl; return (*self)(i,j); 
    }
    
    void set(Index i, Index j, @STYPE@ value) {  
        std::cerr << "Setting " << i << "," << j << " to " << value << std::endl;
        (*self)(i,j) = value; 
    } 
}



