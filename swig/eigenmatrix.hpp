#include <kqp/kqp.hpp>

namespace kqp {
    namespace swig {
        using namespace Eigen;
        using namespace kqp;
    
        template<typename _Scalar>
        class EigenMatrix 
        #ifndef SWIG
            : public Eigen::Matrix<_Scalar, Dynamic, Dynamic> 
        #endif
        {
        public:
            typedef _Scalar Scalar;
            typedef Matrix<Scalar, Dynamic, Dynamic> Base;
        
            // Constructors
            EigenMatrix() {}
            EigenMatrix(Index rows, Index cols) : Base(rows,cols) {}
        
            // Getters and setters
            void set(Index i, Index j, Scalar value) { (*this)(i,j) = value; }
            Scalar get(Index i, Index j) const { return (*this)(i,j); }
        };
    }
}