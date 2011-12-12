#include "dense.hpp"

namespace kqp {
    template class DenseVector<double>;
    template class DenseVector<float>;
    template class DenseVector<std::complex<double> >;
    template class DenseVector<std::complex<float> >;
    
    template class DenseMatrix<double>;
    template class DenseMatrix<float>;
    template class DenseMatrix<std::complex<double> >;
    template class DenseMatrix<std::complex<float> >;

}
