#include "dense_direct.hpp"

namespace kqp {
    // Instanciation of standard classes
    template class DenseDirectBuilder<double>;
    template class DenseDirectBuilder<float>;
    template class DenseDirectBuilder<std::complex<double> >;
    template class DenseDirectBuilder<std::complex<float> >;
}