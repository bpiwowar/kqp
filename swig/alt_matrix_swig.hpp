#include <kqp/alt_matrix.hpp>

namespace kqp {
    namespace swig {
        
        // Wrapper for AltVector
        template<typename Scalar> class AltVector {
            typename kqp::AltVector<Scalar>::type m_value;
        public:
            enum Type { DENSE, CONSTANT };
            
            Type getType() {
                return m_value.isT1() ? DENSE : CONSTANT;
            }
        };
        
        template<typename Scalar> class AltMatrix {
            typename kqp::AltDense<Scalar>::type m_value;
        public:
            enum Type { DENSE, IDENTITY };
            
            Type getType() {
                return m_value.isT1() ? DENSE : IDENTITY;
            }
            
        };
    }
}
