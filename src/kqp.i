%module kqp
%{
  #include "kernel_evd.h"
%}

namespace kqp {
   template <typename scalar> class ScalarMatrix {
	public:
		/** Set the variable i */
		void set(int i);
   };

}

%template(DenseDoubleFeatureMatrix) kqp::ScalarMatrix<double>;

