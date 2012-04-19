// ---- Probabilities

%include "kqp/probabilities.hpp"

%template(KernelOperator@FNAME@) kqp::KernelOperator< @FTYPE@ >;
%template(Event@FNAME@) kqp::Event< @FTYPE@ >;
%template(Density@FNAME@) kqp::Density< @FTYPE@ >;
