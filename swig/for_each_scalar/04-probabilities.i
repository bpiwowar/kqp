// ---- Probabilities

%include "kqp/probabilities.hpp"

%template(KernelOperator@SNAME@) kqp::KernelOperator< @STYPE@ >;
%template(Event@SNAME@) kqp::Event< @STYPE@ >;
%template(Density@SNAME@) kqp::Density< @STYPE@ >;
