#include <kqp/cleanup.hpp>

# define KQP_FMATRIX_GEN(type)  \
  template class kqp::Cleaner<type>; \
  template class kqp::StandardCleaner<type>; 
# include <kqp/for_all_fmatrix_gen>
