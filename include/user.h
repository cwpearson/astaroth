// clang-format off
#ifdef PENCIL_ASTAROTH

  #include "../cparam.inc_c.h"
  #define NGHOST nghost
  #define STENCIL_ORDER (2*nghost)
  #include "PC_moduleflags.h"
  #define CONFIG_PATH
  #define AC_MULTIGPU_ENABLED (false)
  #ifdef DOUBLE_PRECISION
    #define AC_DOUBLE_PRECISION 1
  #else
    #define AC_DOUBLE_PRECISION 0
  #endif

  #define LENTROPY (1) // TODO above
  #define LFORCING (1) // TODO above
  #define STENCIL_ORDER (6) // nghost is not 1, 2 or 3 (as it is not fetched from fortran yet). This causes the compilation to fail. TODO remove this line

  #define USER_PROVIDED_DEFINES
#endif
// clang-format on
