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
  #define USER_PROVIDED_DEFINES
#endif

