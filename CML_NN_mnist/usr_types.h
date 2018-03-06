/**
 *  file name:  usr_types.h
 *  function:   User Defined Types
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#ifndef __USR_TYPES_H
#define __USR_TYPES_H

#include <stdio.h>

typedef char            c8;
typedef unsigned char   u8;
typedef unsigned int    u32;
typedef int             i32;
typedef float           f32;
typedef double          d64;

///#define DEBUG

#ifdef DEBUG
    #define dprintf(fmt,args...)     printf(fmt,##args)
#else
    #define dprintf(fmt,args...)
#endif


#endif
