
#ifndef __IT_TYPES__
#define __IT_TYPES__

#include <stdint.h>


#ifdef __SIZEOF_FLOAT__
#define SIZE_FLOAT __SIZEOF_FLOAT__
#else
#define SIZE_FLOAT 4
#endif

#ifdef __SIZEOF_DOUBLE__
#define SIZE_DOUBLE __SIZEOF_DOUBLE__
#else
#define SIZE_DOUBLE 8
#endif

#ifdef __SIZEOF_LONG_DOUBLE__
#define SIZE_LONG_DOUBLE __SIZEOF_LONG_DOUBLE__
#else
#define SIZE_LONG_DOUBLE 16
#endif


namespace it {

    /* Define int type aliases */
    using i8  = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;

    using u8  = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    /* Define float type aliases */
    using f32  = float;
    using f64  = double;
    using f128 = long double;

    #if SIZE_DOUBLE == 4
    using f32 = double;
    #endif

    #if SIZE_LONG_DOUBLE == 8
    using f64 = long double;
    #endif

};


#endif