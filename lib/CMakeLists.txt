set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(ITLIB_PATH it)
include_directories(${ITLIB_PATH})

add_library(ITLIB
    it/ittypes.h
    it/itvector/vector.h
    it/itvector/vector.cpp
    it/itlib.h)