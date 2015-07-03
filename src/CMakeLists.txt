set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Magnum REQUIRED
    Shaders
    Sdl2Application)

include_directories(${MAGNUM_INCLUDE_DIRS} ${MAGNUM_APPLICATION_INCLUDE_DIRS})
include_directories(../lib/it)
link_directories(../lib/it)
link_libraries(ITLIB)

set(SOURCE_FILES main.cpp)
set(NEURAL_SRC neuralnet.h cost_functions.h util.h activation_functions.h layer.h neuron.h)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CORRADE_CXX_FLAGS} -std=c++14")

add_executable(ML ${SOURCE_FILES} ${NEURAL_SRC})

target_link_libraries(ML
    ${MAGNUM_APPLICATION_LIBRARIES}
    ${MAGNUM_LIBRARIES}
    ${MAGNUM_SHADERS_LIBRARIES})
