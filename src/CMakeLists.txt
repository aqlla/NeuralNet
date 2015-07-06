set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(../lib/it)
link_directories(../lib/it)
link_libraries(ITLIB)

set(SOURCE_FILES main.cpp)
set(NEURAL_SRC neuralnet.hpp cost_functions.hpp util.h activation_functions.hpp neuron.hpp neuron.cpp synapse.hpp synapse.cpp)

add_executable(ML ${SOURCE_FILES} ${NEURAL_SRC} layer.hpp layer.cpp connected_layer.hpp)
