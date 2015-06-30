set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(../lib/it)
link_directories(../lib/it)
link_libraries(ITLIB)

set(SOURCE_FILES main.cpp)
set(NEURAL_SRC neuralnet.h cost_functions.h util.h activation_functions.h layer.h neuron.h)
add_executable(ML ${SOURCE_FILES} ${NEURAL_SRC})