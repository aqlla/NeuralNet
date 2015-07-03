
#include <iostream>
#include "neuralnet.h"



int main() {
    NeuralNet net;
    net << InputLayer{2}
        << FullyConnectedLayer<activation::sigmoid>{3};
    net.forward(.7, .44);

    return EXIT_SUCCESS;
}
