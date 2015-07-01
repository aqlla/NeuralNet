#include <iostream>
#include "util.h"
#include "neuralnet.h"



int main() {
    srand(static_cast<unsigned>(time(0)));

    NeuralNet net;

    net << InputLayer<1>{}
        << FullyConnectedLayer<4, activation::sigmoid>{}
        << FullyConnectedLayer<2, activation::tanh>{};

    net.update({.7, .243});
    net.update({.4, .6});

    return 0;
}



