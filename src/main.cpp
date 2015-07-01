#include <iostream>
#include "util.h"
#include "neuralnet.h"



int main() {
    srand(static_cast<unsigned>(time(0)));

    NeuralNet net;

    net << InputLayer<5>{}
        << FullyConnectedLayer<150, activation::sigmoid>{}
        << FullyConnectedLayer<150, activation::sigmoid>{}
        << FullyConnectedLayer<150, activation::sigmoid>{};

    net.update({.7, .9, .2, .9, .2});
    net.update({.6, .9, .23, .2, .1});

    return 0;
}



