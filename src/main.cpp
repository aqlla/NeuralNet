#include <iostream>
#include "util.h"
#include "neuralnet.h"



int main() {
    srand(static_cast<unsigned>(time(0)));

    NeuralNet net{};
    net.addLayer(3);
    net.update({.1, .2, .3});
    net.update({.6, .7, .8});

    return 0;
}



