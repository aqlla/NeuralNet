
#include <iostream>
#include "neuralnet.hpp"

int main() {
    vector<vector<f64>> inputs {
            {3, 5},
            {5, 1},
            {10, 2},
    };

    vector<vector<f64>> expected {
            {75},
            {82},
            {93},
    };

    vector<f64> e2 { 75, 82, 93 };


    vector<vector<vector<f64>>> trainingData{inputs, expected};

    NeuralNet<cost::mse> net;
    net << InputLayer{2}
        << FullyConnectedLayer<activation::sigmoid>{2}
        << OutputLayer<activation::sigmoid>{1};

    for (size_t i = 0; i < inputs.size(); ++i) {
        net.forward(inputs[i])
           .getCost(expected[i]);
    }

    return EXIT_SUCCESS;
}
