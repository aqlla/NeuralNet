// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_NEURALNET_H
#define ML_NEURALNET_H

#include "itlib.h"
#include <iostream>
#include "layer.h"
#include "cost_functions.h"

class NeuralNet
{
public:
    NeuralNet() = default;
    ~NeuralNet() = default;

    void addLayer(size_t numberOfNeurons) {
        for (int i = 0; i < numberOfNeurons; ++i)
            inputs.emplace_back(new double);
        std::unique_ptr<InputLayer<>> l{new InputLayer<>(numberOfNeurons, inputs)};
        layers.push_back(std::move(l));
    }

    void update(std::vector<f64> in) {
        for (int i = 0; i < in.size(); ++i) {
            *inputs[i] = in[i];
            layers[0].get()->neurons[i]->setOutput();
        }
    };

private:
    std::vector<std::unique_ptr<_BaseLayer>> layers;
    std::vector<f64*> inputs;
};



#endif //ML_NEURALNET_H
