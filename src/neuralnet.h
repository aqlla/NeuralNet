// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_NEURALNET_H
#define ML_NEURALNET_H

#include "itlib.h"
#include <iostream>
#include <assert.h>
#include "layer.h"
#include "cost_functions.h"

class NeuralNet
{
private:
    std::vector<shared_ptr<LayerBase>> layers;
    std::vector<shared_ptr<f64>> inputs;

    size_t inputIndex;
    size_t inputNeuronCount;
    size_t outputNeuronCount;

public:
    NeuralNet()
        : inputNeuronCount{0},
          inputIndex{0} {
        // Init random seed;
        srand(static_cast<unsigned>(time(0)));
    };

    ~NeuralNet() = default;


    NeuralNet& addInputLayer(size_t neuronCount) {
        inputNeuronCount = neuronCount;
        for (size_t i = 0; i < neuronCount; ++i)
            inputs.push_back(make_shared<f64>(0));
        auto layer = make_shared<InputLayer>(neuronCount, inputs);
        layers.push_back(std::move(layer));
        return *this;
    };

    template <class ActivationFunc>
    NeuralNet& addHiddenLayer(size_t neuronCount) {
        auto previousLayer = layers[layers.size()-1];
        auto layer = make_shared<FullyConnectedLayer<ActivationFunc>>(neuronCount, previousLayer);
        layers.push_back(std::move(layer));
        return *this;
    };

    template <class T>
    NeuralNet& forward(T t) {
        *inputs[inputIndex++] = t;
        propagate();
        return *this;
    };

    template <class T, class... Args>
    NeuralNet& forward(T t, Args... args) {
        static constexpr int argc = sizeof...(Args) + 1;
        if (inputIndex == 0) assert(argc == inputNeuronCount);

        *inputs[inputIndex++] = t;
        return forward(args...);
    };

    NeuralNet& forward(vector<f64> &&values) {
        assert(values.size() == inputNeuronCount);
        for (size_t i = 0; i < inputs.size(); ++i)
            *inputs[i] = values[i];
        propagate();
        return *this;
    };

    void propagate() {
        for (auto &layer : layers) {
            layer->forward();
        }
    };
};

NeuralNet &operator <<(NeuralNet &net, InputLayer const& layer) {
    net.addInputLayer(layer.neuronCount);
    return net;
};

template <class ActivationFunc>
NeuralNet &operator <<(NeuralNet &net, FullyConnectedLayer<ActivationFunc> const& layer) {
    net.addHiddenLayer<ActivationFunc>(layer.neuronCount);
    return net;
};


#endif //ML_NEURALNET_H
