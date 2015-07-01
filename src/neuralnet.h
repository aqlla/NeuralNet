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
private:
    std::vector<unique_ptr<LayerBase>> layers;
    std::vector<f64*> inputs;

public:
    NeuralNet()
        : inputs{
            (f64 *) calloc(sizeof(f64), 1),
            (f64 *) calloc(sizeof(f64), 1)} {};
    ~NeuralNet() = default;

    template <size_t NeuronCount>
    void addInputLayer() {
        auto layer = make_unique<InputLayer<NeuronCount>>(inputs);
        layers.push_back(std::move(layer));
    };

    template <size_t NeuronCount, class ActivationFunction>
    void addHiddenLayer() {
        auto layer = make_unique<FullyConnectedLayer<NeuronCount, ActivationFunction>>(layers[layers.size()-1].get());
        layers.push_back(std::move(layer));
    };

    void update(std::vector<f64> &&values) {
        for (int i = 0; i < inputs.size(); ++i)
            *inputs[i] = values[i];
        for (auto &layer : layers)
            layer->forward();
    };
};


template <size_t NeuronCount>
NeuralNet &operator <<(NeuralNet &net, InputLayer<NeuronCount>  const& layer) {
    net.addInputLayer<NeuronCount>();
    return net;
};

template <size_t NeuronCount, class ActivationFunction>
NeuralNet &operator <<(NeuralNet &net, FullyConnectedLayer<NeuronCount, ActivationFunction>  const& layer) {
    net.addHiddenLayer<NeuronCount, ActivationFunction>();
    return net;
};


#endif //ML_NEURALNET_H
