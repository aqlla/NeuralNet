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
    size_t inputNeuronCount;

public:
    NeuralNet()
            : inputNeuronCount{0} {};
    ~NeuralNet() = default;

    template <size_t NeuronCount>
    void addInputLayer() {
        inputNeuronCount = NeuronCount;
        for (size_t i = 0; i < NeuronCount; ++i)
            inputs.push_back(make_shared<f64>(0));
        auto layer = make_shared<InputLayer<NeuronCount>>(inputs);
        layers.push_back(std::move(layer));
    };

    template <size_t NeuronCount, class ActivationFunc>
    void addHiddenLayer() {
        auto previousLayer = layers[layers.size()-1];
        auto layer = make_shared<FullyConnectedLayer<NeuronCount, ActivationFunc>>(previousLayer);
        layers.push_back(std::move(layer));
    };

    void update(vector<f64> &&values) {
        assert(values.size() == inputNeuronCount);
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

template <size_t NeuronCount, class ActivationFunc>
NeuralNet &operator <<(NeuralNet &net, FullyConnectedLayer<NeuronCount, ActivationFunc>  const& layer) {
    net.addHiddenLayer<NeuronCount, ActivationFunc>();
    return net;
};


#endif //ML_NEURALNET_H
