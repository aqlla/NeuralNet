// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_NEURALNET_H
#define ML_NEURALNET_H

#include "itlib.h"
#include <iostream>
#include <assert.h>
#include "layer.h"
#include "cost_functions.h"

template <class CostFunc>
class NeuralNet
{
private:
    std::vector<shared_ptr<LayerBase>> layers;
    std::vector<shared_ptr<f64>> inputs;

    size_t inputIndex;
    size_t inputCount;
    size_t outputCount;

    CostFunc costFunction;

public:
    NeuralNet(size_t inputCount = 0, size_t outputCount = 0)
        : inputCount{inputCount},
          outputCount{outputCount},
          inputIndex{0}
    {
        // Init random seed;
        srand(static_cast<unsigned>(time(0)));
    };

    ~NeuralNet() = default;


    NeuralNet<CostFunc>& setInputLayer(size_t neuronCount) {
        if (inputCount == 0)
            inputCount = neuronCount;
        for (size_t i = 0; i < neuronCount; ++i)
            inputs.push_back(make_shared<f64>(0));
        auto layer = make_shared<InputLayer>(inputCount, inputs);
        layers.push_back(std::move(layer));
        return *this;
    };

    template <class ActivationFunc>
    NeuralNet<CostFunc>& addHiddenLayer(size_t neuronCount) {
        auto previousLayer = layers[layers.size()-1];
        auto layer = make_shared<FullyConnectedLayer<ActivationFunc>>(neuronCount, previousLayer);
        layers.push_back(std::move(layer));
        return *this;
    };

    template <class ActivationFunc>
    NeuralNet<CostFunc>& setOutputLayer(size_t neuronCount) {
        if (outputCount == 0)
            outputCount = neuronCount;

        auto previousLayer = layers[layers.size()-1];
        auto layer = make_shared<OutputLayer<ActivationFunc>>(outputCount, previousLayer);
        layers.push_back(std::move(layer));
        return *this;
    };


    template <class T>
    NeuralNet<CostFunc>& forward(T t) {
        *inputs[inputIndex++] = t;
        propagate();
        return *this;
    };

    template <class T, class... Args>
    NeuralNet<CostFunc> & forward(T t, Args... args) {
        static constexpr int argc = sizeof...(Args) + 1;
        if (inputIndex == 0) assert(argc == inputCount);

        *inputs[inputIndex++] = t;
        return forward(args...);
    };

    NeuralNet<CostFunc>& forward(vector<f64> &values) {
        assert(values.size() == inputCount);
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

    void getCost(vector<f64> expected) {
        auto outputLayer = layers[layers.size() - 1];
        auto output = outputLayer->outputs;
        for (size_t i = 0; i < output.size(); ++i) {
            std::cout << "Cost " << i << ": "
                      << this->costFunction.f(*output[i], expected[i])
                      << std::endl;
        }

        std::cout << std::endl;
    }
};

template <class CostFunc>
NeuralNet<CostFunc> &operator <<(NeuralNet<CostFunc> &net, InputLayer const& layer) {
    return net.setInputLayer(layer.neuronCount);
};

template <class CostFunc, class ActivationFunc>
NeuralNet<CostFunc> &operator <<(NeuralNet<CostFunc> &net, FullyConnectedLayer<ActivationFunc> const& layer) {
    return net.template addHiddenLayer<ActivationFunc>(layer.neuronCount);
};

template <class CostFunc, class ActivationFunc>
NeuralNet<CostFunc> &operator <<(NeuralNet<CostFunc> &net, OutputLayer<ActivationFunc> const& layer) {
    return net.template setOutputLayer<ActivationFunc>(layer.neuronCount);
};


#endif //ML_NEURALNET_H
