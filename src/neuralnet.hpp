// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.

#ifndef ML_NEURALNET_H
#define ML_NEURALNET_H

#include <iostream>
#include <assert.h>
#include <functional>

#include "layer.hpp"
#include "connected_layer.hpp"
#include "cost_functions.hpp"

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

    auto setInputLayer(const size_t neuronCount) -> NeuralNet<CostFunc>& {
        if (inputCount == 0)
            inputCount = neuronCount;
        for (size_t i = 0; i < neuronCount; ++i)
            inputs.push_back(make_shared<f64>(0));
        auto layer = make_shared<InputLayer>(inputCount, inputs);
        layers.push_back(std::move(layer));
        return *this;
    };

    template <class ActivationFunc>
    auto addHiddenLayer(const size_t neuronCount) -> NeuralNet<CostFunc>& {
        auto layer = make_shared<FullyConnectedLayer<ActivationFunc>>(neuronCount, layers.back());
        layers.push_back(std::move(layer));
        return *this;
    };

    template <class ActivationFunc>
    auto setOutputLayer(const size_t neuronCount) -> NeuralNet<CostFunc>& {
        if (outputCount == 0)
            outputCount = neuronCount;
        auto layer = make_shared<OutputLayer<ActivationFunc>>(outputCount, layers.back());
        layers.push_back(std::move(layer));
        return *this;
    };


    template <class T>
    auto forward(const T t) -> NeuralNet<CostFunc>& {
        *inputs[inputIndex++] = t;
        propagate();
        return *this;
    };

    template <class T, class... Args>
    auto forward(const T t, const Args... args) -> NeuralNet<CostFunc>& {
        static constexpr int argc = sizeof...(Args) + 1;
        if (inputIndex == 0) assert(argc == inputCount);

        *inputs[inputIndex++] = t;
        return forward(args...);
    };

    auto forward(const vector<f64> &values) -> NeuralNet<CostFunc>& {
        assert(values.size() == inputCount);
        for (size_t i = 0; i < inputs.size(); ++i)
            *inputs[i] = values[i];
        propagate();
        return *this;
    };

    auto propagate() -> NeuralNet<CostFunc>& {
        for (auto &layer : layers)
            layer->forward();
        return *this;
    };

    auto getCost(const vector<f64> desiredOutput) -> NeuralNet<CostFunc>& {
        auto outLayer = layers.back();
        for (size_t i = 0; i < outLayer->size(); ++i) {
            f64 desired = normalizeOutput(desiredOutput[i], 0, 100);
            f64 cost = costFunction.f((*outLayer)[i], desired);

            std::cout << "Output " << i << ": \n"
                      << "\tExpected: " << desired << std::endl
                      << "\tResult:   " << (*outLayer)[i] << std::endl
                      << "\tCost:     " << cost << std::endl
                      << "\tLocal G:  " << (*outLayer).neurons[i]->calculateGrad(desired)
                      << std::endl;
        }
        std::cout << std::endl;
        return *this;
    };

    auto normalizeOutput(f64 x, f64 in_min, f64 in_max) -> f64 {
        return (x - in_min) / (in_max - in_min);
    };
};




template <class CostFunc>
auto operator <<(NeuralNet<CostFunc> &net, InputLayer const& layer) -> NeuralNet<CostFunc>& {
    return net.setInputLayer(layer.size());
};

template <class CostFunc, class ActivationFunc>
auto operator <<(NeuralNet<CostFunc> &net, FullyConnectedLayer<ActivationFunc> const& layer) -> NeuralNet<CostFunc>& {
    return net.template addHiddenLayer<ActivationFunc>(layer.size());
};

template <class CostFunc, class ActivationFunc>
auto operator <<(NeuralNet<CostFunc> &net, OutputLayer<ActivationFunc> const& layer) -> NeuralNet<CostFunc>& {
    return net.template setOutputLayer<ActivationFunc>(layer.size());
};


#endif //ML_NEURALNET_H
